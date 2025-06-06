###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ante Bilic                          #
# Email: ante.bilic.mr@gmail.com                  #
# Status: N/A                                     #
###################################################

"""
This module implements various policy and pathway nodes (inspections, decisions, treatment).
"""
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Callable, Union
from string import ascii_lowercase
import numpy as np
from numpy.random import Generator
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from loguru import logger
import iteround
from flatpack.src.inspectors import SamplingRegime

logger.remove()
logger.add(sys.stderr, level="INFO")

type ArrayLike[T] = np.ndarray[T] | Series | DataFrame
type CallerNode = Union["DecisionNode", "PathwayNode", "DecisionNodeNumSplit", "TreatmentNode"]


def rn_random() -> float:
    """
    Get a random float from the [0.0, 1.0) interval

    Returns
    -------
    ...:  A random number from the [0.0, 1.0) interval

    """
    rng: Generator = np.random.default_rng()
    return rng.random()


@dataclass(kw_only=True)
class PolicyNode:
    """
    The base class of all nodes (Root, Pathway, Treatment, Decision...).
    Provides the record() method, inherited by all Nodes, which records the direction
    history applied to each of the virtual containers.

    Parameters:
    ----------
    name : str
        The formal name assigned to it.

    """
    name: str
    original_name: str = field(default="")

    def __post_init__(self) -> None:
        """
        Simply for convenience, to assign the original_name the same value as the name
        """
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.original_name:
            self.original_name = self.name
            logger.debug(f"\n{type(self)}, original_name: {self.original_name}")


    # record that containers picked out by "mask" went to this node
    def record(self, containers: DataFrame, mask: ArrayLike[bool], msg: str | None=None) -> None:
        """
        Inheroted and used by all Node types to keep track of the directions applied to each
        virtual container as they move through the policy network (i.e., the flowchart).

        Parameters
        ----------
        containers : DataFrame
            The table whose rows correspond to the virtual containers.
        mask : 1-D array-like sequence
            Either True or False value for each virtual container.
        msg : Optional[str], optional
            The direction being applied to a container (if any). The default is None.

        Returns
        -------
        None.

        """
        # Add new column 'directions' and set it to [] for each row:
        if not 'directions' in containers.columns:
            containers['directions'] = [[] for _ in range(containers.shape[0])]
        if not msg:
            msg = f"Node: {self.name}"
        # For rows the mask is True append msg to the 'directions' column values (these are lists)
        new_directs = [(x + [msg] if m else x) for x, m in zip(containers['directions'], mask)]
        containers['directions'] = new_directs


@dataclass(kw_only=True)
class RootNode(PolicyNode):
    """
    The starting point for any policy, typically with the top_node a DecisionNode

    Parameters
    ----------
    top_node : PolicyNode
        Usually a DecisionNode kind, to split the CAL & non-CAL containers.

    """
    top_node: CallerNode

    def __post_init__(self) -> None:
        """
        Ensure the right type of top_node.
        Normally a DecisionNode, which is yet to be defined. PolicyNode tested instead.
        """
        if not isinstance(self.top_node, PolicyNode):
            raise TypeError("top_node must be a Decision/PathwayNode")


    def __call__(self, containers: DataFrame) -> None:
        """
        Assigns all containers the mask=True and a new column 'detected', initially False
        (no contamination undetected) .Then passes the containers to the top_node
        (commonly for CAL vs non-CAL separation)

        Parameters
        ----------
        containers : DataFrame
            The table of virtual containers with their attributes

        Returns
        -------
        None

        """
        mask = Series([True for _ in range(containers.shape[0])])
        # add column 'directions' with [] values and append the message msg to those:
        self.record(containers, mask, msg=f'Processed under policy "{self.name}"')
        # add column 'detected' with False values:
        containers['detected'] = Series([False for _ in range(containers.shape[0])])
        # call the top node (usually PathwayNode) with the containers table and mask
        self.top_node(containers, mask)


@dataclass(kw_only=True)
class DecisionNode(PolicyNode):
    """
    A node that divides the containers by calling a decision function to update the mask.

    Parameters
    ----------
    decision_func : Callable
        Needs to return a list of pairs (mask, node) values for each container,
        which updates the existing and dictates the next destination in the pathway.

    """
    decision_func: Callable

    def __post_init__(self) -> None:
        """
        Simply testing the argument type
        """
        if not hasattr(self.decision_func, "__call__") or not callable(self.decision_func):
            raise TypeError("decision_func must be callable")


    def __call__(self, containers: DataFrame, mask: ArrayLike[bool]) -> None:
        """
        Divides the containers by calling a decision function to update the mask.
        Then, with the new mask, the containers are directed propagated by calling
        the designated nodes.

        Parameters
        ----------
        containers : DataFrame
            The table of virtual containers with their attributes
        mask : 1-D array-like sequence
            Either True or False value for each virtual container.

        Returns
        -------
        None

        """
        self.record(containers, mask)
        mask_node_pairs = self.decision_func(containers)
        for _m, node in mask_node_pairs:
            new_mask = _m & mask  # elementwise logical 'and' between values of two masks
            node(containers, new_mask)


@dataclass(kw_only=True)
class DecisionNodeNumSplit(PolicyNode):
    """
    A node that divides the containers based on a number (or fraction). Only the required
    number (or propoprtion) of randomly selected containers are directed to the yes_node.
    This would typically be for the Cargo Compliance Verification (CCV) sampling of the
    low risk (non-CAL) containers.

    Parameters
    ----------
    yes_node : CallerNode
        Where the required volume of containers need to be directed to.
    no_node : CallerNode
        Where the rest of the containers need are directed to.
    number : int | float
        The required volume (whole number or proportion) of containers to send to the yes_node

    """
    yes_node: CallerNode
    no_node: CallerNode
    number: int | float

    def __post_init__(self) -> None:
        """
        Simply testing the argument types
        """
        if not all(isinstance(_v, PolicyNode) for _v in (self.yes_node, self.no_node)):
            raise TypeError("The yes/no_node parameters must be Pathway/Decsion/TreatmentNodes")
        if not isinstance(self.number, float | int):
            raise TypeError("The 'number' must be a positive number")
        if self.number < 0:
            raise ValueError("The 'number' must be POSITIVE")


    def __call__(self, containers: DataFrame, mask: ArrayLike[bool]) -> None:
        """
        Divides the containers so that the required volume is sent to the yes_node
        and the rest to the no_node.
        Then, with the new mask, the containers are directed propagated by calling
        the designated nodes.

        Parameters
        ----------
        containers : DataFrame
            The table of virtual containers with their attributes
        mask : 1-D array-like sequence
            Either True or False value for each virtual container.

        Returns
        -------
        None

        """
        self.record(containers, mask)
        num_containers = sum(mask)
        frac = self.number / num_containers
        # in case there aren't enough containers (i.e., frac > 1) for CCV:
        frac = min(frac, 1)
        ccv_flag = np.random.binomial(1, frac, containers.shape[0]) == 1
        yes_mask = mask & ccv_flag
        no_mask = mask & ~ccv_flag
        self.yes_node(containers, yes_mask)
        self.no_node(containers, no_mask)


@dataclass(kw_only=True)
class PathwayNode(PolicyNode):
    """
    This kind of node perform the sampling of containers (for inspections) in the designated way.

    Parameters
    ----------
    sampler : SamplingRegime
        The required sampler of containers.
    child_node : CallerNode
        Where the the containers are directed to after the sampling is completed and mask updated.
    mixed_sensitivities : dict[str, tuple[float, float]]
        For inspections carried out by DAFF and AA (in the flowchart after the blue decision node
        "Inspect by AA?") specify sensitivity & proportion e.g. {'DAFF': (0.9, 0.2)'AA': (0.6, 0.8)}

    Other fields:
    leakages: int
        The count of containers still remaining undetected.
    fixed: int
        The count of successfully treated (i.e., cleaned) containers.
    """

    sampler: SamplingRegime
    child_node: CallerNode | None = None
    leakages: int = field(init=False, default=0)
    seen: int = field(init=False, default=0)
    mixed_sensitivities: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        """
        ensure two kinds of sensitivities match
        """
        if not isinstance(self.sampler, SamplingRegime):
            raise TypeError("The sampler must be a Sampler type")
        if not isinstance(self.child_node, PolicyNode | None):
            raise TypeError("The child_node must be a DecisionNode")
        if not isinstance(self.mixed_sensitivities, dict | None):
            raise TypeError("The format of mixed_sensitivities is wrong")

        # the following line is NECESSARY to assign the self.original_name the proper value:
        super().__post_init__()
        if self.mixed_sensitivities:
            # the effective sensitivity is the weighted (by respective proportions) average:
            self.sampler.sensitivity = sum(s * p for s, p in self.mixed_sensitivities.values())


    def __call__(self, containers: DataFrame, mask: ArrayLike[bool]) -> None:
        """
        Divides the containers so that the required volume is sent to the yes_node
        and the rest to the no_node.
        Then, with the updated mask, the containers are directed propagated by calling
        the designated nodes.

        Parameters
        ----------
        containers : DataFrame
            The table of virtual containers with their attributes
        mask : 1-D array-like sequence
            Either True or False value for each virtual container.

        Returns
        -------
        None

        """
        self.record(containers, mask)

        init_count: int = self.sampler.contamination_count
        # need to exclude contaminated containers that are already detected and should be clean:
        init_previously_detected: int = containers.loc[mask, 'detected'].sum()
        # run the inspection procedure on the containers, based on the selected sampling:
        self.sampler.inspect_batch(containers, mask)
        final_count: int = self.sampler.contamination_count

        contam_tot: int= containers.loc[mask, 'contamination'].isin(self.sampler.contam_names).sum()

        # the contamination detected at this node = final_count - init_count
        self.seen += sum(mask)
        self.leakages += (contam_tot - init_previously_detected) - (final_count - init_count)
        # run the child node if present:
        if self.child_node:
            self.child_node(containers, mask)


@dataclass(kw_only=True)
class TreatmentNode(PolicyNode):

    """
    The Node for cleaning of contaminated containers (if detected at a PathwayNode) with a presumed
    effectiveness.

    Parameters
    ----------
    effectiveness: float
        The presumed probability that the contamination will be removed by the treatment.
    child_node: PathwayNode
        Where the the containers are directed to after the sampling is completed and mask updated.

    Other fields:
    contam_names: Tuple[str, ...]
        Strings which designate some form of container external contamination e.g. "yes", "BRM etc
    seen: int
        The total count of containers at this node.
    fixed: int
        The count of successfully treated (i.e., cleaned) containers.
    """

    effectiveness: float
    child_node: CallerNode
    contam_names: tuple[str, ...] = field(init=False, default=('LLC', 'HLC', 'Pest', 'yes'))
    seen: int = field(init=False, default=0)
    fixed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """
        Simply test the parameter types
        """
        if not isinstance(self.effectiveness, float | int):
            raise TypeError("The effectiveness must be a number")
        if (self.effectiveness < 0) or (self.effectiveness > 1):
            raise ValueError("effectiveness value needs to be in the (0, 1) range")
        if not isinstance(self.child_node, PathwayNode):
            raise TypeError("The child_node must be a PathwayNode")


    def __call__(self, containers: DataFrame, mask: ArrayLike[bool]) -> None:
        """
        Performs the cleaning of contaminated containers with the presumed effectiveness.
        Then it directs the containers with the same mask to the child node.

        Parameters
        ----------
        containers : DataFrame
            The table of virtual containers with their attributes
        mask : 1-D array-like sequence
            Either True or False value for each virtual container.

        Returns
        -------
        None

        """
        self.record(containers, mask)
        self.seen += sum(mask)

        for i, cont in containers.loc[mask, :].iterrows():
            if cont['contamination'] in self.contam_names \
                    and rn_random() < self.effectiveness:
                containers.loc[i, 'contamination'] = "no"
                self.fixed += 1
        # run the child node:
        self.child_node(containers, mask)


def summarise_mixed(node: PathwayNode) -> dict[str, dict[str, int | float]]:
    """
    The output (for the nodes_{...}.csv file) from mixed sensitivity Nodes, e.g.
    containers, inspections, contamination found, leakage counts:
    1a. (SCHS) CSP-3 inspection - DAFF,43,43,0,0
    1b. (SCHS) CSP-3 inspection - AA,29,29,0,0
    1c. (SCHS) CSP-3 inspection - release,8,0,0,0
    3. CAL-medium DAFF inspection,212,212,2,0
    4. CAL-medium AA inspection,129,129,0,0

    Parameters
    ----------
    node : PathwayNode
        A mixed Node (e.g., DAFF/AA simultaneously performing sampling & inspections)

    Returns
    -------
    out_nodes : Dict[str, Dict[str, int]]
        The quantity name ('containers', 'inspections', 'contamination found', 'leakage') and count.
        for each of the mixed Nodes (their name is the 1st str).

    """
    # The dictionary of node dictionaries:
    out_nodes = {}
    letter_i: int = 0
    new_node_name: str = ""
    # Containers, inspections, contaminations_found and leakages for each agency (AA, DAFF, etc):
    this_node: dict[str, float]
    for name, (sens, prop) in node.mixed_sensitivities.items():
        this_node = {}
        this_node['containers'] = node.sampler.inspection_count * prop
        this_node['inspections'] = node.sampler.inspection_count * prop
        # P(detected by this node | detected) = P(sent to this node AND detected) / P(detected),
        # P(detected) = overall Node sensitivity (ie., sum(sens * prop)) = node.sampler.sensitivity
        prob_detected_here: float = sens * prop / node.sampler.sensitivity
        this_node['contamination found'] = node.sampler.contamination_count * prob_detected_here
        # node.sampler.inspection_count / node.seen  is  e.g. 54321 / 100k  and
        # prob that container got sent here is the agency prop times this ratio
        try:
            prob_sent_here: float = prop * node.sampler.inspection_count / node.seen
        except ZeroDivisionError as _e:
            # Error triggerred for CSP3 (instead of SCHS( when all CALMedium containers are non-SCHS
            logger.error(f"\nFor node {node} node.seen = {node.seen}:\n{_e}")
            # In this case node.sampler.inspection_count is also 0, so prob_sent_here = 0:
            prob_sent_here = 0.0
        # leakage = num_contaminated sent to this node minus num_contamination_found here:
        this_node['leakage'] = (node.sampler.contamination_count + node.leakages) * \
            prob_sent_here - this_node['contamination found']
        # Now we label each agency/out_node (e.g., AA, DAFF, etc) with "a", "b", "c" etc:
        node_name_split: list[str]
        if len(node_name_split := node.name.split(".")) == 2:
            # update each ordinal number in the Node name with a new letter from the alphabet:
            new_node_name = f"{node_name_split[0]}{ascii_lowercase[letter_i]}.{node_name_split[1]}"
        else:
            new_node_name = node.name
        # Combine the new_node_name and the agency name (AA, DAFF,...) into a new key:
        out_nodes[f'{new_node_name} - {name}'] = this_node
        letter_i += 1

    # Finally, the containers released without inspection and the associated leakage:
    if len(node_name_split := node_name_split) == 2:
        new_node_name = f"{node_name_split[0]}{ascii_lowercase[letter_i]}.{node_name_split[1]}"
    else:
        new_node_name = node.name
    out_nodes[f'{new_node_name} - release'] = {
        'containers': float(node.seen - node.sampler.inspection_count),
        'inspections': 0.0,
        'contamination found': 0.0,
        'leakage': node.leakages - sum(n['leakage'] for n in out_nodes.values())
        }
    # The dict of node dicts to DataFrame:
    out_nodes_df = pd.DataFrame(out_nodes).T
    # Simply round the 4 cols (containers, inspections, contaminations_found, leakage) to int:
    for colname in out_nodes_df.columns:
        out_nodes_df[colname] = list(map(int, iteround.saferound(out_nodes_df[colname], 0)))
    # Update the dict with these values:
    out_nodes = out_nodes_df.T.to_dict()

    return out_nodes


def summarise_inspection(nodes_dict: dict[str, PathwayNode | TreatmentNode]
                         ) -> tuple[DataFrame, DataFrame]:
    """
    The output (for the summary_{...}.csv file) from the inspection and treatment nodes
    containers, inspections, contamination found, leakage, treated counts.

    Parameters
    ----------
    nodes_dict : Dict[str, PathwayNode | TreatmentNde]
        The set of

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        The summary tables from the Pathway and Treatment Nodes

    """
    pathway_dict: dict[str, dict[str, int]] = {}
    for node_name, node in nodes_dict.items():
        if isinstance(node, PathwayNode):
            if node.mixed_sensitivities:
                pathway_dict.update(summarise_mixed(node))
                continue
            pathway_dict[node_name] = {'containers': node.seen,
                                       'inspections': node.sampler.inspection_count,
                                       'contamination found': node.sampler.contamination_count,
                                       'leakage': node.leakages}
    # dict to DataFrame:
    pathway_df = pd.DataFrame(pathway_dict).T
    pathway_df.index.rename('node', inplace=True)
    with pd.option_context('display.max_columns', None):
        logger.info(f"\npathway_df:\n{pathway_df}")
    treatment_dict: dict[str, dict[str, int]] = {}
    for node_name, node in nodes_dict.items():
        if isinstance(node, TreatmentNode):
            treatment_dict[node_name] = {'containers': node.seen, 'treated': node.fixed}
    treatment_df = pd.DataFrame(treatment_dict).T
    treatment_df.index.rename('node', inplace=True)
    logger.info(f"\ntreatment_df:\n{treatment_df}")

    return pathway_df, treatment_df


def create_policy(policy_name: str,
                  top_node: CallerNode,
                  node_list_for_detailed_output: list[PathwayNode | TreatmentNode]
                  ) -> tuple[str, RootNode, dict[str, PathwayNode | TreatmentNode]]:
    """
    Generates the named sequence of Pathway/TreatmentNodes (without the associated DecisionNodes)
    which contribute to the flowchart from the conceptual model. The top node is a RootNode.

    Parameters
    ----------
    policy_name : str
        Typically, the name is shared with that of the output directory from the YAML input
    top_node : CallerNode
        Normally the first DecisionNode based on the CAL attribute value (True or False)
    node_list_for_detailed_output : List[PathwayNode | TreatmentNode]
        Normally, the full list of Pathway & Treatment Nodes in the flowchart

    Returns
    -------
    Tuple[str, RootNode, Dict[str, PathwayNode]]
        The policy_name, the RootNode named policy_name with the top_node, and a dict
        created from external_risks_policy_new

    """
    # list to a dict using the "original_name" node values as the keys and nodes as their values:
    other_node_dict = {getattr(node, 'original_name'): node
                           for node in node_list_for_detailed_output}
    # also set up a RootNode for the top node and return it together with the dict:
    return policy_name, RootNode(name=policy_name, top_node=top_node), other_node_dict


def simulate_policy(simulation_name: str,
                    root_node: RootNode,
                    other_nodes: dict[str, PathwayNode | TreatmentNode],
                    cs_fname: Path | str,
                    save_results: bool=True,
                    save_container_results: bool=True,
                    save_summary: bool=True,
                    output_dir: Path | str=Path.cwd() / "results" / "raw"
                    ) -> tuple[DataFrame, DataFrame]:
    """
    It carries out a single simulation for the designated external container pathway policy.

    Parameters
    ----------
    simulation_name : str
        Typically, the same as policy_name and that of the output directory from the YAML input
    root_node : RootNode
        The start node with a name (normally policy_name) & its top_node (normally CAL DecisionNode)
    other_nodes : Dict[str, PathwayNode | TreatmentNode]
        Normally all the inspection & treatment nodes with their names as the key
    cs_fname : Union[Path, str]
        The CSV table whose rows correspond to the virtual containers.
    save_results : bool, optional
        Whether to save Node summary as the nodes_{...}.csv file. The default is True.
    save_container_results : bool, optional
        The default is True for single-simulation runs and False for multi-simulations
    save_summary : bool, optional
        The default is True.
    output_dir : Union[Path, str], optional
        The default is Path("results") / "raw" (or "multi").

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        The summary tables from the Pathway and Treatment Nodes

    """
    # read the CSV file with containers attributes generated by containers.csv:
    cs_df = pd.read_csv(cs_fname, index_col=0)
    # in case 'CALSourceCountryMedium' not in the columns, add it and set to [False]:
    if not 'CALSourceCountryMedium' in cs_df.columns:
        cs_df.insert(cs_df.shape[1], 'CALSourceCountryMedium', False)
    # the number of contaminated containers in the CSV file:
    start_contam = cs_df['contamination'].isin(['LLC', 'HLC', 'Pest', 'yes']).sum()
    # Now root node will call PolicyNode record() with a mask and message (as the the start of
    # 'directions' column), then set the 'detected' column to False, and finally call the top node:
    root_node(cs_df)
    # The updated number of contaminated containers:
    total_contam = cs_df['contamination'].isin(['LLC', 'HLC', 'Pest', 'yes']).sum()
    detected = cs_df['detected'].sum()
    logger.info(f"\nDetected {detected} contaminated containers of {total_contam} total.")
    logger.info(f"\nNumber of successful treatments is {start_contam - total_contam}")

    pathway_df, treatment_df = summarise_inspection(other_nodes)

    if save_results is True:
        combined = pd.concat([pathway_df, treatment_df])
        if save_container_results:
            cs_df.to_csv(Path.cwd() / output_dir / f"containers_{simulation_name}.csv")
        combined.to_csv(Path.cwd() / output_dir / f"nodes_{simulation_name}.csv")

    if save_summary:
        total_containers = cs_df.shape[0]
        total_contamination = start_contam
        total_inspections = pathway_df['inspections'].sum()
        contamination_found = detected
        if treatment_df.shape[0] > 0:
            total_treatments = treatment_df['containers'].sum()
            treated_contamination = treatment_df['treated'].sum()
        else:
            total_treatments = 0
            treated_contamination = 0

        leakage = total_contamination - contamination_found - treated_contamination
        summary_data = {'total containers': [total_containers],
                        'total contamination': [total_contamination],
                        'total inspections': [total_inspections],
                        'total treatments': [total_treatments],
                        'contamination found': [detected],
                        'contamination treated': [treated_contamination],
                        'contamination leakage': [leakage]}

        df_out = pd.DataFrame(summary_data)
        df_out.to_csv(Path.cwd() / output_dir /f"summary_{simulation_name}.csv", index=False)

    return pathway_df, treatment_df


if __name__ == '__main__':
    """
    Below is a simple demo of this module for the external container inspections.

    """
    from flatpack.src.fun_decision import cbis_node_mixed
    # testing cbis_mixed, define mixed (sensitivity, proportion) pairs:
    senss = {'AA': (0.4, 0.7), 'DAFF': (0.9, 0.3)}
    # create a PathwayNode with a CSP1Sampler and overall sensitivity 0.55:
    mixed_inspect = cbis_node_mixed('1. CBIS', senss, 10, 0.5, csp3=False)
    # The associated RootNode and single-item dictionary:
    _, root, node_dict = create_policy('tester', mixed_inspect, [mixed_inspect])
    # get one of the container attribute files generated by containers.py
    csv_file = Path("containers") / \
        "original_ext_int_containers_external_with_medium.csv"
    out_dir = Path.cwd() / "results" / "test"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    pathway, treatment = simulate_policy('Test mixed',
                                         root,
                                         node_dict,
                                         save_results=True,
                                         save_container_results=True,
                                         save_summary=True,
                                         cs_fname=csv_file,
                                         output_dir=out_dir)
