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
Utility functions for a more convenient usage of various PolicyNodes

"""
from functools import partial
from enum import Enum
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from flatpack.src.policy import DecisionNode, PathwayNode, DecisionNodeNumSplit, TreatmentNode
from flatpack.src.inspectors import (get_num_ccv_samples,
                                      BasicSampler,
                                      RandomSampler,
                                      CSP3Sampler,
                                      CSP1Sampler)

type CallerNode = DecisionNode | PathwayNode | DecisionNodeNumSplit | TreatmentNode


def apply_tests(cs_df: DataFrame,
                tests: dict[str, str | bool],
                yes_node: CallerNode,
                no_node: CallerNode,
                how: str="all"
                ) -> list[tuple[Series, CallerNode]]:
    """
    Provides a way to define the decision_func in the DecisionNode by splitting the virtual
    containers based on the values of multiple attributes (see multiple_attrs_decision_node)

    Parameters
    ----------
    cs_df : DataFrame
        The table whose rows correspond to the virtual container attributes
    tests : Dict[str, Union[str, bool]]
        {attribute: value, ...}. An example: {'RuralDest': True, 'TypeOfGoods': 'normal'}
    yes_node : CalleryNode
        The node where the containers that satisfy the tests are directed to.
    no_node : CalleryNode
        The node where the containers that do not satisfy the tests are directed to.
    how : str, optional
        Flag that indicates how the separate sets are applied. The default is 'all'.

    Raises
    ------
    ValueError
        In case a silly "how" value is passed.

    Returns
    -------
    List[Tuple[Series, CallerNode]]
        The two mutually complementary boolean Series of the containers which are being directed
        (those with True values) to the associated Nodes (either yes_node or no_node).

    """
    # Create a table whose boolean columns are the keys from the dict tests and
    # the values can be found in the cs_df Series named with the same key.
    # True values are in the rows that match, False values in the rest of rows:
    tests_df = pd.DataFrame({_k: cs_df[_k] == val for _k, val in tests.items()})
    # Aggregate the True values (either by "all" or "any") for the rows as a bool Series yes_mask:
    if how == "all":
        yes_mask = tests_df.all(axis=1)
    elif how == "any":
        yes_mask = tests_df.any(axis=1)
    else:
        raise ValueError("apply_tests: 'how' must be either 'all' or 'any'.")

    return [(yes_mask, yes_node), (~yes_mask, no_node)]


def keys_to_tuples(a_dict: dict) -> dict:
    """
    Simply converts dictionary keys to single-element tuples (in case they are not already).
    Used only in decide_on_attr() below.
    For example: {True: SomeNode, "no": AnotherNode} -> {(True,): SomeNode, ("no",): AnotherNode}

    Parameters
    ----------
    a_dict : Dict
        Any dict will do

    Returns
    -------
        : Dict
        The key-tuplerised new dict.

    """
    return {_k if isinstance(_k, tuple) else (_k,): _v for _k, _v in a_dict.items()}


def decide_at_random(cs_df: DataFrame,
                     frac: float | int,
                     yes_node: CallerNode,
                     no_node: CallerNode
                     ) -> list[tuple[np.ndarray[tuple[int], np.dtype[bool]], CallerNode]]:
    """
    Split the dataframe of containers based on a binomial distrubution (with probability
    equal to frac) into two groups, assigned to two nodes.

    Parameters
    ----------
    cs_df : DataFrame
        The table whose rows correspond to the virtual container attributes
    frac : float (or int >= 1)
        The proportion (or total number) of containers directed to the yes_node
    yes_node : CallerNode
        The node where the selected containers are directed to.
    no_node : CalleryNode
        The node where the rest of the containers are directed to.

    Returns
    -------
    output : List[Tuple[np.ndarray[np.bool_], CallerNode]]
        The two mutually complementary boolean arrays of the containers which are being directed
        (those with True values) to the associated Nodes (either yes_node or no_node).

    """
    if  1 <= frac <= cs_df.shape[0]:
        frac /= cs_df.shape[0]
    elif (frac > cs_df.shape[0]) or (not 0 <= frac < 1):
        raise ValueError(f"Invalid value {frac}, frac should range 0-1 or 1-{cs_df.shape[0]}")

    rng = np.random.Generator(np.random.PCG64())
    destination_yes = rng.binomial(1, frac, cs_df.shape[0]) == 1
    destination_no = ~destination_yes
    return [(destination_yes, yes_node), (destination_no, no_node)]


def decide_on_attr(cs_df: DataFrame,
                   attr_name: str,
                   vals_to_nodes: dict[bool | str, CallerNode],
                   default_node: CallerNode | None=None
                   ) -> list[tuple[Series, CallerNode]]:
    """
    Based on a value (or several values) of a container attribute, each container is
    directed to the designated Node.

    Parameters
    ----------
    cs_df : DataFrame
        The table whose rows correspond to the virtual container attributes
    attr_name : str
        Container attribute, e.g. 'CALSourceCountry' or 'SCHS', 'TypeOfGoods', 'RuralDest'
    vals_to_nodes : Dict
        E.g. {"normal": DecisionNode_1, "exception": DecisionNode_2, "none": AnotherNode}
    default_node : CallerNode
        Optional, where the containers that whose attribute value doesn't match any of the keys
        from the vals_to_node key are going to be directed to.

    Returns
    -------
    output : List[Tuple[Series, CallerNode]]
       The mutually complementary boolean Series of the containers which are being directed
       (those with True values) to the associated Nodes (either yes_node or no_node).

    """
    tupvals_to_nodes: dict[tuple[str | bool], CallerNode] = keys_to_tuples(vals_to_nodes)
    masks_df = pd.DataFrame({str(_k): (cs_df[attr_name].isin(_k)) for _k in tupvals_to_nodes})
    masks_df['default'] = ~masks_df.any(axis=1).fillna(True)
    output = [(masks_df[str(_k)], val) for _k, val in tupvals_to_nodes.items()]

    if default_node:
        output.append((masks_df['default'], default_node))
    return output


def attr_decision_node(node_name: str,
                       attr_name: str,
                       vals_to_nodes: dict[str | bool, CallerNode],
                       default_node: CallerNode | None=None) -> DecisionNode:
    """
    A utilty function to set up a new DecisionNode (so as to avoid doing it directly)
    which divides containers based on the values of an attribute (e.g. 'RuralDest').

    Parameters
    ----------
    node_name : str
        The formal name assigned to the new DecisionNode.
    attr_name : str
        Container attribute, e.g. 'CALSourceCountry' or 'SCHS', 'TypeOfGoods', 'RuralDest'
    vals_to_nodes : Dict
        E.g. {"normal": DecisionNode_1, "exception": DecisionNode_2, "none": AnotherNode}
    default_node : PolicyNode
        Optional, where the containers that whose attribute value doesn't match any of the keys
        from the vals_to_node key are going to be directed to.

    Returns
    -------
    ... : DecisionNode
        The newly created DecisisonNode with required properties.

    """

    decide = partial(decide_on_attr, attr_name=attr_name,
                     vals_to_nodes=vals_to_nodes, default_node=default_node)

    return DecisionNode(name=node_name, decision_func=decide)


def random_decision_node(node_name: str, frac: float, yes_node: CallerNode, no_node: CallerNode
                         ) -> DecisionNode:
    """
    A utilty function to set up a new DecisionNode (so as to avoid doing it directly)
    which divides containers based on the fraction/volume.

    Parameters
    ----------
    node_name : str
        The formal name assigned to the newly created Node.
    frac : float (or int >= 1)
        The proportion (or total number) of containers directed to the yes_node
    yes_node : PolicyNode
        The node where the selected containers are directed to.
    no_node : PolicyyNode
        The node where the rest of the containers are directed to.

    Returns
    -------
    ... : DecisionNode
        The newly created DecisisonNode with required properties.

    """

    decide = partial(decide_at_random, frac=frac, yes_node=yes_node, no_node=no_node)
    return DecisionNode(name=node_name, decision_func=decide)


class How(Enum):
    """
    Simply to avoid the too general str type when there are only two
    acceptable words, 'all' or 'any'
    """
    ANY = "any"
    ALL = "all"


def multiple_attrs_decision_node(node_name: str,
                                 tests: dict[str, bool | str],
                                 yes_node: CallerNode,
                                 no_node: CallerNode,
                                 how: How=How.ALL) -> DecisionNode:
    """
    A utilty function to set up a new DecisionNode (so as to avoid doing it directly)
    which divides containers based on the values of an attribute (e.g. 'RuralDest').

    Parameters
    ----------
    node_name : str
        The formal name assigned to the newly created Node.
    tests : Dict[str, Union[str, bool]]
        {attribute: value, ...}. An example: {'RuralDest': True, 'TypeOfGoods': 'normal'}
    yes_node : PolicyyNode
        The node where the containers that satisfy the tests are directed to.
    no_node : PolicyyNode
        The node where the containers that do not satisfy the tests are directed to.
    how : str, optional
        Flag that indicates how the separate sets are applied. The default is 'all'.

    Returns
    -------
    ... : DecisionNode
        The newly created DecisisonNode with required properties.

    """

    decide = partial(apply_tests, tests=tests, yes_node=yes_node, no_node=no_node, how=how.value)
    return DecisionNode(name=node_name, decision_func=decide)


def ccv_decision_node(node_name: str,
                      yes_node: CallerNode,
                      no_node: CallerNode,
                      prior_n: int=10000,
                      prior_y: int=7,
                      t_risk: float=1.0,
                      t_change_level: int=95) -> DecisionNodeNumSplit:
    """
    A simple utility to evaluate the number of containers appropriate for CCV inspections
    and create the required DecisionNode which will direct a required volume to CCV.

    Parameters
    ----------
    node_name : str
        The formal name assigned to the newly created Node.
    yes_node : CallerNode
        The Node where the selected containers are directed to.
    no_node : CallerNode
        The Node where the rest of containers are directed to.
    prior_n : int
        The prior sample size as the CCV parameter. The default is 10000.
    prior_y : int
        The number of detected non-compliances. The default is 7.
    t_risk : float
        The acceptable threshold risk. The default is 1.
    t_change_level : int, optional
        The value of T1_level_percent from the CCV table, corresponding to T_change,
        i.e., the quantile of the rate of non-compliance distribution. The default is 95%.

    Returns
    -------
    ... : DecisionNodeNumSplit
        A new DecisisonNode which directs the required number of container to CCV inspectionsA

    """
    num_for_yes = get_num_ccv_samples(prior_n, prior_y, t_risk, t_change_level)
    return DecisionNodeNumSplit(name=node_name, number=num_for_yes,
                                yes_node=yes_node, no_node=no_node)


def by_number_decision_node(node_name: str, con_y: int, yes_node: CallerNode, no_node: CallerNode
                         ) -> DecisionNodeNumSplit:
    """
    A simple alternative to the random_decision_node(), which divides the containers by a number,
    rather than a fraction, when the policy dictates so.

    Parameters
    ----------
    node_name : str
        The formal name assigned to the newly created Node.
    con_y : int
        How many containers to the yes_node.
    yes_node : PolicyNode
        The node where the selected containers are directed to.
    no_node : PolicyyNode
        The node where the rest of the containers are directed to.

    Returns
    -------
    ... : DecisionNodeNumSplit
        A new DecisisonNode which directs the required number of container to the yes_node

    """
    return DecisionNodeNumSplit(name=node_name, number=con_y, yes_node=yes_node, no_node=no_node)


def basic_inspect_node(name: str, mon_frac: float, sensitivity: float, child: CallerNode | None=None
                      ) -> PathwayNode:
    """
    A simple utility to create a PathwayNode with a Basic sampling, rather than doing it directly.

    Parameters
    ----------
    name: str
        The formal name assigned to the newly created Node.
    mon_frac : float
        The monitoring fraction (i.e., the container proportion required to inspect)
    sensitivity : float
        The presumed probability of detecting contamination, when present
    child : PathwayNode, optional
        Where the the containers are directed to after the sampling is completed and mask updated.
        The default is None.

    Returns
    -------
    ... : PathwayNode
        The Node that implements the required sampling protocol

    """
    sampler = BasicSampler(sensitivity=sensitivity, monit_frac=mon_frac)
    if not child:
        new_node = PathwayNode(name=name, sampler=sampler)
    else:
        new_node = PathwayNode(name=name, sampler=sampler, child_node=child)
    new_node.name = f'{name} sampled {int(mon_frac * 100)}%'
    return new_node


def basic_treatment(name: str, effectiveness: float, child_node: PathwayNode) -> TreatmentNode:
    """
    A simple utility to create a TreatmentNode, rather than doing it directly.

    Parameters
    ----------
    name : str
        The formal name assigned to the newly created Node.
    effectiveness: float
        The presumed probability that the contamination will be removed by the treatment.
    child_node: PathwayNode
        Where the the containers are directed to after the sampling is completed and mask updated.

    Returns
    -------
    ... : TreatmentNode
        The Node for cleaning of contaminated containers (if detected at a PathwayNode)

    """
    return TreatmentNode(name=name, effectiveness=effectiveness, child_node=child_node)


def release(name: str) -> PathwayNode:
    """
    A utility to create a 'RelaseNode', i.e. a PathwatNode which doesn't sample/inspect anything.

    Parameters
    ----------
    name: str
        The formal name assigned to the newly created Node.

    Returns
    -------
    PathwayNode
        The non-sampling PathwayNode (i.e., it let's everything through).

    """
    return basic_inspect_node(name, 0, 0)


def ccv_node(name: str,
             sensitivity: float,
             prior_n: int=10_000,
             prior_y: int=7,
             t_risk: float=1.0,
             t_change_level: int=95,
             child: CallerNode | None=None) -> PathwayNode:
    """
    A utility function to generate a PathwayNode which implements the CCV sampling regime.
    Unused for external inspections (originally suggested for goods inspections only).

    Parameters
    ----------
    name : str
        The formal name assigned to the newly created Node.
    sensitivity : float
        The presumed probability of detecting contamination, when present
    prior_n : int
        The prior sample size as the CCV parameter. The default is 10000.
    prior_y : int
        The number of detected non-compliances. The default is 7.
    t_risk : float
        The acceptable threshold risk. The default is 1.
    t_change_level : int, optional
        The value of T1_level_percent from the CCV table, corresponding to T_change,
        i.e., the quantile of the rate of non-compliance distribution. The default is 95%.
    child : PathwayNode, optional
        Where the containers would be directed afterwards. The default is None.

    Returns
    -------
    ... : PathwayNode
        A Node with the CCV sampling protocol.

    """
    sampler = RandomSampler(sensitivity=sensitivity,
            prior_n=prior_n,
            prior_y=prior_y,
            t_risk=t_risk,
            t_change_level=t_change_level)
    if not child:
        return PathwayNode(name=name, sampler=sampler)
    return PathwayNode(name=name, sampler=sampler, child_node=child)


def cbis_node(name: str,
              sensitivity: float ,
              clearance_num: int,
              monit_frac: float,
              tight_cens_num: int=4,
              csp3: bool=True,
              child_node: CallerNode | None=None) -> PathwayNode:
    """
    A utility function to generate a PathwayNode which implements a compliance based
    intervention scheme (CBIS) with a continuous sampling protocol (CSP 1 or 3) for
    medium high risk CAL containers.

    Parameters
    ----------
    name : str
        The formal name assigned to it.
    sensitivity : float
        The presumed probability of detecting contamination, when present
    clearance_num : int
        The number of passed ('clean') inspections required to move from the strict
        census mode to a more relaxed monitoring mode (for an importer or a loading port)
    monit_frac : float
        Monitoring fraction (the inspected proportion) for a port/importer in the monitoring mode.
    tight_cens_num : int, optional
        In CSP3 only. The required number of inspections after the 1st failed case. The default is 4
    csp3 : bool, optional
        Whether CSP3 or CSP1 regime is applied. The default is True.
    child_node : CallerNode, optional
        Where the containers are directed to afterwards. The default is None.

    Returns
    -------
    ... : PathwayNode
        A Node which implements the CSP (1 or 3) sampling protocol for CAL containers.

    """
    sampler: CSP1Sampler | CSP3Sampler
    if csp3:
        sampler = CSP3Sampler(sensitivity=sensitivity,
                clearance_num=clearance_num,
                monit_frac=monit_frac,
                tight_cens_num=tight_cens_num)
    else:
        sampler = CSP1Sampler(sensitivity=sensitivity,
                clearance_num=clearance_num,
                monit_frac=monit_frac)
    return PathwayNode(name=name, sampler=sampler, child_node=child_node)


def cbis_node_mixed(name: str,
                    sensitivities: dict[str, tuple[float, ...]],
                    clearance_num: int,
                    monit_frac: float,
                    tight_cens_num: int=4,
                    csp3: bool=True,
                    child_node: CallerNode | None=None) -> PathwayNode:
    """
    A utility function to generate a PathwayNode which implements a compliance based
    intervention scheme (CBIS) with a continuous sampling protocol (CSP 1 or 3) where the
    inspections can be carried out either by the DAFF or approved arangement (AA) provider.

    Parameters
    ----------
    name : str
        The formal name assigned to it.
    sensitivities : Dict[str, Tuple[float, float]]
        The name of each participant (e.g., DAFF, AA) with their respective (prop, sensitivity) pair
    clearance_num : int
        The number of passed ('clean') inspections required to move from the strict
        census mode to a more relaxed monitoring mode (for an importer or a loading port)
    monit_frac : float
        Monitoring fraction (the inspected proportion) for a port/importer in the monitoring mode.
    tight_cens_num : int, optional
        In CSP3 only. The required number of inspections after the 1st failed case. The default is 4
    csp3 : bool, optional
        Whether CSP3 or CSP1 regime is applied. The default is True.
    child_node : CallerNode, optional
        Where the containers are directed to afterwards. The default is None.

    Returns
    -------
    ... : PathwayNode
        A Node which implements the CSP (1 or 3) sampling protocol by DAFF/AA for CAL containers.

    """
    # effective sensitivity as a weighted (by respective proportions) average of the sensitivities:
    sensi = sum(s * p for s, p in sensitivities.values())
    sampler: CSP1Sampler | CSP3Sampler
    if csp3:
        sampler = CSP3Sampler(sensitivity=sensi,
                clearance_num=clearance_num,
                monit_frac=monit_frac,
                tight_cens_num=tight_cens_num)
    else:
        sampler = CSP1Sampler(sensitivity=sensi, clearance_num=clearance_num, monit_frac=monit_frac)

    return PathwayNode(name=name, sampler=sampler, child_node=child_node,
                       mixed_sensitivities=sensitivities)
