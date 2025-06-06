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
Executes the simulations via the inspection_pathways module for a set number (e.g. 100)
of times. While the input virtual container file is common to all the simulations,
the outputs from their inherently random nature varies. Hence the outputs yield
sufficient variety from which descriptive statistics and uncertaintes (means, medians,
standard deviations) can be evaluated.

Running: python inspection_pathways_multi.py -y a_input_yaml_file
or:      python inspection_pathways_multi.py --yaml a_input_yaml_file
"""

import argparse
from pathlib import Path
import sys
from typing import Callable
import time
import pandas as pd
from tqdm import tqdm
from loguru import logger

from flatpack.src.policy import simulate_policy
from flatpack.src.simulate_pathways import (external_risks_policy_current,
                                            external_risks_policy_new)
from flatpack.src.config import Config

logger.remove()

def printtime(start: float, label: str) -> None:
    """
    Measueres the execution time, beginning from 'start' (seconds as a float)

    Parameters
    ----------
    start : float
        The time (in seconds) at the beginning of run_pathway_multi() execution
    label : str
        Some wording to describe what is being measeured

    Returns
    -------
    None

    """
    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"\nTime to {label}: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")


def aggregate_and_summarise_results(folder_path: str | Path,
                                    total_num_sims: int,
                                    policy_name: str) -> None:
    """
    Aggregates all the individual simulations into a big file, for both nodes and summary files
    Plus summarises the results (means, medians, standard deviation)

    Parameters
    ----------
    folder_path : Path | str
        The output folder name (the base from the YAML config extrapolated with a value, e.g. _schs)
    total_num_sims : int
        The number of siumulations to run, defined in YAML input as 'sim_total'
    policy_name : strA
        The formal name assigned to the policy, e.g. 'External Risks Current'

    Returns
    -------
    None

    """
    # aggregate the summary files
    sims_list = []
    for _i in range(total_num_sims):
        csv_file = Path.cwd() / folder_path / f"summary_{policy_name}_{_i}.csv"
        _df = pd.read_csv(csv_file)
        _df.insert(0, 'simulation number', _i )
        sims_list.append(_df)

    summary_df = pd.concat(sims_list, ignore_index=True)
    summary_df.to_csv(
        Path.cwd() / folder_path / f"summary_{policy_name}_total.csv", index=False)


    # summarise (summary statistics) the summary files
    summary_stats = summary_df.drop(columns='simulation number').\
        describe(include="number", percentiles=[0.025, 0.5, 0.975]).\
        drop(['count', 'min', 'max'], axis=0).\
        rename(index={'std': 'standard_deviation',
                        '50%': 'median',
                        '2.5%': 'quantile_2.5%',
                        '97.5%': 'quantile_97.5%'}).T.\
        rename_axis(index="quantity")
    summary_stats.to_csv(
        Path.cwd() / folder_path / f"summary_{policy_name}_statistics.csv", index=True)

    # aggregate the node files
    sims_list = []
    for _i in range(total_num_sims):
        csv_file = Path.cwd() / folder_path / f"nodes_{policy_name}_{_i}.csv"
        try:
            _df = pd.read_csv(csv_file)
        except (OSError, FileNotFoundError) as _e:
            logger.error(f"\nFile {csv_file} missing? {_e}")
        else:
            _df.insert(0, 'simulation number', _i)
            sims_list.append(_df)

    nodes_df = pd.concat(sims_list, ignore_index=True)
    nodes_df.to_csv(Path.cwd() / folder_path / f"nodes_{policy_name}_total.csv", index=False)

    # summarise the node files
    summary_statistics = []
    list_of_unique_node_names = nodes_df['node'].unique()
    list_of_quantities = nodes_df.columns.to_list()
    list_of_quantities.remove('node')
    list_of_quantities.remove('simulation number')

    for node in list_of_unique_node_names:
        node_df = nodes_df[nodes_df['node'] == node]
        summary_stats = node_df.drop(columns='simulation number').\
            describe(include="number", percentiles=[0.025, 0.5, 0.975]).\
            drop(['count', 'min', 'max'], axis=0).\
            rename(index={'std': 'standard_deviation',
                            '50%': 'median',
                            '2.5%': 'quantile_2.5%',
                            '97.5%': 'quantile_97.5%'}).T.\
            rename_axis(index="quantity").\
            reset_index()
        summary_stats.insert(0, "node", node)
        summary_statistics.append(summary_stats)

    summary_statistics_df = pd.concat(summary_statistics, ignore_index=True)
    summary_statistics_df.to_csv(Path.cwd() / folder_path / f"nodes_{policy_name}_statistics.csv",
                                 index=False)


def run_pathway_multi(pathway_function: Callable,
                      cs_fname: Path,
                      folder_name: Path,
                      policy_name: str,
                      total_num_sims: int,
                      **kwargs) -> None:
    """
    Simulates the container pathway (current or new) for the required number of times.

    Parameters
    ----------
    pathway_function : Callable
        The function that implements the state of pathway of interest.
        Either external_risks_policy_new() or external_risks_policy_current()
    cs_fname : Union[Path, str]
        The path to the virtual container file.
    folder_name : Path | str
        The output folder name (the base from the YAML config extrapolated with a value, e.g. _schs)
    total_num_sims : int
        The number of siumulations to run, defined in YAML input as 'sim_total'
    policy_name : str
        The formal name assigned to the policy, e.g. 'External Risks Current'
    kwargs : Optional[Dict], optional
        Keyword arguments for pathway_function().

    Returns
    -------
    None

    """

    start_time = time.time()

    # makes the dedicated output folder if folder does not yet exist, inside results/multi
    folder_name.mkdir(parents=True, exist_ok=True)

    print_output_file = folder_name / f"{policy_name}_output.txt"

    original_stderr = sys.stderr   # Save a reference to the original standard output

    for _i in tqdm(range(total_num_sims)):
        with open(print_output_file, 'a', encoding="utf-8") as fout:
            sys.stderr = fout  # Change the standard output to the file we created.
            logger.add(sys.stderr, level="INFO")
            logger.info(f"\nsimulation {_i}")
            policy_name_i = f'{policy_name}_{_i}'

            if kwargs:
                simulate_policy(*pathway_function(policy_name_i, cs_fname, **kwargs),
                                cs_fname=cs_fname,
                                save_results=kwargs.get("save_results", True),
                                save_container_results=kwargs.get("save_container_results", False),
                                save_summary=kwargs.get("save_summary", True),
                                output_dir=folder_name)
            else:
                simulate_policy(*pathway_function(policy_name_i, cs_fname),
                                cs_fname=cs_fname,
                                save_results=True,
                                save_container_results=False,
                                save_summary=True,
                                output_dir=folder_name)
        sys.stderr = original_stderr # Reset the standard output to its original value
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    aggregate_and_summarise_results(folder_name, total_num_sims, policy_name)

    printtime(start_time, "run entire thing")


def run_helper(pathway_fun: Callable,
               total_num_sims: int,
               dir_base: Path,
               cs_base: Path,
               frac_eff_risk_sensi: float | tuple[float]=None,
               **kwargs) -> None:
    """
    Replacing a dozen of nearly identical blocks with a call to helper()

    Parameters
    ----------
    pathway_fun : Callable
        The function that implements the state of pathway of interest.
        Either external_risks_policy_new() or external_risks_policy_current()
    total_num_sims : int
        The number of siumulations to run, defined in YAML input as 'sim_total'
    dir_base : Union[str, Path]
        The first part to of the folder name which is to be extrapolated with value(s)
        from frac_eff_risk_sensi
    cs_base : Union[str, Path]
        The first part to of the container file name which is to be extrapolated with value(s)
        from frac_eff_risk_sensi.
    frac_eff_risk_sensi : Union[float, Tuple[float]]
        Either a single value or a tuple of values for fraction, sensitivity, risk, effectiveness
    kwargs : Optional[Dict], optional
        Keyword arguments to pathway_fun().

    Returns
    -------
    None

    """
    fers: tuple[float, ...] = frac_eff_risk_sensi if isinstance(frac_eff_risk_sensi, tuple) \
                                                else (frac_eff_risk_sensi,)

    the_key: str | None = None
    if kwargs:
        # retain the_key which may be present in the kwargs but without a proper value:
        for _k, _v in kwargs.items():
            if (not _v) and (_k != "vcon_file"):
                the_key = _k

    dir_name: Path | str = dir_base
    if pathway_fun.__name__ == external_risks_policy_current.__name__:
        # default is SCHS for the current ("old") policy:
        dir_name = Path(f"{dir_base}_csp") if kwargs.get("csp_instead_of_schs") else dir_base
    else:
        # default is CSP3 for the new policy:
        dir_name = dir_base if kwargs.get("csp_instead_of_schs") else Path(f"{dir_base}_schs")
    pol_name: str = dir_name.name
    cs_fname: Path = Path(cs_base)
    for value in fers:
        if str(dir_base)[-1] == "_" and value:
            # if it ends with the underscore, extrapolate it with the value
            dir_name = f"{dir_base}{value}"
            pol_name = dir_name
        if str(cs_base)[-1] == "_" and value:
            cs_fname = f"{cs_base}{value}.csv"
        if the_key and value:
            # update it if the_key was initially present in kwargs (without value)
            kwargs.update({the_key: value})
        if kwargs and ("vcon_file" in kwargs):
            # update it if this key was initially present in kwargs (without value)
            kwargs.update({"vcon_file": cs_fname})
        run_pathway_multi(pathway_fun, cs_fname, Path(dir_name), pol_name, total_num_sims, **kwargs)


def run_external_pathways_current(conf: Config) -> None:
    """
    Executes the simulations for the current external pathway (optionally  wtih CSP3 protocol)

    Parameters
    ----------
    conf : Config
        The parameters from the input YAML file.

    Returns
    -------
    None

    """
    run_helper(external_risks_policy_current,
               conf.sim_total,
               conf.old_wdir_multi,
               conf.vcon_file,
               **conf.fun_params)


def run_external_pathways_new(conf: Config) -> None:
    """
    Runs the simulations for the external pathways, future state.

    Parameters
    ----------
    conf : Config
        The parameters from the input YAML file.

    Returns
    -------
    None

    """
    run_helper(external_risks_policy_new,
               conf.sim_total,
               conf.new_wdir_multi,
               conf.vcon_file,
               **conf.fun_params)

    # try the alternate SCHS policy:
    conf.fun_params["csp_instead_of_schs"] = not conf.fun_params["csp_instead_of_schs"]
    run_helper(external_risks_policy_new, conf.sim_total, conf.new_wdir_multi, conf.vcon_file,
               **conf.fun_params)
    # restore the original boolean value
    conf.fun_params["csp_instead_of_schs"] = not conf.fun_params["csp_instead_of_schs"]


def main(conf: Config) -> None:
    """
    The driver which executes the current and future state of the container pathway.

    Parameters
    ----------
    conf : Config
        The parameters from the input YAML file.

    Returns
    -------
    None

    """
    run_external_pathways_current(conf)
    run_external_pathways_new(conf)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    main(Config(ARG["yaml"]))
