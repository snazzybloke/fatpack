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
Figure output

Various different plot/figure outputs for summary figures (over many simulations)

Running: python -m plot_multisim_data -y a_yaml_input_file

"""

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from matplotlib import rc
from flatpack.src.config import Config
from flatpack.plot import plot_data

# rc('text', usetex=True)
# rc('font', **{'family': 'sans-serif'})
rc('text', usetex=False)
plt.switch_backend('agg')


def plot_mean_median_contamination_breakdown(mean_or_median: str,
                                             folder: Path,
                                             policy_name: str,
                                             save_folder: Path) -> None:
    """
    Function description

    Parameters
    ----------
    mean_or_median : str
        DESCRIPTION.
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    data_file = folder / f"summary_{policy_name}_statistics.csv"
    pd_obj = pd.read_csv(data_file)

    pd_obj.set_index('quantity', inplace=True)
    mean_or_median_dict = pd_obj.to_dict()[mean_or_median]

    # total_contamination = mean_or_median_dict['total contamination']
    total_contamination_found = mean_or_median_dict['contamination found']

    leakage = mean_or_median_dict['contamination leakage']

    treated =mean_or_median_dict['contamination treated']

    if treated > 0:
        pie = [total_contamination_found, leakage, treated]
        pie_labels = ["Found", "Undetected", "Treated"]
        pie_colours = ["#084081","#4495ab","#6cad7e"]
    else:
        pie  = [total_contamination_found, leakage]
        pie_labels = ["Found", "Undetected",]
        pie_colours = ["#084081","#4495ab"]

    def helper(pct: float, allvals: List[int]) -> str:
        """
        helper function description

        Parameters
        ----------
        pct : float
            DESCRIPTION.
        allvals : List[int]
            DESCRIPTION.

        Returns
        -------
        str
            DESCRIPTION.

        """
        absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
        if "goods" not in policy_name:
            return f"{pct:.1f}%\n({absolute:d} containers)"
        return f"{pct:.1f}%\n({absolute:d} consignments)"

    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    _, _, autotexts = ax0.pie(pie,
                             labels=pie_labels,
                             autopct=lambda pct: helper(pct, pie),
                             colors=pie_colours,shadow=False,
                             startangle=90,
                             wedgeprops = {"edgecolor" : "w", "linewidth": 2, "antialiased": True},
                             textprops={"fontsize": 14})

    for autotext in autotexts:
        autotext.set_color("white")

    # ax.xaxis.set_label_position('top')
    # ax.set_xlabel(f'{mean_or_median.capitalize()} contaminated containers')
    fig.suptitle(f'{mean_or_median.capitalize()} contaminated containers')
    if "goods" in policy_name:
        ax0.set_xlabel(f'{mean_or_median.capitalize()} contaminated lines (consignments)')

    if "treatment" in policy_name.lower():
        pass
    else:
        ax0.set_ylim(top=1)

    save_file = save_folder / f"summary_{policy_name}"
    plt.savefig(f"{save_file}_contamination_breakdown.png", bbox_inches="tight", transparent=True)
    plt.close()


def get_triplet(folder: Path, policy_name: str, median: Optional[bool]=False
                ) -> Tuple[Dict, List, List]:
    """
    Evaluate new_dict, list_of_inspections, and list_of_contamination_found
    which are needed for bar plots. Works for both mean and median.

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    median : Optional[bool], optional
        DESCRIPTION. The default is False.

    Returns
    -------
    Tuple[Dict, List, List]
        DESCRIPTION.

    """

    data_file = folder / f"nodes_{policy_name}_statistics.csv"
    pd_obj = pd.read_csv(data_file)
    dict_obj = pd_obj.to_dict(orient="records")

    list_of_inspections = []
    inspections_std = []
    list_of_contamination_found = []
    contamination_std = []
    new_dict = {}
    for row in dict_obj:
        if "release" in row['node'].lower():
            continue # skip this
        if "treatment" in row['node'].lower() and "treatment" not in policy_name.lower():
            continue # also skip this

        if row['node'] not in new_dict:
            new_dict[row['node']] = {}

        if row['quantity'] == "contamination found" :
            inspected_contam = row['mean'] if not median else row['median']
            list_of_contamination_found.append(inspected_contam)
            contamination_std.append(row['standard_deviation'])
            new_dict[row['node']]['contamination found'] = inspected_contam
            new_dict[row['node']]['contamination found std'] = row['standard_deviation']
            if median:
                new_dict[row['node']]['contamination found quantile 2.5%'] = row['quantile_2.5%']
                new_dict[row['node']]['contamination found quantile 97.5%'] = row['quantile_97.5%']
        elif row['quantity'] == "inspections":
            list_of_inspections.append(row['mean'])
            inspections_std.append(row['standard_deviation'])
            new_dict[row['node']]['inspections'] = row['mean']
            new_dict[row['node']]['inspections std'] = row['standard_deviation']
            if median:
                new_dict[row['node']]['inspections quantile 2.5%'] = row['quantile_2.5%']
                new_dict[row['node']]['inspections quantile 97.5%'] = row['quantile_97.5%']

    return new_dict, list_of_inspections, list_of_contamination_found


def bar_plotter(nrows: int,
               new_dict: Dict,
               policy_name: str,
               list_of_inspections: List[float],
               list_of_contamination_found: List[float],
               save_folder: Path,
               median: Optional[bool]=False) -> None:
    """
    Produces the summary (mean or median) bar plot for 1D (no axis break) and 2D (with axis break).

    Parameters
    ----------
    nrows : int
        DESCRIPTION.
    new_dict : Dict
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    list_of_inspections : List[float]
        DESCRIPTION.
    list_of_contamination_found : List[float]
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.
    median : Optional[str], optional
        Summary with(out) median to plot. The default is None.

    Returns
    -------
    None
        DESCRIPTION.

    """
    fig, axs = plt.subplots(1, nrows, figsize=(10, 5), sharey=True)
    # make a 1D array for a single row plot (no axis break):
    if nrows == 1:
        axs = np.array([axs])
    elif nrows == 2:
        lo_lim, hi_lim = plot_data.find_gap(
            list_of_inspections + list_of_contamination_found)
        logger.info(f"Low and high lim: {lo_lim}, {hi_lim}")
    else:
        logger.error(f"\nThe nrows value {nrows} is invalid, only 1 or 2 is acceptable")
    y_pos = []
    node_names = []
    _y = 0
    for key, row in new_dict.items():
        node_name = key.replace("_", " ")
        node_name = node_name.replace("%", r"\%")
        if "release" in node_name.lower():
            continue # skip this
        if "treatment" in node_name.lower() and "treatment" not in policy_name.lower():
            continue # also skip this
        node_names.append(node_name)

        inspections = row['inspections']
        if np.isnan(inspections):
            inspections = 0

        inspected_contam = row['contamination found']
        # inspected_no_contam = inspections - inspected_contam

        list_of_inspections.append(inspections)
        list_of_contamination_found.append(inspected_contam)
        if median:
            e_bars = [[inspected_contam - row["contamination found quantile 2.5%"]],
                      [row["contamination found quantile 97.5%"] - inspected_contam]]
            f_bars = [[inspections - row["inspections quantile 2.5%"]] ,
                      [row["inspections quantile 97.5%"] - inspections]]
            d_row = row["inspections quantile 97.5%"]
            tmp = [row["inspections quantile 2.5%"], row["inspections quantile 97.5%"]]
            b_txt = f"{str(round(inspections, 1))} ({round(tmp[0], 1)}, {round(tmp[1], 1)})"
        else:
            e_bars = row["contamination found std"]
            f_bars = row['inspections std']
            d_row = inspections + row['inspections std']
            b_txt = f"{inspections:.2f}±{row['inspections std']:.2f}"

        for _i in range(axs.shape[0]):
            axs[_i].barh(_y,
                    inspected_contam,
                    xerr=e_bars,
                    left=0,
                    color="#084081",
                    capsize=5,
                    zorder=10)
            axs[_i].barh(_y,
                    inspections,
                    xerr=f_bars,
                    left=0,
                    color="#90b7e4",
                    capsize=5,
                    zorder=0)

        if (nrows == 1) or inspections <= lo_lim:
            _ix = 0
        else:
            _ix = 1
        axs[_ix].text(d_row + 60,
                _y,
                b_txt,
                color="black",
                va="center")

        y_pos.append(_y)
        _y -= 1

    axs[0].set_yticks(y_pos)
    axs[0].set_yticklabels(node_names)
    fig.suptitle("Total inspections")

    if nrows > 1:
        # only for 2D (with axis break)
        axs[0].set_xlim(0, lo_lim + 200)
        if max(list_of_inspections) > 10_000:
            axs[1].set_xlim(left=hi_lim - 1_000)
        else:
            axs[1].set_xlim(left=hi_lim - 200)

    extra = "bar"
    _ix = axs.shape[0] - 1
    axs[_ix].legend(["Contamination detected", "Total inspections"],
                  bbox_to_anchor=(1.04, 0),
                  loc="lower left",
                  borderaxespad=0,
                  frameon=False,
                  handlelength=0.7)
    if nrows == 2:
        axs[1].legend(["Contamination detected", "Total inspections"],
                  loc="upper center",
                  bbox_to_anchor=(0.5, -0.05),
                  borderaxespad=0,
                  frameon=False,
                  handlelength=0.7)
        extra = "_with_axis_break"
    if median:
        extra += "_median_version"

    for _i in range(axs.shape[0]):
        axs[_i].xaxis.grid()
        axs[_i].set_axisbelow(True)

    save_file = save_folder / policy_name
    plt.savefig(
        f"{save_file}_summary_{extra}.png", bbox_inches="tight", transparent=True)
    plt.close()


def plot_summary_bar(folder: Path, policy_name: str, save_folder: Path) -> None:
    """
    No gap, single panel plot

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    new_dict, list_of_inspections, list_of_contamination_found = get_triplet(folder, policy_name)
    bar_plotter(1,
               new_dict,
               policy_name,
               list_of_inspections,
               list_of_contamination_found,
               save_folder)


def plot_summary_bar_with_break(folder: Path, policy_name: str, save_folder: Path) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    new_dict, list_of_inspections, list_of_contamination_found = get_triplet(folder, policy_name)
    bar_plotter(2,
               new_dict,
               policy_name,
               list_of_inspections,
               list_of_contamination_found,
               save_folder)


def plot_summary_bar_median_version(folder: Path, policy_name: str, save_folder: Path) -> None:
    """
    No gap, single panel plot

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    new_dict, list_of_inspections, list_of_contamination_found = get_triplet(folder,
                                                                             policy_name,
                                                                             median=True)
    bar_plotter(1,
                new_dict,
                policy_name,
                list_of_inspections,
                list_of_contamination_found,
                save_folder,
                median=True)


def plot_summary_bar_with_break_median_version(folder: Path, policy_name: str, save_folder: Path
                                               ) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    new_dict, list_of_inspections, list_of_contamination_found = get_triplet(folder,
                                                                             policy_name,
                                                                             median=True)
    bar_plotter(2,
                new_dict,
                policy_name,
                list_of_inspections,
                list_of_contamination_found,
                save_folder,
                median=True)


def output_truncated_table(folder: Path, policy_name: str, save_folder: Path) -> DataFrame:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    policy_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    DataFrame
        DESCRIPTION.

    """

    data_file = folder / f"summary_{policy_name}_statistics.csv"
    pd_obj = pd.read_csv(data_file,
                         usecols=['quantity', 'median', 'quantile_2.5%', 'quantile_97.5%'])

    rows_to_keep = ["total inspections", "contamination found", "contamination leakage"]
    if "treatment" in policy_name.lower():
        rows_to_keep.extend(["total treatments", "contamination treated"])

    pd_obj = pd_obj.query('quantity in @rows_to_keep')
    kwargs = {'95% quantiles': [f"({round(a, 1)}, {round(b, 1)})"
                                for a, b in zip(pd_obj['quantile_2.5%'], pd_obj['quantile_97.5%'])]
                                }
    pd_obj = pd_obj.assign(**kwargs)
    pd_obj = pd_obj[['quantity', 'median', '95% quantiles']]
    pd_obj.to_latex(save_folder / f"summary_{policy_name}_truncated_statistics.tex", index=False)
    pd_obj[policy_name] = [f"{a} {b}" for a, b in zip(pd_obj["median"], pd_obj["95% quantiles"])]
    pd_obj.rename(columns={'quantity': 'policy'}, inplace=True)
    pd_obj = pd_obj.reset_index(drop=True)
    pd_obj = pd_obj[['policy', policy_name]]
    pd_obj = pd_obj.set_index('policy')
    pd_obj = pd_obj.T
    pd_obj.to_latex(save_folder / f"summary_{policy_name}_truncated_transposed.tex", index=True)
    return pd_obj


def plot_individual_sims(root_list_of_policies: List[str],
                         original_container_folder: Path,
                         original_container_file: str,
                         workdir: Path,
                         savedir: Path,
                         sims_to_plot: List[int]) -> None:
    """
    Function description

    Parameters
    ----------
    root_list_of_policies : List[str]
        DESCRIPTION.
    original_container_folder : Path
        DESCRIPTION.
    original_container_file : str
        DESCRIPTION.
    workdir : Path
        DESCRIPTION.
    savedir : Path
        DESCRIPTION.
    sims_to_plot : List[int]
        DESCRIPTION.

    Returns
    -------
    None

    """
    # first, outputing the individual things:
    for policy_name in root_list_of_policies:
        folder = workdir / policy_name
        save_folder = savedir / policy_name
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=False)

        for _i in sims_to_plot:
            root_file_name = f"{policy_name}_{_i}.csv"
            input_file_name = f"nodes_{root_file_name}"
            logger.info(f"\nSimulation {_i} input file: {input_file_name}")
            plot_data.plot_contamination_breakdown_single_scenario(
                folder,
                root_file_name,
                save_folder,
                original_container_folder,
                original_container_file)
            plot_data.print_report_card(folder,
                                        root_file_name,
                                        save_folder,
                                        original_container_folder,
                                        original_container_file)
            # plot_data.plot_single_scenario_bar_with_break(folder, ...)
            plot_data.plot_single_scenario_bar(folder,
                                                input_file_name,
                                                save_folder)


def agg_aa_daff_call_inspections(policy_name: str,
                                 num_sims: int,
                                 pd_obj_truncd_transpd: DataFrame,
                                 save_folder: Path) -> None:
    """
    Originally called 'aggregated_table_with_AA_DAFF_CAL_inspections'

    Parameters
    ----------
    policy_name : str
        DESCRIPTION.
    num_sims : int
        DESCRIPTION.
    pd_obj_truncated_transposed : pd.DataFrame
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None

    """
    if "external_future" not in policy_name:
        raise ValueError("should not be run with non external future ")

    daff_names = ["1a. (SCHS) CSP-3 inspection - DAFF",
                  "2. CAL-high DAFF inspection",
                  "3. CAL-medium DAFF inspection"]
    aa_names = ["1b. (SCHS) CSP-3 inspection - AA", "4. CAL-medium AA inspection"]
    # directorty to read results for this policy:
    folder = Path.cwd().parent / "results" / "multi" / policy_name

    daff_inspections = []
    aa_inspections = []

    for i in range(num_sims):
        node_filename = folder / f"nodes_{policy_name}_{i}.csv"
        node_file = pd.read_csv(node_filename)

        node_df = node_file[["node","inspections"]]
        node_dict = node_df.to_dict('records')
        for row in node_dict:
            if row['node'] in daff_names:
                daff_inspections.append(row['inspections'])
            elif row['node'] in aa_names:
                aa_inspections.append(row['inspections'])
            else:
                pass

    daff_median = round(np.median(daff_inspections), 1)
    daff_lower_quantile = round(np.quantile(daff_inspections, 0.025), 1)
    daff_upper_quantile = round(np.quantile(daff_inspections, 0.975), 1)
    pd_obj_truncd_transpd['CAL DAFF inspections'] = [
        f"{daff_median} ({daff_lower_quantile}, {daff_upper_quantile})"]

    aa_median = round(np.median(aa_inspections), 1)
    aa_lower_quantile = round(np.quantile(aa_inspections, 0.025), 1)
    aa_upper_quantile = round(np.quantile(aa_inspections, 0.975), 1)
    pd_obj_truncd_transpd['CAL AA inspections'] = [
        f"{aa_median} ({aa_lower_quantile}, {aa_upper_quantile})"]

    pd_obj_truncd_transpd.to_latex(
        save_folder / f"summary_{policy_name}_truncated_T_with_CAL_inspections.tex", index=True)

    pd_untranspd = pd_obj_truncd_transpd.T
    pd_untranspd.to_latex(save_folder / f"summary_{policy_name}_truncated_with_CAL_inspections.tex",
                             index=True)


def plot_aggregated_results(root_list_of_policies: List[str],
                            workdir: Path,
                            savedir: Path,
                            num_sims: Optional[int]=100) -> None:
    """
    Function description

    Parameters
    ----------
    root_list_of_policies : List[str]
        DESCRIPTION.
    workdir : Path
        DESCRIPTION.
    savedir : Path
        DESCRIPTION.
    num_sims : Optional[int], optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None

    """
    # first, outputing the individual things:
    for policy_name in root_list_of_policies:
        # root_file_name = f"{policy_name}.csv"
        # input_file_name = f"nodes_{policy_name}"

        logger.info(f"\nPolicy: {policy_name}")
        folder = workdir / policy_name
        save_folder = savedir / policy_name
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=False)

        mean_or_median = "mean"
        plot_mean_median_contamination_breakdown(mean_or_median, folder, policy_name, save_folder)

        # plot_summary_bar_with_break(folder, policy_name,save_folder)
        plot_summary_bar(folder, policy_name, save_folder)

        plot_summary_bar_median_version(folder, policy_name, save_folder)

        pd_obj_truncd_transpd = output_truncated_table(folder, policy_name, save_folder)

        if "external_future" in policy_name:
            agg_aa_daff_call_inspections(policy_name, num_sims, pd_obj_truncd_transpd, save_folder)


def main(conf: Config) -> None:
    """
    The program driver (when called from command line)

    Parameters
    ----------
    conf : Config
        DESCRIPTION.

    Returns
    -------
    None

    """
    root_list_of_policies = [
        # current external state
        conf.old_wdir_multi.name,
        # future external state
        conf.new_wdir_multi.name,
        f"{conf.new_wdir_multi.name}_schs",
        ]
    # a list of individual siumlations to plot:
    sims_to_plot = [0]
    # first, outputing the individual things:
    original_container_folder = conf.vcon_dir # needed if treatment occurs (on the external pathway)
    original_container_file = conf.vcon_file.name
    plot_individual_sims(root_list_of_policies,
                         original_container_folder,
                         original_container_file,
                         conf.old_wdir_multi.parent,
                         conf.old_vdir_multi.parent,
                         sims_to_plot)

    # second, outputing some summary plots:
    plot_aggregated_results(root_list_of_policies,
                            conf.old_wdir_multi.parent,
                            conf.old_vdir_multi.parent,
                            num_sims=conf.sim_total)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    main(Config(ARG["yaml"]))
