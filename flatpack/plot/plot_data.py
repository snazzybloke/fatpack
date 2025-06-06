###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ante Bilic                          #
# Email: ante.bilic.mr@gmail.com                  #
# Status: N/A                                     #
###################################################

"""Figure output

Various different plot/figure outputs, for individual simulations

Running: python -m plot_data -y a_yaml_input_file

"""
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rc
from flatpack.src.config import Config
# rc('text', usetex=True)
rc("text", usetex=False)
rc("font", **{"family": "sans-serif"})

plt.switch_backend('agg')



def plot_contamination_breakdown_single_scenario(folder: Path,
                                                 input_file_name: str,
                                                 save_folder: Path,
                                                 original_container_folder: Optional[Path]=None,
                                                 original_container_file: Optional[str]=None
                                                 ) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.
    original_container_folder : Path, optional
        DESCRIPTION. The default is None.
    original_container_file : str, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None

    """
    try:
        data_file = folder / f"containers_{input_file_name}"
        pd_obj = pd.read_csv(data_file)

        total_contamination = (pd_obj[
                                pd_obj['contamination'].isin(["yes", "HLC", "LLC", "Pest"])
                                    ]).shape[0]
        total_contamination_found = pd_obj[pd_obj['detected']].shape[0]

        df_nodes = pd.read_csv(folder / f"nodes_{input_file_name}")

        try:
            treated = df_nodes['treated'].sum()
            # if things were treated, need to fix the total contamination value
            original_containers_file = original_container_folder / original_container_file
            original_containers = pd.read_csv(original_containers_file)
            total_contamination = (original_containers[
                    original_containers['contamination'].isin(["yes", "HLC", "LLC", "Pest"])
                                    ]).shape[0]
        except Exception:
            treated = 0

        leakage = total_contamination - total_contamination_found - treated
    except Exception:
        # if we don't have any container files, we should have summary files:
        data_file = folder / f"summary_{input_file_name}"
        pd_obj = pd.read_csv(data_file)
        total_contamination = pd_obj['total contamination'].sum()
        total_contamination_found = pd_obj['contamination found'].sum()
        leakage = pd_obj['contamination leakage'].sum()
        treated = pd_obj['contamination treated'].sum()

    if treated > 0:
        pie = [total_contamination_found, leakage, treated]
        pie_labels = ["Found", "Undetected", "Treated"]
        pie_colours = ["#084081", "#4495ab", "#6cad7e"]
    else:
        pie = [total_contamination_found, leakage]
        pie_labels = ["Found", "Undetected",]
        pie_colours = ["#084081", "#4495ab"]


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
        if input_file_name != "goods_future.csv":
            return f"{pct:.1f}%\n({absolute:d} containers)"
        return f"{pct:.1f}%\n({absolute:d} consignments)"

    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    _, _, autotexts = ax0.pie(pie,
                             labels=pie_labels,
                             autopct=lambda pct: helper(pct, pie),
                             colors=pie_colours,
                             shadow=False,
                             startangle=90,
                             wedgeprops = {"edgecolor" : "white",
                                           "linewidth": 2,
                                           "antialiased": True},
                             textprops={"fontsize": 14})

    for autotext in autotexts:
        autotext.set_color("white")

    # ax.xaxis.set_label_position('top')
    # ax.set_xlabel('Contaminated containers')
    fig.suptitle("Contaminated containers")
    if input_file_name == "nodes_goods_future.csv":
        ax0.set_xlabel("Contaminated lines (consignments)")

    if input_file_name != "external_future_Treatment_CAL_medium_effect_0.5.csv":
        ax0.set_ylim(top=1)

    save_file = save_folder / input_file_name[:-4]

    plt.savefig(f"{save_file}_contamination_breakdown.png", bbox_inches="tight", transparent=True)
    plt.close()


def print_report_card(folder: Path,
                      input_file_name: str,
                      save_folder: Path,
                      original_container_folder: Path,
                      original_container_file: str) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.
    original_container_folder : Path
        DESCRIPTION.
    original_container_file : str
        DESCRIPTION.

    Returns
    -------
    None

    """
    try:
        data_file = folder / f"containers_{input_file_name}"
        pd_obj = pd.read_csv(data_file)

        total_containers = pd_obj.shape[0]
        total_contamination = (pd_obj[
                                pd_obj['contamination'].isin(["yes", "HLC", "LLC", "Pest"])
                                    ]).shape[0]
        total_contamination_found = pd_obj[pd_obj['detected']].shape[0]

        df_nodes = pd.read_csv(folder / f"nodes_{input_file_name}")
        total_inspections =  df_nodes['inspections'].sum()

        try:
            treated = df_nodes["treated"].sum()
            # if things were treated, need to fix the total contamination value
            original_containers_file = original_container_folder / original_container_file
            original_containers = pd.read_csv(original_containers_file)
            total_contamination = (original_containers[
                    original_containers['contamination'].isin(["yes", "HLC", "LLC", "Pest"])
                                    ]).shape[0]
        except Exception:
            treated = 0

        leakage = total_contamination - total_contamination_found - treated
    except Exception:
        # if we don't have any container files, we should have summary files:
        data_file = folder / f"summary_{input_file_name}"
        pd_obj = pd.read_csv(data_file)
        total_containers =  pd_obj['total containers'].sum()
        total_contamination = pd_obj['total contamination'].sum()
        total_contamination_found = pd_obj['contamination found'].sum()
        total_inspections = pd_obj['total inspections'].sum()
        leakage = pd_obj['contamination leakage'].sum()
        treated = pd_obj['contamination treated'].sum()

    save_file = save_folder / f"{input_file_name[:-4]}_report_card.txt"

    with open(save_file, mode="w", encoding="utf-8") as fout:
        print("SUMMARY", file=fout)
        print(f"Total Inspections: {int(total_inspections)}", file=fout)
        print(f"Contamination found: {int(total_contamination_found)}", file=fout)
        if treated > 0:
            print(f"Contamination treated: f{int(treated)}", file=fout)
        print(f"Leakage: {int(leakage)} ({leakage / total_containers * 100:.4f}%)", file=fout)


def plot_single_scenario_bar_old(folder: Path, input_file_name: str, save_folder: Path) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    data_file = folder / input_file_name
    pd_obj = pd.read_csv(data_file)

    # columns: node,containers,inspections,contamination found,leakage,treated

    dict_obj = pd_obj.to_dict(orient="records")

    _, ax0 = plt.subplots(1,1, figsize=(5, 5))

    y_pos =  []
    node_names = []
    _y = 0
    for row in dict_obj:
        node_name = row['node'].replace("_", " ")
        # node_name = node_name.replace("%", "\%")
        if node_name == "release":
            continue # skip this
        node_names.append(node_name)

        inspected_contam = row['contamination found']
        inspected_no_contam = row['inspections'] - inspected_contam

        left = 0

        ax0.barh(_y, inspected_contam, left=left, color="#084081")
        left += inspected_contam

        ax0.barh(_y, inspected_no_contam, left=left, color="#90b7e4")
        left += inspected_no_contam

        y_pos.append(_y)
        _y -= 1

    # print(node_names)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(node_names)
    ax0.set_xlabel('Total inspections')

    ax0.legend(["Contamination detected", "No contamination detected"],
              bbox_to_anchor=(1.04, 0),
              loc="lower left",
              borderaxespad=0,
              frameon=False,
              handlelength=0.7)

    ax0.xaxis.grid()
    ax0.set_axisbelow(True)

    save_file = save_folder / input_file_name[:-4]

    plt.savefig(save_file / "_single_scenario_bar.png", bbox_inches='tight', transparent=True)
    plt.close()


def plot_single_scenario_bar(folder: Path, input_file_name: str, save_folder: Path) -> None:
    """
    Produces the single panel bar chart (i.e., no break)

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    data_file = folder / input_file_name
    pd_obj = pd.read_csv(data_file)

    # columns: node,containers,inspections,contamination found,leakage,treated

    dict_obj = pd_obj.to_dict(orient="records")

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 5))

    y_pos =  []
    node_names = []
    _y = 0
    list_of_inspections = []
    list_of_contamination_found = []
    for row in dict_obj:
        # total = row['containers']
        node_name = row['node'].replace("_", " ")
        # node_name = node_name.replace("%", "\%")
        if "release" in node_name.lower():
            continue # skip this
        if "treatment" in row['node'] and "treatment" not in input_file_name:
            continue # also skip this
        node_names.append(node_name)

        inspections = row['inspections']
        if np.isnan(inspections):
            inspections = 0

        inspected_contam = row['contamination found']
        inspected_no_contam = inspections - inspected_contam

        list_of_inspections.append(inspections)
        list_of_contamination_found.append(inspected_contam)

        left = 0

        ax0.barh(_y, inspected_contam, left=left, color="#084081")
        left += inspected_contam

        ax0.barh(_y, inspected_no_contam, left=left, color="#90b7e4")
        left += inspected_no_contam

        ax0.text(inspections + 20, _y, f"{int(inspections)}", color='black', va="center")
        y_pos.append(_y)
        _y -= 1

    # print(node_names)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(node_names)
    fig.suptitle("Total inspections")

    ax0.legend(["Contamination detected", "No contamination detected"],
              bbox_to_anchor=(1.04, 0),
              loc="lower left",
              borderaxespad=0,
              frameon=False,
              handlelength=0.7)

    ax0.xaxis.grid()
    ax0.set_axisbelow(True)

    save_file = save_folder / input_file_name[:-4]

    plt.savefig(f"{save_file}_s_sc_bar.png", bbox_inches="tight", transparent=True)
    plt.close()


def find_gap(list_of_numbers: List[int], cut_point: Optional[int]=4_000) -> Tuple[int]:
    """
    Function description

    Parameters
    ----------
    list_of_numbers : List[int]
        DESCRIPTION.
    cut_point : Optional[int], optional
        DESCRIPTION. The default is 4_000.

    Returns
    -------
    Tuple[int]
        DESCRIPTION.

    """
    logger.info(f"\nfind_gap() called with cut_point = {cut_point}")
    try:
        halfway = cut_point

        numbers_below = [ ]
        numbers_above = [ ]

        for num in list_of_numbers:
            if num < halfway:
                numbers_below.append(num)
            else:
                numbers_above.append(num)

        low_lim = max(numbers_below)
        high_lim = min(numbers_above)
    except ValueError as _e:
        logger.error(f"\n{_e}\ncut_point = {cut_point}")
        # call itself recursively until no more error:
        low_lim, high_lim = find_gap(list_of_numbers, halfway - 1000)

    return low_lim, high_lim


def plot_single_scenario_bar_with_break(folder: Path,
                                        input_file_name: str,
                                        save_folder: Path) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    data_file = folder / input_file_name
    pd_obj = pd.read_csv(data_file)

    # columns: node,containers,inspections,contamination found,leakage,treated

    dict_obj = pd_obj.to_dict(orient="records")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    list_of_inspections = []
    list_of_contamination_found = []
    ####### currrently need to run this first to calculate high and low lim first...
    for row in dict_obj:
        if "release" in row['node'].lower():
            continue # skip this
        if "treatment" in row['node'] and "treatment" not in input_file_name:
            continue # also skip this
        inspected_contam = row['contamination found']
        list_of_inspections.append(row['inspections'])
        list_of_contamination_found.append(inspected_contam)

    low_lim, high_lim = find_gap(list_of_inspections + list_of_contamination_found)
    print(low_lim)
    print(high_lim)

    y_pos =  []
    node_names = []
    _y = 0
    for row in dict_obj:
        # total = row['containers']
        node_name = row['node'].replace("_", " ")
        # node_name = node_name.replace("%", "\%")
        if "release" in node_name.lower():
            continue # skip this
        if 'treatment' in row['node'] and 'treatment' not in input_file_name:
            continue # also skip this
        node_names.append(node_name)

        inspections = row['inspections']
        if np.isnan(inspections):
            inspections=0

        inspected_contam = row['contamination found']
        inspected_no_contam = inspections - inspected_contam

        list_of_inspections.append(inspections)
        list_of_contamination_found.append(inspected_contam)

        left = 0

        axs[0].barh(_y, inspected_contam, left=left, color="#084081")
        axs[1].barh(_y, inspected_contam, left=left, color="#084081")
        left += inspected_contam

        axs[0].barh(_y, inspected_no_contam, left=left, color="#90b7e4")
        axs[1].barh(_y, inspected_no_contam, left=left, color="#90b7e4")
        left += inspected_no_contam

        if inspections <= low_lim:
            axs[0].text(inspections + 20, _y , f"{int(inspections)}", color="black", va="center")
        else:
            axs[1].text(inspections + 20, _y , f"{int(inspections)}", color="black", va="center")
        y_pos.append(_y)
        _y -= 1

    # print(node_names)
    axs[0].set_yticks(y_pos)
    axs[0].set_yticklabels(node_names)
    fig.suptitle("Total inspections")
    axs[0].set_xlim(0, low_lim + 200)
    if max(list_of_inspections) > 10_000:
        axs[1].set_xlim(left=high_lim - 1_000)
    else:
        axs[1].set_xlim(left=high_lim - 200)

    axs[1].legend(["Contamination detected", "No contamination detected"],
               bbox_to_anchor=(1.04, 0),
               loc="lower left",
               borderaxespad=0,
               frameon=False,
               handlelength=0.7)
    axs[1].legend(["Contamination detected", "No contamination detected"],
               loc="upper center",
               bbox_to_anchor=(0.5, -0.05),
               borderaxespad=0,
               frameon=False,
               handlelength=0.7)

    axs[0].xaxis.grid()
    axs[0].set_axisbelow(True)

    axs[1].xaxis.grid()
    axs[1].set_axisbelow(True)

    save_file = save_folder / input_file_name[:-4]

    plt.savefig(save_file / "_s_sc_bar_w_ax_break.png", bbox_inches="tight", transparent=True)
    plt.close()


def icl_helper(tot_daff: float,
               tot_aa: float,
               txt_off: int,
               icl: str,
               xlim: int,
               save_file: Path,
               daff_pos: Optional[int]=1,
               aa_pos: Optional[int]=0) -> None:
    """
    Nearly the same lines used for plotting inspections, contaminations and leakages

    Parameters
    ----------
    tot_daff : float
        DESCRIPTION.
    tot_aa : float
        DESCRIPTION.
    txt_off : int
        DESCRIPTION.
    icl : str
        DESCRIPTION.
    xlim : int
        DESCRIPTION.
    save_file : Path
        DESCRIPTION.
    daff_pos : Optional[int], optional
        DESCRIPTION. The default is 1.
    aa_pos : Optional[int], optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None
        DESCRIPTION.

    """
    _, ax0 = plt.subplots(1, 1, figsize=(3, 1.5))

    ax0.barh(daff_pos, tot_daff, color="#185091")
    ax0.barh(aa_pos, tot_aa, color="#c65f75")

    ax0.text(tot_daff + txt_off,
            daff_pos,
            f"{int(tot_daff)}",
            color="#185091",
            va="center")
    ax0.text(tot_aa + txt_off,
            aa_pos,
            f"{int(tot_aa)}",
            color="#c65f75",
            va="center")

    ax0.set_yticks([daff_pos, aa_pos])
    # node_names = [f"DAFF {icl}", "AA {icl}"]
    if icl.lower() == "inspections":
        ax0.set_title("Inspections done by:")
    elif icl.lower() == "contamination":
        ax0.set_title("Contamination found by:")
    elif icl.lower() == "leakage":
        ax0.set_title("Contamination missed by:")
    else:
        logger.error(f"\nThe icl value {icl} is not right.")

    node_names = ["DAFF", "AA"]
    ax0.set_yticklabels(node_names)

    ax0.set_xlim([0, xlim])

    plt.savefig(save_file / f"_single_scenario_bar_summary_{icl}.png",
                bbox_inches="tight",
                transparent=True)
    plt.close()


def plot_single_scenario_bar_summary(folder: Path, input_file_name: str, save_folder: Path) -> None:
    """
    Function description

    Parameters
    ----------
    folder : Path
        DESCRIPTION.
    input_file_name : str
        DESCRIPTION.
    save_folder : Path
        DESCRIPTION.

    Returns
    -------
    None

    """
    data_file = folder / input_file_name
    pd_obj = pd.read_csv(data_file)

    # columns: node,containers,inspections,contamination found,leakage,treated

    dict_obj = pd_obj.to_dict(orient="records")

    tot_daff_inspections = 0
    tot_daff_contamination_found = 0
    tot_daff_leakage = 0

    tot_aa_inspections = 0
    tot_aa_contamination_found = 0
    tot_aa_leakage = 0

    for row in dict_obj:
        # total = row['containers']
        node_name = row['node']
        inspections = row['inspections']
        if np.isnan(inspections):
            inspections = 0
        contam = row['contamination found']
        if np.isnan(contam):
            contam = 0
        leakage = row["leakage"] # note this is leakage by individiual node, NOT the total leakage
        if np.isnan(leakage):
            leakage = 0

        if "AA" in node_name:
            tot_aa_inspections += inspections
            tot_aa_contamination_found += contam
            tot_aa_leakage += leakage
        else:
            tot_daff_inspections += inspections
            tot_daff_contamination_found += contam
            tot_daff_leakage += leakage

    save_file = save_folder / input_file_name[:-4]

    # inspections figure
    icl_helper(tot_daff_inspections, tot_aa_inspections, 20, "inspections", 1e5, save_file)

    # contamination
    icl_helper(tot_daff_contamination_found,
               tot_aa_contamination_found,
               10,
               "contamination",
               2_000,
               save_file)

    # leakage
    icl_helper(tot_daff_leakage, tot_aa_leakage, 10, "leakage", 6_000, save_file)


def run_individual_outputs(conf: Config) -> None:
    """
    function description

    Parameters
    ----------
    conf : Config
        DESCRIPTION.

    Returns
    -------
    None

    """

    list_of_files = [conf.old_wdir_single.name,
                     conf.new_wdir_single.name,
                     f"{conf.new_wdir_single.name}_schs"]

    # needed if treatment occurs
    original_container_folder = conf.vcon_dir
    original_container_file = conf.vcon_file.name

    for  file_name in list_of_files:
        csv_file = file_name + ".csv"
        input_file_name = "nodes_" + csv_file
        # print(input_file_name)
        folder = conf.old_wdir_single.parent / file_name
        save_folder = conf.old_vdir_single.parent / file_name
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=False)

        # the three main ones
        plot_contamination_breakdown_single_scenario(folder,
                                                     csv_file,
                                                     save_folder,
                                                     original_container_folder,
                                                     original_container_file)
        print_report_card(folder,
                          csv_file,
                          save_folder,
                          original_container_folder,
                          original_container_file)
        # plot_single_scenario_bar_with_break(folder, input_file_name,save_folder)
        plot_single_scenario_bar(folder, input_file_name, save_folder)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    run_individual_outputs(Config(ARG["yaml"]))
