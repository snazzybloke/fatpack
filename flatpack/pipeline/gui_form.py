###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ant Bilic                           #
# Email: ante.bilic.mr@gmail.com                  #
# Status: N/A                                     #
###################################################
"""
A simple GUI form to enter simulation input parameters and execute the workflow
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
import yaml
from flatpack22b.src.config import Config
from flatpack22b.pipeline import full_wflow
import streamlit as st
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def run_gui() -> None | dict[str, str | bool | int | float | None | dict]:
    """_
    Opens a simole GUI form in the browser to input simulation paramaters.
    Once the user clicks the sidebar button 'Run" it returns the input config for processing.

    Returns:
    None | confdic: None | dict[str, str | bool | int | float | None | dict]
        Every response triggers function return (None), only when the sidebar button is selected
        the confdic is returned.

    """
    st.title("The flatpack22b input form")
    confdic: dict[str, str | bool | int | float | None | dict] = {}

    # Config inputs section:
    datadir = st.text_input("Data directory path?")
    if datadir:
        st.write(datadir)
    extdir = st.text_input("External contamination subdir?")
    if extdir:
        st.write(extdir)
    cpt_pref: str | None = None
    cpttop = st.file_uploader("CPT top file?", accept_multiple_files=False)
    if cpttop:
        st.write(cpttop)
        cpt_pref = cpttop.name.partition(".")[0]
    datadic: dict[str, str] = {"data_dir": datadir,
                               "ext_dir": extdir,
                               "top_cpt": cpt_pref}
    confdic.update({"inputs": {"data": datadic}})
    calmed_approach = st.slider("CAL Medium approach rate [%]:", 0, 100) / 100.0
    if calmed_approach:
        st.write(calmed_approach)
    calmed_frac = st.slider("CAL Medium fraction [%]:", 0, 100) / 100.0
    if calmed_frac:
        st.write(calmed_frac)
    confdic["inputs"].update({"cal_medium_approach_rate": calmed_approach,
                              "cal_medium_fraction": calmed_frac})
    vcon_dir = st.text_input("Virtual container dir path?")
    if vcon_dir:
        st.write(vcon_dir)
    tot_con = st.number_input("How many virtual containers?")
    if tot_con:
        tot_con = round(tot_con)
        st.write(tot_con)
    num_label = st.text_input("Number label (e.g. 100k)?")
    if num_label:
        st.write(num_label)
    tot_importers = st.number_input("How many 'importers' (or ports/years/labels...)?")
    named: str = "dummy"
    if tot_importers:
        tot_importers = round(tot_importers)
        named = st.radio("Importer kind:", ["dummy", "name/volume pairs"])
        if named == "name/volume pairs":
            name_vol = []
            for i in range(tot_importers):
                name = st.text_input(f"Importer/port/label {i+1} name:")
                vol = st.number_input(f"Importer/port/label {i+1} volume:")
                name_vol.append({"name": name, "volume": vol if vol <= 1 else round(vol)})
            st.write(f"Import volumes {name_vol}")
    confdic["inputs"].update({
        "virtconts": {"vcon_dir": vcon_dir,
                      "tot_containers": tot_con,
                      "num_label": num_label,
                      "tot_importers": tot_importers if named == "dummy" else name_vol}})

    # Config work section:
    workdir = st.text_input("Work directory path?")
    if workdir:
        st.write(workdir)
    visdir = st.text_input("Visualisation directory path?")
    if visdir:
        st.write(visdir)
    single_dir = st.text_input("Single simulation subdirectory name (e.g., single?/raw)?")
    if single_dir:
        st.write(single_dir)
    multi_dir = st.text_input("Multi simulation subdirectory name (e.g., single?/raw)?")
    if multi_dir:
        st.write(multi_dir)
    sim_tot = st.number_input("How many simulation?")
    if sim_tot:
        sim_tot = round(sim_tot)
        st.write(sim_tot)
    old_dir = st.text_input("Current pathway subdirectory name?")
    if old_dir:
        st.write(old_dir)
    new_dir = st.text_input("New/future pathway subdirectory name?")
    if new_dir:
        st.write(new_dir)
    confdic.update({"work": {"w_dir": workdir,
                             "vis_dir": visdir,
                             "single_dir": single_dir,
                             "multi_dir": multi_dir,
                             "sim_total": sim_tot,
                             "old": old_dir,
                             "new": new_dir}})

    # Config sensi_rates section:
    schs_daff_sensitivity = st.slider("SCHS DAFF Sensitivity [%]:", 0, 100) / 100.0
    if schs_daff_sensitivity:
        st.write(schs_daff_sensitivity )
    cal_aa_inspect_fraction = st.slider("CALL AA inspection fraction [%]:", 0, 100) / 100.0
    if cal_aa_inspect_fraction:
        st.write(cal_aa_inspect_fraction)
    aa_frac_or_num = st.number_input("AA fraction (or whole number)?")
    if aa_frac_or_num:
        st.write(aa_frac_or_num)
    ccv_frac_or_num = st.number_input("CCV fraction (or whole number)?")
    if ccv_frac_or_num:
        st.write(ccv_frac_or_num)
    cal_tg_daff_sensitivity= st.slider("CAL Tailgate DAFF Sensitivity [%]:", 0, 100) / 100.0
    if cal_tg_daff_sensitivity:
        st.write(cal_tg_daff_sensitivity)
    cal_tg_aa_sensitivity= st.slider("CAL Tailgate AA Sensitivity [%]:", 0, 100) / 100.0
    if cal_tg_aa_sensitivity:
        st.write(cal_tg_aa_sensitivity)
    daff_ccv_sensitivity= st.slider("DAFF CCV Sensitivity [%]:", 0, 100) / 100.0
    if daff_ccv_sensitivity:
        st.write(daff_ccv_sensitivity)
    rural_tg_aa_sensitivity= st.slider("Rural Tailgate AA Sensitivity [%]:", 0, 100) / 100.0
    if rural_tg_aa_sensitivity:
        st.write(rural_tg_aa_sensitivity)
    rural_tg_daff_sensitivity= st.slider("Rural Tailgate DAFF Sensitivity [%]:", 0, 100) / 100.0
    if rural_tg_daff_sensitivity:
        st.write(rural_tg_daff_sensitivity)
    cal_aa_inspection_sensitivity= st.slider("CAL AA Sensitivity [%]:", 0, 100) / 100.0
    if cal_aa_inspection_sensitivity:
        st.write(cal_aa_inspection_sensitivity)
    ccv_t_risk = st.number_input("CCV T_risk (optional):")  # null
    if ccv_t_risk:
        st.write(ccv_t_risk)
    ccv_prior_n = st.number_input("CCV prior_n (e.g., 10000):")
    if ccv_prior_n:
        ccv_prior_n = round(ccv_prior_n)
        st.write(ccv_prior_n)
    ccv_prior_y = st.number_input("CCV prior_y (optional)")   # null
    if ccv_prior_y:
        ccv_prior_y = round(ccv_prior_y)
        st.write(ccv_prior_y)
    cbis_mf = st.number_input("CBIS monitoring fraction (optional):")  # null
    if cbis_mf:
        st.write(cbis_mf)
    cbis_cn = st.number_input("CBIS clearance number(e.g., 10):")
    if cbis_cn:
        cbis_cn = round(cbis_cn)
        st.write(cbis_cn)
    cbis_tc = st.number_input("CBIS tight census number (e.g. 4)")
    if cbis_tc:
        cbis_tc = round(cbis_tc)
        st.write(cbis_tc)
    csp_instead_of_schs = st.radio("CSP instead of SCHS:", [False, True])
    if csp_instead_of_schs:
        st.write(csp_instead_of_schs)
    aa_lite_sensitivity = 0.2         # unused
    treatment_cal_medium_effect = 1   # unused
    treatment_non_cal_effect = 1      # unused
    multiplier = st.number_input("Approach rate multiplier (e.g. 2)")
    if multiplier:
        multiplier = round(multiplier)
        st.write(multiplier)
    confdic.update({"sensi_rates": {"schs_daff_sensitivity": schs_daff_sensitivity,
                                    "cal_aa_inspect_fraction": cal_aa_inspect_fraction,
                                    "aa_frac_or_num": aa_frac_or_num,
                                    "ccv_frac_or_num": ccv_frac_or_num,
                                    "cal_tailgate_daff_sensitivity": cal_tg_daff_sensitivity,
                                    "cal_tailgate_aa_sensitivity": cal_tg_aa_sensitivity,
                                    "daff_ccv_sensitivity": daff_ccv_sensitivity,
                                    "rural_tailgate_aa_sensitivity": rural_tg_aa_sensitivity,
                                    "rural_tailgate_daff_sensitivity": rural_tg_daff_sensitivity,
                                    "cal_aa_inspection_sensitivity": cal_aa_inspection_sensitivity,
                                    "ccv_t_risk": ccv_t_risk,
                                    "ccv_prior_n": ccv_prior_n,
                                    "ccv_prior_y": ccv_prior_y,
                                    "cbis_mf": cbis_mf,
                                    "cbis_cn": cbis_cn,
                                    "cbis_tc": cbis_tc,
                                    "csp_instead_of_schs": csp_instead_of_schs,
                                    "aa_lite_sensitivity": aa_lite_sensitivity,
                                    "treatment_non_cal_effect": treatment_non_cal_effect,
                                    "treatment_cal_medium_effect": treatment_cal_medium_effect,
                                    "multiplier": multiplier
                                    }})
    # Config ro_save section:
    save_results = st.radio("Save results?", [False, True])
    if save_results:
        st.write(save_results )
    save_container_results = st.radio("Save container results?", [False, True])
    if save_container_results:
        st.write(save_container_results )
    save_summary = st.radio("Save summary?", [False, True])
    if save_summary:
        st.write(save_summary)
    confdic.update({"to_save": {"save_results": save_results,
                                "save_container_results": save_container_results,
                                "save_summary": save_summary}})

    monitor = st.radio("Port monitoring:", [False, True])
    med_col = None
    if monitor:
        med_col = st.text_input("The monitor column name?")
        st.write(med_col)
    confdic.update({"port_monitor": monitor, "monitor_column": med_col})
    with st.sidebar:
        if st.button("Run the workflow"):
            return confdic


def main(new_yaml: Path) -> None:
    """
    Simply calls the run_gui() function above to receive the input config parameters
    which are then written in the required YAML input file and executes the workflow

    Parameters:

    new_yaml: Path
        the path+name of the input YAML file
    """
    config_dic = run_gui()
    if config_dic:
        logger.info(f"\nInput configuration:\n{config_dic}")
        with open(new_yaml, mode="w", encoding="utf-8") as fout:
            yaml.dump(config_dic, fout, default_flow_style=False)
        logger.info(f"\nFind the YAML input file: {new_yaml}")
        config: Config = Config(new_yaml)
        logger.info("\nStarting the full workflow")
        full_wflow.main(config)


if __name__ == "__main__":
    agp = ArgumentParser()
    agp.add_argument("-yaml", "-y", required=True, help="The path/name of the new YAML input file")
    ARG = vars(agp.parse_args())
    main(Path(ARG["yaml"]))
