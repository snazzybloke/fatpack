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
Contains the code for the inspection pathways

This file contains the construction for the different current and future state
pathways/policies, which can be called to simulate.

Running: python inspection_pathways.py -y a_input_yaml_file
or:      python inspection_pathways.py --yaml a_input_yaml_file
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, TypedDict, Unpack
from loguru import logger

from flatpack.src.config import Config
from flatpack.src.policy import (RootNode,
                                  PathwayNode,
                                  TreatmentNode,
                                  create_policy,
                                  simulate_policy)
from flatpack.src.fun_decision import (basic_inspect_node,
                                        random_decision_node,
                                        by_number_decision_node,
                                        multiple_attrs_decision_node,
                                        attr_decision_node,
                                        cbis_node,
                                        cbis_node_mixed,
                                        ccv_decision_node,
                                        release)
from flatpack.src.csp import find_mf
from flatpack.src.inspectors import get_num_ccv_samples
from flatpack.src.ccv_cbis_samplings import (get_approach_rate_atts,
                                              ccv_inspect_from_approach_rate)

logger.remove()
logger.add(sys.stderr, level="INFO")


def external_risks_policy_current(name: str, vcon_file: Path, **kwargs: float
                                  ) -> tuple[str, RootNode, dict[str, PathwayNode | TreatmentNode]]:
    """
    Implements the current DAFF risk management policy of the external container contamination.
    If csp_instead_of_schs selected, the SCHS-stratified inspection scheme is replaced
    with a CSP3 (CBIS-like) protoco and the monitoring fraction defined by cbis_mf.

    Parameters
    ----------
    name : Optional[str], optional
        The formal name assigned to the policy. The default is 'External Risks Current'.
    vcon_file: Path
         A DUMMY (unused) argument in this function, added to have the same API as ...policy_new()
    schs_daff_sensitivity : Optional[float], optional
        Inspection effectiveness of DAFF inspections on the CAL-SCHS containers. Default 0.93
    cal_aa_inspect_fraction : Optional[float], optional
        The proportion of CAL containers directed to AA (the rest to DAFF). The default is 0.4.
    cal_tailgate_daff_sensitivity : Optional[float], optional
        Inspection effectiveness of DAFF tailgate inspections on CAL containers. Default 0.8
    cal_tailgate_aa_sensitivity : Optional[float], optional
        Inspection effectiveness of AA tailgate inspections on CAL containers. The default is 0.6.
    daff_ccv_sensitivity : Optional[float], optional
        Inspection effectiveness of DAFF CCV inspections. The default is 0.3.
    rural_tailgate_aa_sensitivity : Optional[float], optional
        Inspection effectiveness of AA rural tailgate inspections, non-CAL containers. Default 0.6
    rural_tailgate_daff_sensitivity : Optional[float], optional
        Inspection effectiveness of DAFF rural tailgate inspections, non-CAL containers. Default 0.8
    cbis_mf : Optional[float], optional
        The proportion of CAL containers inspected in CBIS monitoring mode. The default is 0.1.
    csp_instead_of_schs : Optional[bool], optional
        Use CBIS (e.g. CSP1 or 3) instead of SCHS for CAL inspections. The default is False.

    Returns
    -------
    Tuple[str, RootNode, Dict[str, PathwayNode]]
        Policy name, RootNode, and a dictionary of PathwayNodes

    """
    logger.warning(f"The parameter {vcon_file} is unused in this function.")
    cal_aa_tailgate_inspect = basic_inspect_node("7. AA rural tailgate (CAL)", 1,
                                                kwargs.get("cal_tailgate_aa_sensitivity", 0.0))
    cal_daff_tailgate_inspect = basic_inspect_node("6. DAFF rural tailgate (CAL)", 1,
                                                  kwargs.get("cal_tailgate_daff_sensitivity", 0.0))
    cal_aa_inspect_decision = random_decision_node("AA inspect yes/no",
                                                   kwargs.get("cal_aa_inspect_fraction", 0.0),
                                                   cal_aa_tailgate_inspect,
                                                   cal_daff_tailgate_inspect)
    release_cal = release('release')

    full_and_rural_decision = \
        multiple_attrs_decision_node("Full and rural yes/no",
                                     {'RuralDest': True, 'TypeOfGoods': "normal"},
                                     yes_node=cal_aa_inspect_decision,
                                     no_node=release_cal)
    schs_daff_sensitivity: float = kwargs.get("schs_daff_sensitivity", 0.0)
    schs_no = basic_inspect_node(
            "5. No SCHS 100%", 1, sensitivity=schs_daff_sensitivity, child=full_and_rural_decision)

    if not kwargs.get("csp_instead_of_schs"):
        schs_5 = basic_inspect_node(
            "1. SCHS 5%", .05, sensitivity=schs_daff_sensitivity, child=full_and_rural_decision)
        schs_20 = basic_inspect_node(
            "2. SCHS 20%", .20, sensitivity=schs_daff_sensitivity, child=full_and_rural_decision)
        schs_50 = basic_inspect_node(
            "3. SCHS 50%", .50, sensitivity=schs_daff_sensitivity, child=full_and_rural_decision)
        schs_100 = basic_inspect_node(
            "4. SCHS 100%", 1, sensitivity=schs_daff_sensitivity, child=full_and_rural_decision)
        schs_decision = attr_decision_node("schs intervention level", attr_name='SCHS',
                                                vals_to_nodes={"yes5": schs_5,
                                                                "yes20": schs_20,
                                                                "yes50": schs_50,
                                                                "yes100": schs_100,
                                                                "no": schs_no})
    else:
        cbis_cn: int = 10
        cbis_tc: int = 4
        cbis_mf: float = kwargs['cbis_mf'] if kwargs.get("cbis_ms") else 0.1
        logger.info(f"\nChosen monitoring fraction for SCHS CSP-3: {cbis_mf}")
        logger.info(f"""\nCSP-3 params:
                    census number (clearance number) {cbis_cn},
                    monitoring fraction {cbis_mf},
                    qtight census number {cbis_tc}""")
        cal_cbis = cbis_node("1-4. (SCHS) CSP-3 inspection",
                             sensitivity=schs_daff_sensitivity,
                             clearance_num=cbis_cn,
                             monit_frac=cbis_mf,
                             child_node=full_and_rural_decision)
        schs_decision = attr_decision_node("SCHS yes/no",
                                           'SCHS',
                                           {"no": schs_no},
                                           default_node=cal_cbis)

    ccv_inspection = basic_inspect_node("10. CCV",
                                       mon_frac=.005,
                                       sensitivity=kwargs.get("daff_ccv_sensitivity", 0.0))

    rural_aa_tailgate_inspect = basic_inspect_node("9. AA rural tailgate (non-CAL)", 1,
                                                  kwargs.get("rural_tailgate_aa_sensitivity", 0.0))
    rural_daff_tailgate_inspect = basic_inspect_node("8. DAFF rural tailgate (non-CAL)", 1,
                                                 kwargs.get("rural_tailgate_daff_sensitivity", 0.0))
    aa_rural_inspect_decision = random_decision_node("AA inspect yes/no",
                                                     kwargs.get("cal_aa_inspect_fraction", 0.0),
                                                     rural_aa_tailgate_inspect,
                                                     rural_daff_tailgate_inspect)

    rural_dest_decision = attr_decision_node(
        "Rural destination yes/no",
        'RuralDest',
        {True: aa_rural_inspect_decision, False: ccv_inspection})

    cal_decision = attr_decision_node("CAL decision",
                                      'CALSourceCountry',
                                      {True: schs_decision, False: rural_dest_decision})

    if not kwargs.get("csp_instead_of_schs"):
        out_lst = [schs_5, schs_20, schs_50, schs_100]
    else:
        out_lst = [cal_cbis]
    rest_of_output = [schs_no, release_cal, cal_daff_tailgate_inspect, cal_aa_tailgate_inspect,
                      rural_daff_tailgate_inspect, rural_aa_tailgate_inspect, ccv_inspection]
    out_lst.extend(rest_of_output)

    full_policy = create_policy(name, top_node=cal_decision, node_list_for_detailed_output=out_lst)
    return full_policy


class Kwords(TypedDict):
    """
    Type hints for the keyword arguments of external_risks_policy_new()
    """
    schs_daff_sensitivity: float
    cal_aa_inspect_fraction: float
    aa_frac_or_num: float | int
    ccv_frac_or_num: float | int
    cal_tailgate_daff_sensitivity: float
    cal_tailgate_aa_sensitivity: float
    daff_ccv_sensitivity: float
    rural_tailgate_aa_sensitivity: float
    rural_tailgate_daff_sensitivity: float
    cal_aa_inspection_sensitivity: float
    ccv_t_risk: float | None
    ccv_prior_n: int
    ccv_prior_y: int | None
    cbis_mf: float | None
    cbis_cn: int
    cbis_tc: int
    csp_instead_of_schs: bool
    aa_lite_sensitivity: float
    treatment_cal_medium_effect: float
    treatment_non_cal_effect: float
    multiplier: int
    port_monitor: bool
    monitor_column: str | None


def external_risks_policy_new(name:str, vcon_file: Path, **kwargs: Unpack[Kwords]
                              ) -> tuple[str, RootNode, dict[str, PathwayNode | TreatmentNode]]:
    """
    Implements the nex external risks pathway for shipping containers as proposed in Dec 2023.
    In 2025 it was decided to drop the non-CAL container Random survey node (12. Random AA inspec)
    and allow an extra binary column 'Monitor' with True/False values for non-CAL containers.
    To run the pathway in this way one needs to set port_monitor to True in the input YAML config
    and generate the containers with the boolean 'Monitor' column.
    Both the low risk and Monitor=True containers go through the RuralTG and CCV nodes,
    but only the latter can be selected for the Targeted survey, via the addtional decision
    node non_cal_monitor_decision.

    Parameters
    ----------
    Mostly the same as in the external_risk_policy_current() above.
    aa_frac_or_num : float | int
        For Targeted Survey (node 11) and Random Inspection (node 12), sending either number or frac

    Returns
    -------
    Tuple[str, RootNode, Dict[str, PathwayNode]]
        Policy name, RootNode, and a dictionary of PathwayNodes

    """
    # Calculate the parameters for the CSP-3 branch
    if not kwargs.get("cbis_mf"):
        dict_csp3: dict[str, list[str | bool]] = {'SCHS': ["no", False], 'CALSourceCountry': [True]}
        rate: float = get_approach_rate_atts(vcon_file, dict_csp3)
        cbis_mf: float | None = find_mf(max_leakage=rate * kwargs.get("multiplier", 0),
                                        c_num=kwargs.get("cbis_cn", 0),
                                        tc_num=kwargs.get("cbis_tc", 0))[1]
        logger.info(f"\nFor SCHS CSP-3, the approach rate is {rate}")
        logger.info(f"\nThe monitoring frac CBIS_mf: {cbis_mf}")
    else:
        cbis_mf = kwargs["cbis_mf"]
        logger.info(f"\nChosen monitoring fraction for SCHS CSP-3 is {cbis_mf}")

    # Calculate for the low risk random sampling branch
    dict_random: dict[str, list[str | bool]]  = {'RuralDest': [False], 'CALSourceCountry': [False]}
    rate = get_approach_rate_atts(vcon_file, dict_random)
    if not kwargs.get("ccv_t_risk"):
        try:
            ccv_samples, ccv_t_risk, ccv_prior_y, ccv_prior_n = \
                ccv_inspect_from_approach_rate(rate, multiplier=kwargs.get("multiplier", 0))
            logger.info(f"\nLow risk pathway approach rate: {rate}") #, sample size: {ccv_samples}")
        except ValueError as _e:
            logger.error(f"\nFailed to evaluate CCV parameters, maybe not low risk?\n{_e}")
            green_diamond = False   # random decision node will be used
        else:
            green_diamond = True    # CCV decision node will be used
    else:
        green_diamond = True    # CCV decision node will be used
        logger.info(f"\nchosen random sampling T_risk = {kwargs.get('ccv_t_risk')}%")
        ccv_prior_y = round(rate * kwargs.get("ccv_prior_n", 0))
        ccv_samples = get_num_ccv_samples(kwargs.get("ccv_prior_n", 0),
                                          ccv_prior_y,
                                          kwargs.get("ccv_t_risk", 0.0))
        ccv_prior_n = None
        ccv_t_risk = None

    if green_diamond:
        logger.info(f"""\nRandom sampling vol: {ccv_samples}, with prior samples = {ccv_prior_n}
                    prior contamination = {ccv_prior_y}, and T_risk = {ccv_t_risk}""")

    logger.info(f"""\nCSP-3 params: census number (clearance number) {kwargs.get('cbis_cn')},
                monitoring fraction {cbis_mf}, tight census number {kwargs.get('cbis_tc')}""")

    # 1st: CAL + SCHS pathways ###############################################:
    # Level 7a, two inspection nodes:
    cal_full_rural_daff_inspect = basic_inspect_node("6. CAL full rural DAFF inspection", 1,
                                                  kwargs.get("cal_tailgate_daff_sensitivity", 0.0))
    cal_full_rural_aa_inspect = basic_inspect_node("7. CAL full rural AA inspection", 1,
                                                  kwargs.get("cal_tailgate_aa_sensitivity", 0.0))
    # Level 6a, the associated (with Level 7a) decision node:
    cal_full_rural_aa_decision = random_decision_node("CAL_full_rural_AA_decision",
                                                      kwargs.get("cal_aa_inspect_fraction", 0.0),
                                                      yes_node=cal_full_rural_aa_inspect,
                                                      no_node=cal_full_rural_daff_inspect)
    # Level 6a, release "node":
    cal_release = release("CAL release")
    # Level 5a, the associated (with Level 6a) decision node:
    cal_full_rural_decision = multiple_attrs_decision_node(
        "full/rural/no inspect decision",
        {'RuralDest': True, 'TypeOfGoods': 'normal',
         'inspected': False},
        yes_node=cal_full_rural_aa_decision,
        no_node=cal_release)
    # Level 4a, two CBIS inspection nodes (each with Level 5a child decision node):
    schs_daff_sensitivity: float = kwargs.get("schs_daff_sensitivity", 0.0)
    if not kwargs.get("csp_instead_of_schs"):
        schs_5 = basic_inspect_node(
            "1. SCHS 5%", .05, sensitivity=schs_daff_sensitivity, child=cal_full_rural_decision)
        schs_20 = basic_inspect_node(
            "2. SCHS 20%", .20, sensitivity=schs_daff_sensitivity, child=cal_full_rural_decision)
        schs_50 = basic_inspect_node(
            "13. SCHS 50%", .50, sensitivity=schs_daff_sensitivity, child=cal_full_rural_decision)
        schs_100 = basic_inspect_node(
            "14. SCHS 100%", 1, sensitivity=schs_daff_sensitivity, child=cal_full_rural_decision)
    else:
        senss: dict[str, tuple[float, ...]] = {
                "DAFF": (schs_daff_sensitivity, 1.0 - kwargs.get("cal_aa_inspect_fraction", 0.0)),
                "AA": (kwargs.get("cal_aa_inspection_sensitivity", 0.0),
                       kwargs.get("cal_aa_inspect_fraction", 0.0))}
        cal_cbis_med = cbis_node_mixed("1. (SCHS) CSP-3 inspection",
                                       senss,
                                       clearance_num=kwargs.get("cbis_cn", 0),
                                       monit_frac=cbis_mf,
                                       tight_cens_num=kwargs.get("cbis_tc", 0),
                                       child_node=cal_full_rural_decision,
                                       csp3=True)
        cal_cbis_hi = cbis_node("2. (SCHS) CSP-3 inspection",
                                sensitivity=schs_daff_sensitivity,
                                clearance_num=kwargs.get("cbis_cn", 0),
                                monit_frac=cbis_mf,
                                child_node=cal_full_rural_decision)
        # Level 3a, the associated (with Level 4a) decision node:
        cal_hi_med_decision = attr_decision_node(
            "CAL Medium y/n",
            'CALSourceCountryMedium',
            {False: cal_cbis_hi, True: cal_cbis_med})

    # 2nd: CAL, but no SCHS, pathways ########################################:
    # Level 5b, two inspection nodes:
    cal_medium_inspection = basic_inspect_node("3. CAL-medium DAFF inspection", 1,
                                              schs_daff_sensitivity)
    cal_aa_inspection = basic_inspect_node("4. CAL-medium AA inspection", 1,
                                          kwargs.get("cal_aa_inspection_sensitivity", 0.0))
    # Level 4b, the associated (with Level 5b) decision node:
    cal_medium_aa_decision = random_decision_node("CAL medium AA decision",
                                                  kwargs.get("cal_aa_inspect_fraction", 0.0),
                                                  yes_node=cal_aa_inspection,
                                                  no_node=cal_medium_inspection)
    # Level 4b, inspection nodee:
    cal_high_inspection = basic_inspect_node("5. CAL-high DAFF inspection", 1,
                                            schs_daff_sensitivity)
    # Level 3b, the associated (with Level 4b) decision node:
    cal_high_medium_decision = attr_decision_node(
        "CAL medium yes/no",
        'CALSourceCountryMedium',
        {False: cal_high_inspection, True: cal_medium_aa_decision})

    # Level 2, joining the Levela 3a and 3b above ****************************:
    if not kwargs.get("csp_instead_of_schs"):
        schs_decision = attr_decision_node("schs intervention level", attr_name='SCHS',
                                                vals_to_nodes={"yes5": schs_5,
                                                                "yes20": schs_20,
                                                                "yes50": schs_50,
                                                                "yes100": schs_100,
                                                                "no": cal_high_medium_decision})
    else:
        schs_decision = attr_decision_node("SCHS yes/no",
                                           'SCHS', {"no": cal_high_medium_decision},
                                           default_node=cal_hi_med_decision)
    # non-CAL RuralDestination ###############################################:
    # Level 5c, two inspection nodes::
    non_cal_tailgate_daff_inspect = basic_inspect_node("8. Non-CAL DAFF rural tailgate", 1,
                                                kwargs.get("rural_tailgate_daff_sensitivity", 0.0))
    non_cal_tailgate_aa_inspect = basic_inspect_node("9. Non-CAL AA rural tailgate", 1,
                                                kwargs.get("rural_tailgate_aa_sensitivity", 0.0))
    # Level 4c, the associated (with Level 5b) decision node:
    non_cal_aa_decision = random_decision_node("TG inspect at AA",
                                               kwargs.get("cal_aa_inspect_fraction", 0.0),
                                               non_cal_tailgate_aa_inspect,
                                               non_cal_tailgate_daff_inspect)

    # non-CAL and not RuralDestination #######################################:
    # Level 7d, release "node":
    non_cal_release = release("non-CAL release")
    # Level 7d, AA inspect node (Random survey inspection DROP if port Monitor):
    non_cal_aa_random_inspection = basic_inspect_node("12. Non-CAL random AA inspection", 1,
                                                kwargs.get("cal_aa_inspection_sensitivity", 0.0))
    # Level 6d, the associated (with Level 7d) decision node:
    if isinstance((aa_f_n := kwargs.get("aa_frac_or_num", 0.0)), int) and aa_f_n > 1:
        decis_node: Callable = by_number_decision_node
    else:
        decis_node = random_decision_node
    # Random survey decision, DROP if port Monitor:
    non_cal_random_aa_decision = decis_node("Non-CAL_full_rural_AA_decision",
                                            kwargs.get("aa_frac_or_num"),
                                            yes_node=non_cal_aa_random_inspection,
                                            no_node=non_cal_release)
    # Level 6d, AA inspect node:
    non_cal_aa_survey_inspection = basic_inspect_node("11. Non-CAL survey AA inspection", 1,
                                                kwargs.get("cal_aa_inspection_sensitivity", 0.0))
    # Level 5d, the associated (with Level 6d) decision node:
    if kwargs.get("port_monitor"):
        logger.info("Port-level monitor the containers:")
        non_cal_survey_aa_decision = decis_node("Non-CAL_full_rural_AA_decision",
                                                kwargs.get("aa_frac_or_num"),
                                                yes_node=non_cal_aa_survey_inspection,
                                                no_node=non_cal_release)
        non_cal_monitor_decision = attr_decision_node(
            "Non-CAL monitor yes/no",
            kwargs.get("monitor_column", ""),
            {False: non_cal_release, True: non_cal_survey_aa_decision})
    else:
        logger.info("No port-level monitor for the containers:")
        non_cal_survey_aa_decision = decis_node("Non-CAL_full_rural_AA_decision",
                                                kwargs.get("aa_frac_or_num"),
                                                yes_node=non_cal_aa_survey_inspection,
                                                no_node=non_cal_random_aa_decision)
        # proceed to Level 5d if no port Monitor
        non_cal_monitor_decision = non_cal_survey_aa_decision

    # Level 5d, CCV inspect node:
    ccv_inspect = basic_inspect_node("10. Random CCV sampling inspection", 1,
                                    sensitivity=kwargs.get("daff_ccv_sensitivity", 0.0))
    # Level 4d, the associated (with Level 5d) decision node:
    if green_diamond:
        non_cal_ccv_decision = ccv_decision_node("CCV non CAL",
                                                 yes_node=ccv_inspect,
                                                 # no_node=non_cal_survey_aa_decision,
                                                 no_node=non_cal_monitor_decision,
                                                 prior_n=ccv_prior_n,
                                                 prior_y=ccv_prior_y,
                                                 t_risk=ccv_t_risk)
    else:
        if isinstance(kwargs.get("ccv_frac_or_num"), int) and kwargs.get("ccv_frac_or_num") > 1:
            ccv_decis_node: Callable = by_number_decision_node
        else:
            ccv_decis_node = random_decision_node
        non_cal_ccv_decision = ccv_decis_node("In lieu of CCV non CAL",
                                              kwargs.get("ccv_frac_or_num"),
                                              yes_node=ccv_inspect,
                                              # no_node=non_cal_survey_aa_decision)
                                              no_node=non_cal_monitor_decision)

    # Level 3, joining the Levels 4c and 4d above ****************************:
    rural_dest_decision = attr_decision_node(
        "Rural destination", 'RuralDest', {True: non_cal_aa_decision, False: non_cal_ccv_decision})
    # non-CAL and empty:
    # Level 3, release "node":
    non_cal_empty_release = release("non-CAL empty release")
    # Level 2, joining the Level 3 nodes above:
    full_decision = attr_decision_node("Loaded yes/no",
                                       'TypeOfGoods', {"none": non_cal_empty_release},
                                       default_node=rural_dest_decision)
    # Level 1, TOP node, joining the Level 2 nodes above (namely schs_decision and full_decision):
    cal_decision = attr_decision_node(
        "CAL yes/no", 'CALSourceCountry', {True: schs_decision, False: full_decision})

    nodes = [cal_full_rural_daff_inspect, cal_full_rural_aa_inspect,
            cal_medium_inspection, cal_aa_inspection, cal_high_inspection, cal_release,
            non_cal_tailgate_daff_inspect, non_cal_tailgate_aa_inspect, non_cal_release,
            non_cal_aa_survey_inspection, ccv_inspect, non_cal_empty_release]
    if not kwargs.get("port_monitor"):
        nodes.insert(9, non_cal_aa_random_inspection)
    if not kwargs.get("csp_instead_of_schs"):
        out_lst = [schs_5, schs_20, schs_50, schs_100]
    else:
        out_lst = [cal_cbis_med, cal_cbis_hi]
    nodes.extend(out_lst)

    full_policy = create_policy(name, top_node=cal_decision, node_list_for_detailed_output=nodes)
    return full_policy


def run_helper(ext_pol: Callable, wdir: Path, vcon_file: Path, **kwargs) -> None:
    """
    Replacing the repetitive half a dozen lines used in every run...() function below

    Parameters
    ----------
    ext_pol : Callable
        The function that implements the state of pathway of interest.
        Either external_risks_policy_new() or external_risks_policy_current()
    wdir : Union[str, Path]
        The name (suffix) of the output folder (different for each external_policy_...())
    vcon_file : Path
        The path to the virtual container file.
    **kwargs : str
        Keyword arguments for ext_pol() , see Kwords class.

    Returns
    -------
    None

    """
    try:
        if ext_pol.__name__ == external_risks_policy_current.__name__:
            # default is SCHS for the current ("old") policy:
            out_dir: Path = Path(f"{wdir}_csp") if kwargs.get("csp_instead_of_schs") else wdir
        else:
            # default is CSP3 for the new policy:
            out_dir = wdir if kwargs.get("csp_instead_of_schs") else Path(f"{wdir}_schs")
        policy_name: str = out_dir.name
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
        simulate_policy(*ext_pol(policy_name, vcon_file, **kwargs),
                        cs_fname=vcon_file,
                        save_results=kwargs.get("save_results", True),
                        save_container_results=kwargs.get("save_container_results", True),
                        save_summary=kwargs.get("save_summary", True),
                        output_dir=out_dir)
    except FileNotFoundError as _e:
        logger.exception(f"\nCouldn't open the folder/file: {_e}")


def run_external_policies_current(conf: Config) -> None:
    """
    Calls run_helper to execute the old/current state policy with the given set of parameters.

    Parameters
    ----------
    conf : Config
        The parameters from the input YAML file.

    Returns
    -------
    None

    """
    run_helper(external_risks_policy_current,
               conf.old_wdir_single,
               conf.vcon_file,
               **conf.fun_params)


def run_external_policies_new(conf: Config) -> None:
    """
    Calls run_helper to execute the future state policy with the given set of parameters.

    Parameters
    ----------
    conf : Config
        The parameters from the input YAML file.

    Returns
    -------
    None

    """
    run_helper(external_risks_policy_new, conf.new_wdir_single, conf.vcon_file, **conf.fun_params)
    # try the alternate SCHS policy:
    conf.fun_params["csp_instead_of_schs"] = not conf.fun_params["csp_instead_of_schs"]
    run_helper(external_risks_policy_new, conf.new_wdir_single, conf.vcon_file, **conf.fun_params)
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
    # EXTERNAL CONTAMINATION POLICIES, current state
    run_external_policies_current(conf)
    # EXTERNAL CONTAMINATION POLICIES, DAFF's future state
    run_external_policies_new(conf)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    main(Config(ARG["yaml"]))
