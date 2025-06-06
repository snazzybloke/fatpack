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
Utility functions to compute sampling parameters and rates for random sampling ("CCV")
of low-risk containers and risk-based sampling ("CBIS") of higher-risk containers.
"""

from pathlib import Path
import sys
from importlib.resources import files
import pandas as pd
from loguru import logger
from flatpack.src.csp import find_mf
from flatpack.src.inspectors import get_num_ccv_samples
from flatpack import samples

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_approach_rate_atts(fname: Path | str, a_dict: dict[str, list[str | bool]]) -> float:
    """
    Reads the CSV file with synthetic containers (their attributes,
    importer and contamination status) subject to external inspections.
    Depending on the Boolean in the list (i.e., the value of the input dict)
    it will select rows that either match or not match the preceeding string
    for the given key (with the same name as the column used for filtering rows).
    It returns the mean contamination rate for the filtered rows.

    Parameters
    ----------
    fname : Union[Path, str]
        The CSV file with synthetic containers (i.e., their attributes).
    a_dict : Dict[str, List[Union[str, bool]]]
        The keys correspond the chosen container attributes (e.g. 'SCHS'), which can be
        found in the columns of the CSV file with synthetic containers.
        The dict values are lists with 1 (for the column with binary values) or 2 elements
        (for the columns with non-binary values). The rows from the container file are then
        filtered when the column values match the single list value. Or, when the column
        values DON'T match the 1st value in the list (and the 2nd valus is False)

    Returns
    -------
    ... : float
        The mean contamination rate for the filtered rows, i.e., the approach rate.

    """
    cs_df = pd.read_csv(fname, index_col=0)
    # dict keys correspond to selected column names:
    for col, value in a_dict.items():
        if len(value) == 1:
            # append a Boolean True to the list with a single element:
            value.append(True)
        if value[1]:
            # if the Boolean True, keep the rows that match the 0th-element of the value list:
            cs_df = cs_df.query(f'{col} == @value[0]')
        else:
            # keep the rows that DON'T match the 0th-element of the list, e.g., SCHS: ["no", False]:
            cs_df = cs_df.query(f'{col} != @value[0]')
    return (cs_df['contamination'] == 'yes').mean()


def get_approach_rate_atts_internal_low_risk(fname: Path | str) -> float:
    """
    Reads the CSV file with synthetic containers (their attributes,
    importer and contamination status) which are subject to internal inspections.
    It returns the mean contamination rate for the filtered rows.

    Parameters
    ----------
    fname : Union[Path, str]
        The CSV file with synthetic containers (i.e., their attributes).

    Returns
    -------
    ... : float
        The mean contamination rate for the filtered rows, i.e., the approach rate.

    """
    cs_df = pd.read_csv(fname, index_col=0)
    # drop the containers with rural destination and those that had a tailgate inspection (I_TG):
    cs_df = cs_df[~cs_df['RuralDest']]
    cs_df = cs_df[~cs_df['I_TG']]
    # drop the containers from the CAL countries or empty (i.e., "no" type of goods)
    cs_df = cs_df[~cs_df['CALSourceCountry'] |
                  (cs_df['TypeOfGoods'].isin(["normal", "exceptions"]))
                  ]
    return (cs_df['contamination'] == 'yes').mean()


def ccv_inspect_from_approach_rate(rate: float, multiplier: int, prior_n: int=10_000
                                   ) -> tuple[int, float, int, int]:
    """
    Evaluates the CCV parameters, namely the acceptable threshold risk T_risk and
    the number of non-compliant inspections prior_y based on prior_n and contamination rate.
    If risk deemed too high, an exception is raised instead. The sample_size is
    obtained from the call to inspectors.get_num_ccv_samples() with the other 3 params

    Parameters
    ----------
    rate : float
        The estimated approach rate (e.g, an average contamination rate).
    multiplier : int
        The factor to multiply the approach rate (to get an estimate of the maximum).
    prior_n : Optional[int], optional
        The prior sample size as the CCV parameter. The default is 10_000.

    Raises
    ------
    ValueError
        If the roughly estimated T_risk exceeds 3%.

    Returns
    -------
    ... : Tuple[int, float, int, int]
        The CCV recommended sample size, the acceptable threshold risk T_risk,
        the number of detected non-compliances prior_y, and prior_n (unchanged)

    """
    # t_risk as a percentage:
    t_risk_raw = 100 * rate * multiplier
    value_lst = (.5, 1, 2, 3, 4, 5)

    if t_risk_raw < 0:
        logger.error(f"t_risk_raw value: {t_risk_raw}")
        raise ValueError('rate must be positive')
    if t_risk_raw > 3:
        logger.error(f"t_risk_raw value: {t_risk_raw}")
        raise ValueError('rate is too high for low risk sampling')

    # From value_lst get the closest/equal_or_higher value to n.
    # An alternative: min(value_lst, key=lambda x: abs(x - n))
    t_risk = min(_x for _x in value_lst if _x >= t_risk_raw)
    # round prior_y to the nearest int:
    prior_y = round(rate * prior_n)

    sample_size = int(get_num_ccv_samples(prior_n, int(prior_y), t_risk))
    return sample_size, t_risk, prior_y, prior_n


def get_sampling_parameters_external_future(containers_file: Path | str,
                                            clearance_n: int=10,
                                            tight_census: int=4,
                                            multiplier: int=2
                                            ) -> tuple[float, int, float, int, int]:
    """
    Evaluates the CSP3 and CCV paramaters for external inspection of virtual containers.

    Parameters
    ----------
    containers_file : Union[Path, str]
        The CSV file with synthetic containers (i.e., their attributes).
    clearance_n : int
        Clearance number (CN) for the CSP3 and CSP1 sampling protocols (e.g, 10).
    tight_census : int
        Tight census (TC) number for the CSP3 sampling protocol (typically 4).
    multiplier : int, optional
        A factor that multiples approach rate to estimate the leakage rate. The default is 2.

    Returns
    -------
    ... : Tuple[float, int, float, int, int]
        The CBIS monitoring fraction, the CCV recommended sample size, the acceptable threshold
        risk T_risk, the number of detected non-compliances prior_y, and inspection number prior_n.

    """
    # High risk, the CSP-3 branch (i.e., SCHS yes5/20/50/100 AND from CAL Countries):
    # (SCHS: ["no", False] is simply an inversion of all possible ["yes5/20/50/100", True] cases)
    csp3_dict: dict[str, list[str | bool]] = {'SCHS': ["no", False], 'CALSourceCountry': [True]}
    rate = get_approach_rate_atts(containers_file, csp3_dict)
    csp3_mf = find_mf(max_leakage=rate * multiplier, c_num=clearance_n, tc_num=tight_census)[1]
    logger.info(
        f"\nFor SCHS CSP-3, the approach rate is {rate} and mf={csp3_mf} with cn={clearance_n}")

    # Now for the low risk random sampling branch (urban destination AND from non-CAL sountries):
    dict_random: dict[str, list[str | bool]] = {'RuralDest': [False], 'CALSourceCountry': [False]}
    rate = get_approach_rate_atts(containers_file, dict_random)
    sample_size, t_risk, prior_y, prior_n = ccv_inspect_from_approach_rate(rate, multiplier)
    logger.info(f"\nlow risk pathway approach rate is {rate}, sample size is {sample_size}")
    return csp3_mf, sample_size, t_risk, prior_y, prior_n


def get_sampling_parameters_internal_future(containers_file: Path | str,
                                            clearance_n: int=10,
                                            tight_census: int=4,
                                            multiplier: int=2
                                            ) -> tuple[float, int, float, int, int]:
    """
    Evaluates the CSP3 and CCV paramaters for internal inspection of virtual containers.

    Parameters
    ----------
    containers_file : Union[Path, str]
        The CSV file with synthetic containers (i.e., their attributes).
    clearance_n : int
        Clearance number (CN) for the CSP3 and CSP1 sampling protocols (e.g, 10).
    tight_census : int
        Tight census (TC) number for the CSP3 sampling protocol (typically 4).
    multiplier : int, optional
        A factor that multiples approach rate to estimate the leakage rate. The default is 2.

    Returns
    -------
    ... : Tuple[float, int, float, int, int]
        The CBIS monitoring fraction, the CCV recommended sample size, the acceptable threshold
        risk T_risk, the number of detected non-compliances prior_y, and inspection number prior_n.

    """
    # calculate monitoring fraction for the High-risk containers (the CSP-3 branch)
    csp3_dict: dict[str, list[str | bool]] = {'CALSourceCountry': [True],
                                              'TypeOfGoods': ["none"],
                                              'SCHS': ["no", False]
                                              }
    rate = get_approach_rate_atts(containers_file, csp3_dict)
    csp3_mf = find_mf(max_leakage=rate * multiplier, c_num=clearance_n, tc_num=tight_census)[1]
    logger.info(
        f"\nFor CAL empty CSP-3 the approach rate is {rate} and mf={csp3_mf} with cn={clearance_n}")
    # now get the CCB parameters for the low-risk containers
    rate = get_approach_rate_atts_internal_low_risk(containers_file)
    sample_size, t_risk, prior_y, prior_n = ccv_inspect_from_approach_rate(rate, multiplier)
    logger.info(f"\nlow risk pathway approach rate is {rate}, sample size is {sample_size}")
    return csp3_mf, sample_size, t_risk, prior_y, prior_n


def get_sampling_parameters_goods_future(goods_file: Path | str,
                                         clearance_n: int=10,
                                         tight_census: int=4,
                                         multiplier=2
                                         ) -> tuple[float, int, int, int, float, int, int]:
    """
    Evaluates the CSP3 and CCV paramaters for the inspection of virtual containerised goods.

    Parameters
    ----------
    goods_file : Union[Path, str]
        The CSV file with synthetic containers (i.e., their attributes).
    clearance_n : int
        Clearance number (CN) for the CSP3 and CSP1 sampling protocols (e.g, 10).
    tight_census : int
        Tight census (TC) number for the CSP3 sampling protocol (typically 4).
    multiplier : int, optional
        A factor that multiples approach rate to estimate the leakage rate. The default is 2.

    Returns
    -------
    ... : Tuple[float, int, float, int, int]
        The CBIS monitoring fraction for high risk goods, the CCV recommended sample sizes for 3
        non-high risk types of goods, the acceptable threshold risk T_risk, the number of detected
        non-compliances prior_y, and inspection number prior_n.

    """
    # calculate aaproach rate and monitoring fraction for the high-risk CSP-3 branch
    dict_csp3: dict[str, list[str | bool]] = {'Greenlane': ["No"],
                                              'Risk': ["High"],
                                              'Offshore_measure': ["No"]
                                              }
    rate = get_approach_rate_atts(goods_file, dict_csp3)
    csp3_mf = find_mf(max_leakage=rate * multiplier, c_num=clearance_n, tc_num=tight_census)[1]
    logger.info(f"""\nFor non-greenlane, high risk, no offshore measures CSP-3,
                the approach rate: {rate} and mf: {csp3_mf} with cn: {clearance_n}""")

    # Calculate approach rate and CCV sample size for the green lane branch
    dict_random: dict[str, list[str | bool]] = {'Greenlane': ["Yes"]}
    rate = get_approach_rate_atts(goods_file, dict_random)
    samsize_greenlane, _, _, _ = ccv_inspect_from_approach_rate(rate, multiplier)
    logger.info(f"\ngreenlane pathway approach rate: {rate}, sample size: {samsize_greenlane}")

    # Calculate approach rate and CCV sample size for the assurance branch
    dict_random = {'Greenlane': ["No"], 'Risk': ["High"], 'Offshore_measure': ["Yes"]}
    rate = get_approach_rate_atts(goods_file, dict_random)
    samsize_assurance, _, _, _ = ccv_inspect_from_approach_rate(rate, multiplier)
    logger.info(f"\nassurance pathway approach rate: {rate}, sample size: {samsize_assurance}")

    # Calculate approach rate and CCV sample size for the low risk branch
    dict_random = {'Greenlane': ["No"], 'Risk': ["Low"]}
    rate = get_approach_rate_atts(goods_file, dict_random)
    samsize_lowrisk, t_risk, prior_y, prior_n = ccv_inspect_from_approach_rate(rate, multiplier)
    logger.info(f"\nlow risk pathway approach rate: {rate}, sample size: {samsize_lowrisk}")

    return csp3_mf, samsize_greenlane, samsize_assurance, samsize_lowrisk, t_risk, prior_y, prior_n


if __name__ == "__main__":
    MULTIPLIER = 2
    """
    Below are four simple demos of this module for the external container inspections,
    internal container inspections, inspections of containerised goods (using risk by
    tariff only and by tariff and source country, respectively)
    """
    EXT_CSV = "original_ext_int_containers_external_with_medium.csv"
    """
    - get_sampling_parameters_external_future() first calls get_approach_rate_atts()
    which reads the CSV file with 100k synthetic containers (their attributes,
    importer and contamination status). Depending on the Boolean in the list
    (i.e., the value of the input dict) it will select rows that either match or
    not match the preceeding string (e.g., "yes" or "no") for the given key
    (with the same name as the column used for filtering rows).
    The 1st call is with the dict set up to select all containers from the CAL
    countries that might have had SCHS applied. The approach rate is returned
    as the mean contamination across these rows. Then the monit_frac is obtained
    from csp.find_md() using an estimate of the max_leakage (the approach rate times
    a factor). The 2nd call to get_approach_rate_atts() is with a dict set up for
    random sampling (CCV) to select all non-CAL containers with urban destination
    and evaluate the approach rate from their mean contamination. The CCV params
    i.e.,sample_size, t_risk, prior_y and prior_n, are returned from calling
    ccv_inspect_from_approach_rate().
    (NOTE: prior_n is either set as the default value 10k or assigned a different
    value by the user, but the value is NEVER ALTERED in this file)
    - ccv_inspect_from_approach_rate() estimates prior_y and t_risk parameters based
    on the approach rate, while sample_size is obtained from the call to
    inspectors.get_num_ccv_samples() with the other 3 params.
    """
    get_sampling_parameters_external_future(containers_file=files(samples).joinpath(EXT_CSV),
                                            multiplier=MULTIPLIER)

    INT_CSV = "int_no_linehazard_containers_internal_with_medium.csv"
    """
    - get_sampling_parameters_internal_future() first calls get_approach_rate_atts()
    which reads the CSV file with 100k synthetic containers (their attributes,
    importer and contamination status). The 1st call is with the dict set up to
    select CAL countries, TypeOfGoods "none", and SCHS "yes5/20/50/100". The rate
    of approach is calculated as the mean contamination rate of the selected rows.
    The 2nd call is to get_approach_rate_atts_internal_low_risk() which select
    containers with urban desiination, I_TG False, from either non-CAL countries
    or with TypeOfGoods "normal" or "exceptions". The approach rate is taken
    as their mean contamination. The CCV params i.e., sample_size, t_risk, prior_y
    and prior_n, are returned from calling ccv_inspect_from_approach_rate().
    The latter estimates prior_y and t_risk based on the approach rate, while sample_size is
    obtained from the call to inspectors.get_num_ccv_samples() with the other 3 params.
    """
    get_sampling_parameters_internal_future(containers_file=files(samples).joinpath(INT_CSV),
                                            multiplier=MULTIPLIER)

    GTO_CSV = "goods_tariff_only_goods.csv"
    get_sampling_parameters_goods_future(goods_file=files(samples).joinpath(GTO_CSV),
                                         multiplier=MULTIPLIER)

    GTC_CSV = "goods_tariff_country_goods.csv"
    get_sampling_parameters_goods_future(goods_file=files(samples).joinpath(GTC_CSV),
                                         multiplier=MULTIPLIER)
