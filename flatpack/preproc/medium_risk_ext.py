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
Adds in CAL-medium risk attribute for containers.

This file contains code that takes in the conditional probability tables for
the external container contamination and adds in CAL-medium status (true/false)
and modifies CAL-medium and CAL-high contamination rates
(such that the toal CAL contamination rate is the same).

Running: python medium_risk_ext.py -y a_input_yaml_file
or:      python medium_risk_ext.py --yaml a_input_yaml_file
"""

import argparse
from pathlib import Path
import sys
from math import isclose
import pandas as pd
from pandas.core.frame import DataFrame
from loguru import logger
from flatpack.src.config import Config
from flatpack.preproc.make_containers import TheConGen

logger.remove()
logger.add(sys.stderr, level="INFO")


def obj_setup(data_folder: Path, cpt_top: str) -> tuple[
        DataFrame, list[str]]:
    """
    Reads the top CPT file for the main() function below to setup the required
    TheConGen object and return the required information.

    Parameters
    ----------
    data_folder : Path
        Where the conditional probability table (CPT) file can be found
    cpt_top : str
        The name (prefix, without extension) of the CPT file with 4-attribute proportions
        and associate contamination probabilities

    Returns
    -------
    (Tuple[DataFrame, List[str]])
        The top level CPT DataFrame, and 4+ attribute columns, respectively.

    """
    con_object = TheConGen(cpt_top_file=data_folder / f"{cpt_top}.csv",
                       num_containers=0,
                       num_importers=0)
    # the DataFrame of the cpt_top_file and attrribute columns:
    return con_object.cpt_df, con_object.attribs


def main_helper(cpt_df: DataFrame,
                frac_medium: float) -> DataFrame:
    """
    Simply breaking up the cognitive complexity of the main() function below.

    Parameters
    ----------
    cpt_df : DataFrame
        CP table as a pandas dataframe.
    frac_medium : float
        The relative proportion of CALMedium risk containers (= Med / (Med + High)

    Returns
    -------
        DataFrame
        A new CPT table with 240 rows and a new attribute binary 'CALSourceCountryMedium'

    """
    # Select the non-CAL rows:
    df_non_cal = cpt_df[~cpt_df['CALSourceCountry']]
    list_of_falses = [False for _ in range(df_non_cal.shape[0])]
    list_of_trues = [True for _ in range(df_non_cal.shape[0])]
    # Assign the CAL...Medium column to [True] for the non-CAL entries:
    cpt_non_cal_true = df_non_cal.copy()
    cpt_non_cal_true = cpt_non_cal_true.assign(CALSourceCountryMedium=list_of_trues)
    # set 'prop'=0 (because CALSourceCountry = False, but CALSourceCountryMedium = True)
    cpt_non_cal_true['prop'] = 0
    # Now assign the CAL...Medium column to [False] for the non-CAL entries:
    cpt_non_cal_false = df_non_cal.copy()
    cpt_non_cal_false = cpt_non_cal_false.assign(CALSourceCountryMedium=list_of_falses)
    # join the two ([False] and [True], with 'prop'=0) tables into a new NON-CAL DataFrame:
    df_non_cal = pd.concat([cpt_non_cal_false, cpt_non_cal_true])

    # Now select the CAL rows:
    df_medium =  cpt_df[cpt_df['CALSourceCountry']]
    # Assign the CAL...Medium column to [True] for the CAL entries:
    df_medium = df_medium.assign(CALSourceCountryMedium=list_of_trues)
    # re-weigh their probabilities:
    df_medium['prop'] = df_medium['prop'] * frac_medium
    # The same rows, but now assigned CAL...Medium = [False]:
    df_high = cpt_df[cpt_df['CALSourceCountry']]
    df_high = df_high.assign(CALSourceCountryMedium=list_of_falses)
    # re-weigh their probabilities:
    df_high['prop'] = df_high['prop'] * (1 - frac_medium)
    # join non_cal, medium and high into a 240-row DataFrame:
    df_new = pd.concat([df_non_cal, df_medium, df_high])
    return df_new


def main(conf: Config) -> None:
    """
    Provides the 120 row CPT with the 'CALSourceCountryMedium' attribute and
    associated 'prop', 'p_yes' (contamination probability) and 'p_no', which can be used
    by make_containers.py to generate virtual containers. It is aved as a CSV file next to
    the top CPT file, but with a suffix '_with_med' in its name.

    Parameters
    ----------
    conf : Config
        The set of parameters from the YAML input file.

    Raises
    ------
    ValueError

    Returns
    -------
    None

    """
    # data_folder = Path.cwd() / "data" / "original_ext_int"
    cpt_df, attribs = obj_setup(conf.ext_dir, conf.cpt_top)
    df_new = main_helper(cpt_df, conf.cal_med_fraction)
    # Select the CAL rows:
    df_cal = df_new[df_new['CALSourceCountry']]
    # Split the CAL rows into  CALCountryMedium == False ...
    df_cal_hi = df_cal[~df_cal['CALSourceCountryMedium']]
    # ... and CALCountryMedium == True
    df_cal_med = df_cal[df_cal['CALSourceCountryMedium']]
    # evaluate the CAL approach rate:
    current_cal_approach_rate: float = df_cal['prop'].dot(df_cal['p_yes'].fillna(0.0)) / \
                                        df_cal['prop'].sum()
    cal_medium_contamination_multiplier: float = conf.cal_med_approach_rate / \
                                                    current_cal_approach_rate
    # record the original cal_medium_contamination amount
    cal_medium_contamination_original: float = df_cal_med['prop'].dot(df_cal_med['p_yes'].fillna(0))
    # For CALCountryMedium == True renormalise 'p_yes' with this scaling factor:
    df_cal_med.loc[:, 'p_yes'] = df_cal_med ['p_yes'] * cal_medium_contamination_multiplier
    # verification:
    if not isclose(df_cal_med['prop'].dot(df_cal_med['p_yes'].fillna(0.0)),
            conf.cal_med_approach_rate * df_cal_med['prop'].sum()):
        raise ValueError("New CAL medium approach rate does not match the assumed")
    # update the cal_medium_contamination amount with the new 'p_yes' column:
    cal_medium_contamination_new: float = df_cal_med['prop'].dot(df_cal_med['p_yes'].fillna(0))
    # record the difference from the original value:
    contamin_loss: float = cal_medium_contamination_original - cal_medium_contamination_new
    # renormalise 'p_yes' with the scaling factor where CALCountryMedium == True:
    df_new.loc[df_new['CALSourceCountry'] & df_new['CALSourceCountryMedium'],'p_yes'] = \
        df_new[df_new['CALSourceCountry'] & df_new['CALSourceCountryMedium']]['p_yes'] * \
            cal_medium_contamination_multiplier

    # Now similarly for CALCountryMedium == False contamination:
    cal_high_contamination_original: float = df_cal_hi['prop'].dot(df_cal_hi['p_yes'].fillna(0))
    # update the cal_high_contamination amount with the cal_medium_contamination amount loss:
    required_cal_high_contamination: float = cal_high_contamination_original + contamin_loss
    # probabilty rescaling factor:
    cal_high_contamination_multiplier: float = required_cal_high_contamination / \
                                                cal_high_contamination_original
    # renormalise 'p_yes' with the scaling factor where CALCountryMedium == False:
    df_new.loc[df_new['CALSourceCountry'] & ~df_new['CALSourceCountryMedium'],'p_yes'] = \
        df_new[df_new['CALSourceCountry'] & ~df_new['CALSourceCountryMedium']]['p_yes'] *\
            cal_high_contamination_multiplier
    # Adjust the 'p_no' column accordingly:
    df_new['p_no'] = 1 - df_new['p_yes']
    # Prepare to write the selected columns into a new CSV file:
    cols = df_new.columns.tolist()
    # Retain the new_cols (i.e., simply shift 'CALSourceCountry' to the column 1), namely:
    # [CALSourceCountry, RuralDest, TypeOfGoods, SCHS, CALSourceCountryMedium, prop, p_no, p_yes]:
    new_cols = [cols[0], cols[-1], *cols[1:7]]
    df_new = df_new[new_cols]

    df_new_cpt = df_new[attribs + ['CALSourceCountryMedium', 'prop', 'p_no', 'p_yes']].copy()

    # When all non-CAL the CAL=True half of the new CPT would have p_no and p_yes as NaN, but
    # formally, p_no + p_yes must be equal 1. Let's set for those rows p_no=1 and p_yes=0:
    df_new_cpt['p_no'] = df_new_cpt['p_no'].fillna(1)
    df_new_cpt = df_new_cpt.fillna(0)
    test_ps = df_new_cpt['p_no'].sum() + df_new_cpt['p_yes'].sum()
    if not isclose(test_ps, 120):
        logger.error(f"p_no + p_yes do not add up as expected: {test_ps}")

    # fix in case there are probability values that are too high or too low
    if df_new_cpt.query('p_no > 1').shape[0] > 0:
        df_new_cpt.loc[df_new_cpt['p_no'] > 1, 'p_no'] = 1
        df_new_cpt.loc[df_new_cpt['p_yes'] < 0, 'p_yes'] = 0

    if df_new_cpt.query('p_no < 0').shape[0] > 0:
        df_new_cpt.loc[df_new_cpt['p_no'] < 0, 'p_no'] = 0
        df_new_cpt.loc[df_new_cpt['p_yes'] > 1, 'p_yes'] = 1
    df_new_cpt.to_csv(conf.ext_dir / f"{conf.cpt_top}_with_med.csv",
                      index=False)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    main(Config(ARG["yaml"]))
