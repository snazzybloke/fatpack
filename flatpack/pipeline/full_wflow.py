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
Runs all code and generates all results

This file lays at all the functions required to produce the results in the report.
Note that due to stochasticity in the simulations, the results would not be exactly the same.
This is why we ran 100 simulations to acquire some uncertainty in the outputs.

Running: python run_full_workflow.py -y a_input_yaml_file
or:      python run_full_workflow.py --yaml a_input_yaml_file

"""

import sys
import argparse
from flatpack.src.config import Config
from flatpack.preproc import medium_risk_ext, make_containers
from flatpack.src import simulate_pathways, multi_simulate_pathways
from flatpack.plot import plot_data, plot_multisim_data
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(conf: Config) -> None:
    """
    Starts the whole workflow when called with a config object.

    """
    # First, the conditional probability tables need to be created elsewhere and added in.
    # They should currently be in the data folder. Then:

    # run CAL-medium risk calculation
    medium_risk_ext.main(conf)
    # calculate_cal_medium_risk_rates_internal.main()

    # Run containers file
    make_containers.main(conf)

    # For individual results (i.e. one simulation per alternate scenario), run the following:
    simulate_pathways.main(conf)
    # with data visualisations:
    plot_data.run_individual_outputs(conf)

    # For 100 simulations in one go, plus summaries, run the following:
    multi_simulate_pathways.main(conf)
    # with visualisations
    plot_multisim_data.main(conf)


if __name__ == "__main__":
    agp = argparse.ArgumentParser()
    agp.add_argument("--yaml", "-y", required=True, help="path to the YAML input config file")
    ARG = vars(agp.parse_args())
    main(Config(ARG["yaml"]))
