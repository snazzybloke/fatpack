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
Config for pytest
"""

from pathlib import Path
from importlib.resources import files
import pandas as pd
from flatpack import samples
from flatpack.src.fun_decision import release
import pytest


@pytest.fixture(scope="session")
def cont_ext():
    return files(samples).joinpath("original_ext_int_containers_external_with_medium.csv")


@pytest.fixture(scope="session")
def the_ext_df(cont_ext: Path):
    return pd.read_csv(cont_ext, index_col=0)


@pytest.fixture(scope="session")
def dmask():
    return pd.Series([False, True,  True, True, False, True, True, True, True, True, True])


@pytest.fixture(scope="session")
def yes_node():
    return release(name="If yes")


@pytest.fixture(scope="session")
def no_node():
    return release(name="If no")
