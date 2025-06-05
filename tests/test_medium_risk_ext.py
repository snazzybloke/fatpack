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
Testing the flatpack.src.medium_risk_ext module
"""

from importlib.resources import files
import pandas as pd
from flatpack.src.config import Config
from flatpack.preproc.medium_risk_ext import main, obj_setup
from flatpack import samples
import pytest

@pytest.fixture
def cpt_top_wmed():
    return pd.read_csv(files(samples).joinpath("cpt_top_with_med_KEEP.csv"))

@pytest.fixture
def the_conf():
    return Config(files(samples).joinpath("test.yaml"))


def test_obj_setup():
    cpt_top, attributes = obj_setup(files(samples), "cpt_top")
    assert all(_a in attributes for _a in ('CALSourceCountry', 'SCHS', 'RuralDest', 'TypeOfGoods'))
    assert cpt_top.shape == (60, 7)
    assert pytest.approx(cpt_top['prop'].sum()) == 1
    assert pytest.approx(cpt_top['p_yes'].sum(), 1e-6) == 5.6795
    assert pytest.approx(cpt_top['p_no'].sum(), 1e-6) == 54.3205


def test_main(the_conf, cpt_top_wmed):
    main(the_conf)
    main_result = pd.read_csv(files(samples).joinpath("cpt_top_with_med.csv"))
    pd.testing.assert_frame_equal(cpt_top_wmed, main_result)
    files(samples).joinpath("cpt_top_with_med.csv").unlink()
