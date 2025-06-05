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
Testing the flatpack.src.inspectors module
"""

from copy import deepcopy
from pandas.core.frame import DataFrame
from flatpack.src.inspectors import (rn_random,
                                      get_column,
                                      load_ccv_lookup_table,
                                      set_prior_n_ccv,
                                      get_num_ccv_samples,
                                      SamplingRegime,
                                      BasicSampler,
                                      RandomSampler,
                                      CSP1Sampler,
                                      CSP3Sampler)
import pytest


@pytest.fixture
def ext_df(the_ext_df):
    return deepcopy(the_ext_df)


def test_rn_random() -> None:
    assert rn_random(12).shape[0] == 12
    assert rn_random((3, 5)).shape ==(3, 5)
    assert (rn_random(33) < 1).all()
    assert (0 <= rn_random(33)).all()


def test_get_column(ext_df: DataFrame()) -> None:
    assert get_column(ext_df, 'CALSourceCountry').sum() == 5
    assert get_column(ext_df, 'XYZ').isna().all()


def test_load_ccv_lookup_table() -> None:
    assert load_ccv_lookup_table().shape == (36050, 2)


def test_set_prior_n_ccv() -> None:
    assert set_prior_n_ccv(0) == 500
    assert set_prior_n_ccv(501)  == 1000
    assert set_prior_n_ccv(1001)  == 1500
    with pytest.raises(StopIteration):
        set_prior_n_ccv(10001)  == 1000


def test_get_num_ccv_samples() -> None:
    with pytest.raises(ValueError):
        get_num_ccv_samples(500, 10, 3)
    assert pytest.approx(get_num_ccv_samples(500, 10, 4), 1e-6) == 448.3183377
    assert pytest.approx(get_num_ccv_samples(10000, 460, 5), 1e-5) == 23985.45242


def test_SamplingRegime(ext_df: DataFrame()) -> None:
    with pytest.raises(TypeError):
        samreg = SamplingRegime(sensitivity="abc")
    with pytest.raises(ValueError):
        samreg = SamplingRegime(sensitivity=-3)
    samreg = SamplingRegime(sensitivity=0.9)
    assert samreg.sensitivity > 0.899
    get_column(ext_df, 'detected', False)
    # assert samreg.inspect_container(ext_df.iloc[1].to_dict()) == True
    # Test if detecting the contaminated row 8-10 times out of 10 given the sensitivity=0.9:
    assert 7 <= sum(list(map(lambda _: samreg.inspect_container(ext_df.iloc[1].to_dict()),
                             range(10)))) <= 10


def test_Basicsampler(ext_df: DataFrame(), dmask: list) -> None:
    with pytest.raises(TypeError):
        bsam = BasicSampler(sensitivity=0.9, monit_frac="OK")
    with pytest.raises(ValueError):
        bsam = BasicSampler(sensitivity=0.9, monit_frac=-2)
    bsam = BasicSampler(sensitivity=0.9, monit_frac=0.5)
    bsam.inspect_batch(ext_df, dmask)
    assert 0 <= ext_df['inspected'].sum() <= 8
    assert 0 <= ext_df['detected'].sum() <= 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_RandomSampler(ext_df: DataFrame(), dmask: list) -> None:
    with pytest.raises(TypeError):
        rns = RandomSampler(sensitivity=0.9, prior_n=4505, prior_y=0, t_risk="T")
    with pytest.raises(ValueError):
        rns = RandomSampler(sensitivity=0.9, prior_n=4505, prior_y=0, t_risk=-2.)
    rns = RandomSampler(sensitivity=0.9, prior_n=4505, prior_y=0, t_risk=0.5)
    rns.inspect_batch(ext_df, dmask)
    assert ext_df['inspected'].sum() == 9
    assert 0 <= ext_df['detected'].sum() <= 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_CSP1Sampler(ext_df: DataFrame(), dmask: list) -> None:
    with pytest.raises(TypeError):
        csp1 = CSP1Sampler(sensitivity=0.9, clearance_num="CN", monit_frac=0.5)
    with pytest.raises(ValueError):
        csp1 = CSP1Sampler(sensitivity=0.9, clearance_num=-3, monit_frac=0.5)
    csp1 = CSP1Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5)
    csp1.inspect_batch(ext_df, dmask)
    assert ext_df['inspected'].sum() == 9
    assert 1 <= ext_df['detected'].sum() <= 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_CSP3Sampler(ext_df: DataFrame(), dmask: list) -> None:
    with pytest.raises(TypeError):
        csp3 = CSP3Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5, tight_cens_num="TC")
    with pytest.raises(ValueError):
        csp3 = CSP3Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5, tight_cens_num=-8)
    csp3 = CSP3Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5, tight_cens_num=1)
    csp3.inspect_batch(ext_df, dmask)
    assert ext_df['inspected'].sum() == 9
    assert 1 <= ext_df['detected'].sum() <= 2
