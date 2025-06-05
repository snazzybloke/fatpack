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
Testing the flatpack.src.ccv_csp_samplings module
"""

from importlib.resources import files
from pathlib import Path
from flatpack.src.ccv_cbis_samplings import (get_approach_rate_atts,
                                                get_approach_rate_atts_internal_low_risk,
                                                ccv_inspect_from_approach_rate,
                                                get_sampling_parameters_external_future,
                                                get_sampling_parameters_internal_future,
                                                get_sampling_parameters_goods_future
                                                )
from flatpack import samples
import pytest


@pytest.fixture
def cont_int():
    return files(samples).joinpath("int_no_linehazard_containers_internal_with_medium.csv")


@pytest.fixture
def cont_goods():
    return files(samples).joinpath("goods_tariff_only_goods.csv")


@pytest.fixture
def csp3_ext():
    return {'SCHS': ["no", False], 'CALSourceCountry': [True]}


@pytest.fixture
def ccv_ext():
    return {'RuralDest': [False], 'CALSourceCountry': [False]}


@pytest.fixture
def csp3_int():
    return {'SCHS': ["no", False], 'CALSourceCountry': [True]}


@pytest.fixture
def csp3_goods():
    return {'Greenlane': ["No"], 'Risk': ["High"], 'Offshore_measure': ["No"]}


def test_get_approach_rate_atts(cont_ext: Path,
                                csp3_ext: dict,
                                ccv_ext: dict,
                                cont_int: Path,
                                csp3_int: dict,
                                cont_goods: Path,
                                csp3_goods: dict) -> None:
    assert pytest.approx(get_approach_rate_atts(cont_ext, csp3_ext), 1e-3) == 0.333
    assert get_approach_rate_atts(cont_ext, ccv_ext) == 0
    assert get_approach_rate_atts(cont_int, csp3_int) == 0.5
    assert get_approach_rate_atts(cont_goods, csp3_goods) == 0.5


def test_get_approach_rate_atts_internal_low_risk(cont_ext: Path, cont_int: dict) -> None:
    with pytest.raises(KeyError):
        get_approach_rate_atts_internal_low_risk(cont_ext)
    assert get_approach_rate_atts_internal_low_risk(cont_int) == 0


def test_ccv_inspect_from_approach_rate() -> None:
    assert ccv_inspect_from_approach_rate(0, 1, 5_000) == (552, 0.5, 0, 5000)
    assert ccv_inspect_from_approach_rate(0.01, 2, 8_000) == (626, 2, 80, 8000)
    with pytest.raises(ValueError):
        ccv_inspect_from_approach_rate(0.01, 1, 10000)
    with pytest.raises(StopIteration):
        ccv_inspect_from_approach_rate(0.01, 2, 11000)


def test_get_sampling_parameters_external_future(cont_ext: Path) -> None:
    assert pytest.approx(get_sampling_parameters_external_future(cont_ext),
                         1e-12) == (2.993055824991491e-07, 711, 0.5, 0, 10000)


def test_get_sampling_parameters_internal_future(cont_int: Path) -> None:
    assert pytest.approx(get_sampling_parameters_internal_future(cont_int),
                         1e-6) == (0.00032540557854938393, 711, 0.5, 0, 10000)


def test_get_sampling_parameters_goods_future(cont_goods: Path) -> None:
    assert pytest.approx(get_sampling_parameters_goods_future(cont_goods),
                         1e-6) == (0.00032540557854938393, 711, 711, 711, 0.5, 0, 10000)
