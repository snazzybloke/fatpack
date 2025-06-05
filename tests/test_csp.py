###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ant Bilic                           #
# Email: ant.bilic@aff.gov.au                     #
# Status: N/A                                     #
###################################################

"""
Testing the flatpack.src.csp module
"""

from flatpack.src.csp import (csp1_partials,
                                 csp1_leakage,
                                 csp3_partials,
                                 csp3_leakage,
                                 csp_leakage,
                                 find_mf,
                                 inspection_profile,
                                 leakage_estimate
                                 )
from sympy.core.symbol import Symbol
import pytest


@pytest.fixture
def r():
    return Symbol("r")


@pytest.fixture
def mf():
    return Symbol("mf")


def test_csp1_partials(r: Symbol | float, mf: Symbol | float) -> None:
    with pytest.raises(ZeroDivisionError):
        csp1_partials(1, mf, 10)
    assert csp1_partials(r, mf, 8)[1] == r * mf / (1 - r)**8
    assert csp1_partials(r, mf, 10)[10] == r * mf / (1 - r)
    assert max(csp1_partials(0.05, 1, 10)) == 1
    assert min(csp1_partials(0.05, 1, 8)) > 0.05


def test_csp1_leakage(r: Symbol | float, mf: Symbol | float) -> None:
    assert csp1_leakage(r, mf, 0) == r * (1 - mf)
    assert pytest.approx(csp1_leakage(0.05, 0, 10), 1e-9) == 0.05


def test_csp3_partials(r: Symbol | float, mf: Symbol | float) -> None:
    with pytest.raises(ZeroDivisionError):
        csp3_partials(1, 0.333, 10, 4)
    assert len(csp3_partials(r, mf, 5, 4)) == 11
    assert len(csp3_partials(r, mf, 10, 4)) == 21
    assert csp3_partials(r, mf, 5, 4)[6] == r * mf
    assert csp3_partials(r, mf, 10, 4)[11] == r * mf
    assert min(csp3_partials(0, mf, 10, 4)) == 0
    assert max(csp3_partials(0.05, 1, 10, 4)) == 1
    assert pytest.approx(min(csp3_partials(0.05, 1, 10, 4)), 1e-6) == 0.0211191


def test_csp3_leakage(r: Symbol | float, mf: Symbol | float) -> None:
    assert csp3_leakage(r, mf, 0, 0) - r * (1 - mf) == 0
    assert pytest.approx(csp3_leakage(0.05, 0, 10, 4), 1e-9) == 0.05


def test_csp_leakage(r: Symbol | float, mf: Symbol | float) -> None:
    assert csp_leakage(r, mf, 0, 0, True) - csp_leakage(r, mf, 0, 0, False) == 0

def test_find_mf() -> None:
    assert pytest.approx(find_mf(0.06, 5, 4, True, False), 1e-3) == [2.159e-01, 4.553e-01]
    assert pytest.approx(find_mf(0.06, 10, 4, True, False), 1e-3) == [1.446e-01, 2.941e-01]
    assert pytest.approx(find_mf(0.06, 15, 4, True, False), 1e-3) == [1.173e-01, 1.964e-01]
    assert pytest.approx(find_mf(0.06, 5, 4, False, False), 1e-3) == [2.167e-01, 4.351e-01]
    assert pytest.approx(find_mf(0.06, 10, 4, False, False), 1e-3) == [1.455e-01, 2.283e-01]
    assert pytest.approx(find_mf(0.06, 15, 4, False, False), 1e-3) == [1.188e-01, 1.282e-01]


def test_inspection_profile() -> None:
    assert pytest.approx(inspection_profile(0.06, 0.3, 10, 4), 1e-4) == (0.11675, 0.26498, 0.61828)
    assert pytest.approx(inspection_profile(0.06, 0.3, 10, 4, False),
                         1e-4) == (0.20444, 0.23867, 0.55689)


def test_estimate_leakage() -> None:
    assert pytest.approx(leakage_estimate(0.06, 10, 4, 0.3, True),
                         1e-4) == (0.14459, 0.29406, 0.023234)
    assert pytest.approx(leakage_estimate(0.06, 10, 4, 0.3, False),
                         1e-4) == (0.14545, 0.22825, 0.026155)
