###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ante Bilic                          #
# Email: ante.bilic.mr@gmail.co                   #
# Status: N/A                                     #
###################################################

"""
Testing the flatpack.src.medium_risk_ext module
"""

from importlib.resources import files
import pandas as pd
from flatpack.preproc.make_containers import TheConGen
from flatpack import samples
import pytest

@pytest.fixture
def cpt_top_wmed():
    return files(samples).joinpath("cpt_top_with_med_KEEP.csv")


def test_TheConGen(tmp_path, cpt_top_wmed):
    with pytest.raises(TypeError):
        TheConGen(True, 100, 9)
    tcg = TheConGen(cpt_top_file=cpt_top_wmed,
                    num_containers=1_000,
                    num_importers=9)
    vc1 = tcg.generate_containers()
    assert vc1.shape == (1_000, 5)
    tcg.importer_contamination(vc1)
    assert vc1.shape == (1_000, 7)
    tcg.generate_sample_containers(tmp_path / "test_ext_cont_1k.csv")
    cs_df = pd.read_csv(tmp_path / "test_ext_cont_1k.csv", index_col=0)
    assert 0 <= cs_df.query('CALSourceCountry').query('contamination.eq("yes")').shape[0] <=5
    assert 60 < cs_df.query('~CALSourceCountry').query('contamination.eq("yes")').shape[0] < 100
    (tmp_path / "test_ext_cont_1k.csv").unlink()
