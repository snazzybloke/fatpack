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
Testing the flatpack.src.policy module
"""

from copy import deepcopy
from pathlib import Path
from functools import partial
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from flatpack.src.policy import (PolicyNode,
                                  RootNode,
                                  DecisionNode,
                                  DecisionNodeNumSplit,
                                  PathwayNode,
                                  TreatmentNode,
                                  create_policy,
                                  simulate_policy)
from flatpack.src.fun_decision import cbis_node_mixed
import pytest


@pytest.fixture
def ext_df(the_ext_df):
    return deepcopy(the_ext_df)


def test_PolicyNode(ext_df: DataFrame, dmask: Series) -> None:
    with pytest.raises(TypeError):
        PolicyNode(name=1234)
    pln = PolicyNode(name="Testo")
    pln.record(ext_df, dmask)
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns='directions')


def test_RootNode(ext_df: DataFrame, yes_node: PathwayNode) -> None:
    with pytest.raises(TypeError):
        RootNode(name="ABC", top_node="TopDecision")
    ron = RootNode(name="Testo", top_node=yes_node)
    ron(ext_df)
    assert ext_df['inspected'].sum() == 0
    assert ext_df['detected'].sum() == 0
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 11
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_DecisionNode(ext_df: DataFrame, yes_node: PathwayNode, no_node: PathwayNode, dmask: Series
                      ) -> None:
    with pytest.raises(TypeError):
        DecisionNode(name="ABC", decision_func="MyFun")
    from flatpack.src.fun_decision import decide_on_attr
    decide = partial(decide_on_attr, attr_name='CALSourceCountry',
                     vals_to_nodes={True: yes_node, False: no_node})
    dcn = DecisionNode(name="Testo", decision_func=decide)
    ext_df['detected'] = False
    ext_df.loc[1, 'detected'] = True
    dcn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 5
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'directions'])


def test_DecisionNodeNumSplit(ext_df: DataFrame,
                              yes_node: PathwayNode,
                              no_node: PathwayNode,
                              dmask: Series) -> None:
    with pytest.raises(TypeError):
        DecisionNodeNumSplit(name="ABC", number=3, yes_node=yes_node, no_node="No")
    with pytest.raises(ValueError):
        DecisionNodeNumSplit(name="ABC", number=-33, yes_node=yes_node, no_node=no_node)
    dns = DecisionNodeNumSplit(name="ABC", number=3, yes_node=yes_node, no_node=no_node)
    ext_df['detected'] = False
    ext_df.loc[1, 'detected'] = True
    dns(ext_df, dmask)
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_PathwayNode(ext_df: DataFrame, dmask: Series) -> None:
    with pytest.raises(TypeError):
        PathwayNode(name="ABC", sampler="Sam")
    from flatpack.src.inspectors import BasicSampler
    pan = PathwayNode(name="Testo", sampler=BasicSampler(sensitivity=1, monit_frac=1))
    ext_df['detected'] = False
    pan(ext_df, dmask)
    assert ext_df['detected'].sum() == 2
    assert ext_df['inspected'].sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_TreatmentNode(ext_df: DataFrame, dmask: Series, yes_node: PathwayNode) -> None:
    with pytest.raises(TypeError):
        TreatmentNode(name="ABC", effectiveness=1, child_node="SonOfSam")
    with pytest.raises(ValueError):
        TreatmentNode(name="ABC", effectiveness=-3.33, child_node=None)
    tmn = TreatmentNode(name="Testo", effectiveness=1, child_node=yes_node)
    ext_df['detected'] = False
    tmn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])



def test_simulate_policy(cont_ext: Path, tmp_path: Path) -> None:
    senss = {'AA': (1, 0.7), 'DAFF': (1, 0.3)}
    mixed_inspect = cbis_node_mixed('Testo', senss, 10, 0.5, csp3=False)
    # first testing create_policy()
    name, root, node_dict = create_policy('test_pol', mixed_inspect, [mixed_inspect])
    assert name == "test_pol"
    assert root.top_node.name == "Testo"
    assert root.top_node  == mixed_inspect
    assert node_dict == {"Testo": mixed_inspect}
    # now testing simulate_policy()
    pathway, treatment = simulate_policy('Test mixed',
                                         root,
                                         node_dict,
                                         save_results=True,
                                         save_container_results=True,
                                         save_summary=True,
                                         cs_fname=cont_ext,
                                         output_dir=tmp_path)
    assert treatment.empty
    assert pathway.shape == (3, 4)
    for _c in ('containers', 'inspections'):
        assert pathway[_c].sum() == 11
    assert pathway['contamination found'].sum() == 2
    assert (pathway['leakage'] == 0).all()
