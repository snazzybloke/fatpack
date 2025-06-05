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
Testing the flatpack.src.fun_decision module
"""

from copy import deepcopy
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from flatpack.src.fun_decision import (apply_tests,
                                          keys_to_tuples,
                                          decide_at_random,
                                          decide_on_attr,
                                          attr_decision_node,
                                          random_decision_node,
                                          multiple_attrs_decision_node,
                                          ccv_decision_node,
                                          by_number_decision_node,
                                          basic_inspect_node,
                                          basic_treatment,
                                          release,
                                          ccv_node,
                                          cbis_node,
                                          cbis_node_mixed)

from flatpack.src.policy import (PolicyNode,
                                    PathwayNode,
                                    DecisionNode,
                                    DecisionNodeNumSplit,
                                    TreatmentNode,
                                    summarise_mixed)
# from flatpack import samples
import pytest


@pytest.fixture
def ext_df(the_ext_df):
    return deepcopy(the_ext_df)


def test_apply_tests(ext_df: DataFrame, yes_node: PolicyNode, no_node: PolicyNode) -> None:
    assert apply_tests(ext_df, {'RuralDest': False, 'TypeOfGoods': "normal"}, yes_node, no_node
                       )[0][0].sum() == 4
    assert apply_tests(ext_df, {'RuralDest': False, 'TypeOfGoods': "normal"}, yes_node, no_node
                       )[1][0].sum() == 7
    assert apply_tests(ext_df, {'RuralDest': True, 'TypeOfGoods': "exceptions"}, yes_node, no_node
                       )[0][0].sum() == 2


def test_keys_to_tuples() -> None:
    assert keys_to_tuples({"Yes": 1, (False,): 2}) == {('Yes',): 1, (False,): 2}
    assert keys_to_tuples({True: "Good", False: "Bad"}) == {(True,): "Good", (False,): "Bad"}


def test_decide_at_random(ext_df: DataFrame, yes_node: PolicyNode, no_node: PolicyNode) -> None:
    assert 25 < sum(list(map(lambda _: decide_at_random(ext_df, 0.333, yes_node, no_node), range(100))
                          )[i][0][0]
                     for i in range(100)).mean() < 40


def test_decide_on_attr(ext_df: DataFrame, yes_node: PolicyNode, no_node: PolicyNode) -> None:
    assert decide_on_attr(ext_df, "CALSourceCountry", {True: yes_node, False: no_node}
                       )[0][0].sum() == 5
    assert decide_on_attr(ext_df, "TypeOfGoods", {"normal": yes_node, "exceptions": no_node},
                          PolicyNode(name="Default"))[2][0].sum() == 4
    assert  decide_on_attr(ext_df, "RuralDest", {True: yes_node, False: no_node}
                       )[0][0].sum() == 2


def test_attr_decision_node(ext_df: DataFrame,
                            dmask: Series,
                            yes_node: PathwayNode,
                            no_node: PathwayNode) -> None:
    dcn = attr_decision_node("SCHS yes/no", 'SCHS', {"no": no_node}, default_node=yes_node)
    assert isinstance(dcn, DecisionNode)
    ext_df['detected'] = pd.Series([False for i in range(11)])
    dcn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 3
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_random_decision_node(ext_df: DataFrame,
                              dmask: Series,
                              yes_node: PathwayNode,
                              no_node: PathwayNode) -> None:
    rnn = random_decision_node("Inspect yes/no", 0.95, yes_node=yes_node, no_node=no_node)
    assert isinstance(rnn, DecisionNode)
    ext_df['detected'] = False
    rnn(ext_df, dmask)
    assert 7 <= ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() <= 9
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_multiple_attrs_decision_node(ext_df: DataFrame,
                                      dmask: Series,
                                      yes_node: PathwayNode,
                                      no_node: PathwayNode) -> None:
    man = multiple_attrs_decision_node("Full and rural yes/no",
                                       {'RuralDest': True, 'TypeOfGoods': "exceptions"},
                                       yes_node=yes_node,
                                       no_node=no_node)
    assert isinstance(man, DecisionNode)
    ext_df['detected'] = False
    man(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_ccv_decision_node(ext_df: DataFrame,
                           dmask: Series,
                           yes_node: PathwayNode,
                           no_node: PathwayNode) -> None:
    ccn = ccv_decision_node("CCV", yes_node=yes_node, no_node=no_node)
    assert isinstance(ccn, DecisionNodeNumSplit)
    ext_df['detected'] = False
    ccn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_by_number_decision_node(ext_df: DataFrame,
                                 dmask: Series,
                                 yes_node: PathwayNode,
                                 no_node: PathwayNode) -> None:
    byn = by_number_decision_node("ByN", con_y=3, yes_node=yes_node, no_node=no_node)
    assert isinstance(byn, DecisionNodeNumSplit)
    ext_df['detected'] = False
    byn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() <= 6
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_basic_inspect_node(ext_df: DataFrame, dmask: Series) -> None:
    pan = basic_inspect_node("Testo", mon_frac=1, sensitivity=1)
    assert isinstance(pan, PathwayNode)
    ext_df['detected'] = False
    pan(ext_df, dmask)
    assert ext_df['detected'].sum() == 2
    assert ext_df['inspected'].sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_basic_treatment(ext_df: DataFrame, dmask: Series, yes_node: PathwayNode) -> None:
    btn = basic_treatment("Testo", effectiveness=1, child_node=yes_node)
    assert isinstance(btn, TreatmentNode)
    ext_df['detected'] = False
    btn(ext_df, dmask)
    assert ext_df['directions'].explode().str.contains("Node: If yes sampled 0%").sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_release() -> None:
    rln = release("Testo")
    assert isinstance(rln, PathwayNode)
    assert rln.original_name == "Testo"


def test_ccv_node(ext_df: DataFrame, dmask: Series) -> None:
    cpn = ccv_node("Testo", sensitivity=1)
    assert isinstance(cpn, PathwayNode)
    assert cpn.original_name == "Testo"
    ext_df['detected'] = False
    cpn(ext_df, dmask)
    assert ext_df['detected'].sum() == 2
    assert ext_df['inspected'].sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_cbis_node(ext_df: DataFrame, dmask: Series) -> None:
    cbn = cbis_node("Testo", sensitivity=1, clearance_num=10, monit_frac=1)
    assert isinstance(cbn, PathwayNode)
    assert cbn.original_name == "Testo"
    ext_df['detected'] = False
    cbn(ext_df, dmask)
    assert ext_df['detected'].sum() == 2
    assert ext_df['inspected'].sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])


def test_cbis_node_mixed(ext_df: DataFrame, dmask: Series) -> None:
    senss = {'AA': (1, 0.7), 'DAFF': (1, 0.3)}
    cmn = cbis_node_mixed("Testo", senss, 10, 0.5, csp3=False)
    assert cmn.original_name == "Testo"
    ext_df['detected'] = False
    cmn(ext_df, dmask)
    assert ext_df['detected'].sum() == 2
    assert ext_df['directions'].explode().str.contains("yes found").sum() == 2
    assert ext_df['inspected'].sum() == 9
    assert ext_df['directions'].explode().isna().sum() == 2
    # also testing summarise_mixed():
    sum_dic = summarise_mixed(cmn)
    assert len(sum_dic) == 3
    assert sum_dic.get("Testo - AA") == \
        {'containers': 6, 'inspections': 6, 'contamination found': 1, 'leakage': 0}
    assert sum_dic.get("Testo - DAFF") == \
        {'containers': 3, 'inspections': 3, 'contamination found': 1, 'leakage': 0}
    assert sum_dic.get("Testo - release") == \
        {'containers': 0, 'inspections': 0, 'contamination found': 0, 'leakage': 0}
    ext_df = ext_df.drop(columns=['detected', 'inspected', 'directions'])
