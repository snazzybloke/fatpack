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
This module contains source implementing the continous sampling protocols
(CSP-1, CSP-3) in the container management pathways.
The key function is find_mf() which calculates the risk (i.e., contamination/approach rate)
and monitoring fraction at the maximum leakage. The functions above, i.e., csp_lakage() and
EITHER leakage_csp3() and csp3_solutio() (for CSP3 protocol) OR leakage_csp1 and csp1_partials()
called to solve the required pair of equations.
The functions below find_mf(), namely make_tables(), leakage_estimate(), plot_curves(),
and inspection_profile() serve simply as the demo of this module.
"""

import sys
from sympy import init_printing, Matrix, symbols, diff, plot, plotting, lambdify
from sympy.core.symbol import Symbol
from sympy.core.mul import Mul
import numpy as np
import pandas as pd
from scipy.optimize import root
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

init_printing(use_unicode=True)


def csp1_partials(risk: Symbol | float, mfrac: Symbol | float, c_num: int
                  ) -> list[Mul | float]:
    """
    Evaluates the sequence of partial contributions (probabilities?) of the
    CN stages of the census mode.

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk (e.g., approach/contamination rate) as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP1 sampling protocol.

    Returns
    -------
    ... : List[Union(Mul, float)]
        The partial probabilities of inspection at all stages of the CSP1 census mode

    """
    ps_census = [mfrac * risk * (1 - risk)**(_i - c_num) for _i in range(c_num)]
    vec = [1] + ps_census
    return vec


def csp1_leakage(risk: Symbol | float, mfrac: Symbol | float, c_num: int) -> Mul | float:
    """
    The leakage rate (probability) as the product of approach rate times the negative inspection
    (non-monitoring) probability, and the fraction when the sampling is in the monitoring mode
    (via the division by the sum of all the probabilities of the census mode inspection).

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk (e.g., approach/contamination rate) as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP1 sampling protocol.

    Returns
    -------
    ... : Union[Mul, float]
        The leakage rate either as a function of the 3 parameters or a float.

    """
    return risk * (1 - mfrac) / sum(csp1_partials(risk, mfrac, c_num))



def csp3_partials(risk: Symbol | float,
                  mfrac: Symbol | float,
                  c_num: int,
                  tc_num: int) -> list[float | Mul]:
    """
    Evaluates partial contributions of the 2 * CN stages of separate modes
    (monitoring, census, tight census and failure detection) to the inspection probability.

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.

    Returns
    -------
    ... : List[Union[Mul, float]]
        The partial probabilities of inspection at each stage of the CSP3 modes.

    """
    fst_census = mfrac * risk * (1 - (1 - risk)**tc_num *
                                 (1 - mfrac * risk)**(c_num - tc_num)) / (1 - risk)**c_num
    # census mode stages until CN passed inspection:
    census = [fst_census * (1 - risk)**_i for _i in range(c_num)]
    # tight census mode stages until TC passed inspection:
    tight_census = [mfrac * risk * (1 - risk)**_i for _i in range(tc_num)]
    # failure detection mode until CN (total or extra after TC?) passed inspection
    failure_detection = [mfrac * risk * (1 - risk)**tc_num * (1 - mfrac * risk)**_i
                         for _i in range(c_num - tc_num)]
    # combine the 3 lists into a single and return it:
    return [1] + census + tight_census + failure_detection


def csp3_leakage(risk: Symbol | float,
                mfrac: Symbol | float,
                c_num: int,
                tc_num: int) -> Mul | float:
    """
    Evaluates the leakage rate as the product of risk (approach rate), non-monitoring rate,
    and the fraction of time the sampling is in the monitoring mode.

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.

    Returns
    -------
    ... : Union[Mul, float]
        The leakage rate either as a function of the 4 parameters or a float

    """
    # For CN=10 and TC=4 this produces monitoring mode stages [0, 15, 16, ..., 19]:
    monitor_stages = [0] + list(range(c_num + tc_num + 1, 2 * c_num + 1))
    # the census, tight census, failure detection mode probabilities:
    probs = csp3_partials(risk, mfrac, c_num, tc_num)
    # the fraction of time the sampling is in the monitoring mode:
    momo_frac = sum(probs[_i] for _i in monitor_stages) / sum(probs)
    return risk * (1 - mfrac) * momo_frac


def csp_leakage(risk: Symbol | float,
                mfrac: Symbol | float,
                c_num: int,
                tc_num: int,
                csp3=True) -> Mul | float:
    """
    Evaluates the estimated leakage rate for the chosen sampling protocol
    (either CSP3 or CSP1)

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.
    csp3 : bool, optional
        CSP3 or CSP1 sampling. The default is True (CSP3).

    Returns
    -------
    ... : Union[Mul, float]
        The leakage rate either as a function of the 3/4 parameters or a float..

    """
    if csp3:
        leak = csp3_leakage(risk, mfrac, c_num, tc_num)
    else:
        leak = csp1_leakage(risk, mfrac, c_num)
    return leak


def find_mf(max_leakage: float, c_num: int, tc_num: int, csp3=True, print_msg=True) -> list[float]:
    """
    Calculates the leakage as an analytic experession (i.e., a function of rt and ft Symbols) and
    its derivative (with respect to rt, i.e., the risk). Then it numerically finds the solution of
    the pair of two equations: (1) leak - max_leakage = 0, and (2) d_leak/d_rt = 0 (i.e., MAX leak).
    Returns the solution as af 2-float list of the rt (risk) and ft (monit_frac) values.

    Parameters
    ----------
    max_leakage : float
        An estimated maximum leakage (typically as mean contamination multiplied by a factor of 2).
    c_num : int
        Clearance number (CN) for the CSP3 and CSP1 sampling protocols (e.g, 10).
    tc_num : int
        Tight census (TC) number for the CSP3 sampling protocol (typically 4).
    csp3 : bool, optional
        CSP3 (default) or CSP1?
    print_msg : bool, optional
        Whether to print (default) the numerical solution or not.

    Returns
    -------
    ... : List[float]
        The rt (risk) and ft (monit_frac) values resulting in the Max leak

    """
    r_temp, f_temp = symbols("rt, ft", positive=True)
    leak = csp_leakage(r_temp, f_temp, c_num, tc_num, csp3)
    # evaluate the derivative (d leak / d r_temp):
    dv_l = diff(leak, r_temp)
    # Combine this and (leak - max_leakage) into a 2-element vector function, to ensure that the
    # MAXIMUM leak is at the derivative=0 AND leak = max_leakage, i.e., when both the elements = 0:
    vec_func = Matrix([leak - max_leakage, dv_l])
    # convert the symbolic expression into a function (for the sake of efficiency):
    vec_lam = lambdify([(r_temp, f_temp)], vec_func)
    # Find the zeros (i.e., root of the Eq. vec_lam=0), with the initial guess [0.1, 0.1]:
    ans = root(lambda x: np.ravel(vec_lam(x)), x0=np.array([0.1, 0.1]).T)
    if print_msg:
        logger.info(f'\nSolution returned:\n{ans}')

    return list(ans.x)


def inspection_profile(risk: float, mfrac: float, c_num: int, tc_num: int, csp3: bool=True
                        ) -> tuple[float, ...]:
    """
    Evaluates the proportions of censored, monitored and uninspected containers.
    These add up to 1.

    Parameters
    ----------
    risk : Union[Symbol, float]
        Risk as a float value or symbolic expression.
    mfrac : Union[Symbol, float]
        Monitoring fraction as a float of symbolic expression.
    c_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.
    csp3 : bool, optional
        CSP3 (default) or CSP1 sampling.

    Returns
    -------
    ... : Tuple[float]]
        The proportions of censused, monitored and uninspected containers.

    """
    uninspected = csp_leakage(risk, mfrac, c_num, tc_num, csp3=csp3) / risk
    monitored = mfrac / (1 - mfrac) * uninspected
    # since they have to add up to 1, the censused is simply 1 minus those two:
    censused = 1 - (uninspected + monitored)
    return censused, monitored, uninspected


def plot_curves(max_leakage: float, cnums: list[int], tc_num: int=4, csp3: bool=True) -> None:
    """
    It makes two plot: (1) leakage vs approach rate, and (2) proportion inspected vs approach rate
    for the provided sequence of clearance numbers.

    Parameters
    ----------
    max_leakage : float
        An estimate of the max leakage (e.g., based on the approach/comtamination rate).
    cnums : List[int]
        Sequence of Clearance numbers (CN) for the CSP3 and CSP1 sampling protocols.
    tc : int, optional
        Tight census (TC) number for the CSP3 sampling protocol (typically 4).
    csp3 : bool, optional
        Carry on with either the CSP3 (if True) or CSP1. The default is True.

    Returns
    -------
    None

    """
    leakage_lines = []
    inspection_lines = []
    r_temp = symbols("rt", positive=True)

    protocol = 'csp3' if csp3 else 'csp1'

    logger.info(f'\nComputing {protocol} parameters, max leakage = {max_leakage*100:.1f}%')

    for cnum in cnums:
        _, f_val = find_mf(max_leakage, cnum, tc_num, csp3)
        logger.info(f'\n\t With cnum={cnum:2d}, set mf={f_val*100:.1f}%')
        leakage_lines.append(csp_leakage(r_temp, f_val, cnum, tc_num, csp3))
        inspection_lines.append(1 - csp_leakage(r_temp, f_val, cnum, tc_num, csp3) / r_temp)

    tit = f'{protocol}, leakage limit={max_leakage*100}%'

    p_1 = plot(*leakage_lines, (r_temp, 1.0e-5, 1.0 - 1.0e-5), title=tit,
             xlabel='approach rate', ylabel='leakage', legend=True,
             show=False, adaptive=False, nb_of_points=1000)
    p_2 = plot(*inspection_lines, (r_temp, 1.0e-5, 1.0-1e-5),
             xlabel='approach rate', ylabel='proportion inspected',
             legend=False, show=False, ylim=(0,1), axis_center=(0,0))

    for _i, cnum in enumerate(cnums):
        p_1[_i].label = f"cn = {cnum}"

    _ = plotting.PlotGrid(2, 1, p_1, p_2)


def leakage_estimate(max_leakage: float,
                     c_num: int,
                     tc_num: int,
                     approach_estimate: float,
                     csp3: bool=True
                     ) -> tuple[float, ...]:
    """
    Calls find_mf() to get the risk (approach rate) and monitoring fraction values at the maximum
    leakage and then passes those to csp_leakage() to get the max leakage value.

    Parameters
    ----------
    max_leakage : float
        An estimated maximum leakage (possibly derived from approach rate. Or t_risk.)
    c_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.
    approach_estimate : float
        An estimated contamination/approach rate.
    csp3 : bool, optional
        CSP3 (default) or CSP1 sampling.

    Returns
    -------
    ... : Tuple[float]
        The risk (approach rate), monitoring fraction and max leakage values.

    """
    r_val, f_val = find_mf(max_leakage, c_num, tc_num, csp3)
    # risk, monitoring fraction and expected leakage (I believe):
    return r_val, f_val, csp_leakage(approach_estimate, f_val, c_num, tc_num, csp3)


def make_tables(cnums: list[int],
                max_leakage: float,
                tc_num: int,
                expected_approach: float,
                csp3: bool=True) -> None:
    """
    Prints the calculated CSP parameter values and two tables in LaTeX format.

    Parameters
    ----------

    cnums : List[int]
        clearance number (CN) sequence for the CSP1 or CSP3 sampling protocol.
    max_leakage : float
        An estimate of the maximum leakage (e.g., based on the approach rate or other).
    tc_num : int
        tight census (TC) number for the CSP3 sampling protocol.
    expected_approach : float
        The expected approach rate
    csp3 : bool, optional
        CSP3 (default) or CSP1 sampling.

    Returns
    -------
    None

    """
    tab1 = {}
    tab2 = {}

    for cnum in cnums:
        r_val, f_val, l_point = leakage_estimate(
            max_leakage, cnum, tc_num, expected_approach, csp3)
        msg = f"cnum = {cnum:2d} \tmf = {f_val*100:.0f}% \tleakage at r = {expected_approach*100}%:"
        msg += f" {l_point*100:.2f}% \tpeak at r = {r_val*100:4.1f}%"
        logger.info(f"\n{msg}")
        tab1.update({ cnum: {"mf (%)": round(f_val * 100, 0),
                             "leakage (%)": round(l_point * 100, 2)}
                     })

        censused, monitored, uninspected = inspection_profile(
            expected_approach, f_val, cnum, tc_num, csp3)
        msg = f"\n\tcensus {censused*100:3.1f}% \tmonitor {monitored*100:3.1f}% "
        msg += f"\tnot inspected {uninspected*100:3.1f}%"
        logger.info(msg)
        tab2.update({ cnum: {'inspected - census (%)': round(censused * 100, 1),
                             'inspected - monitor (%)': round(monitored * 100, 1),
                             'not inspected (%)': round(uninspected * 100, 1)}
                     })
    t1_df = pd.DataFrame(tab1).T
    logger.info(f"tab1:\n{t1_df.style.to_latex()}")
    t2_df = pd.DataFrame(tab2).T
    logger.info(f"tab2:\n{t2_df.style.to_latex()}")
    t2_df['leakage (%)'] = t1_df['leakage (%)']
    logger.info(f"tab2\n{t2_df.style.to_latex()}")


if __name__ == '__main__':
    # an estimate of max leakage:
    MAX_LEAKAGE = 0.06
    # an estimate of approach rate (i.e., "risk")
    R_EXPECT = 0.3
    # try several choices for the clerance number:
    CNS = [5, 10, 15]
    # tight census number:
    TCN = 4

    """
    Calling the functions below runs a demo for the 4 parameters above:
    - make_tables() for each census_number calls leakage_estimate()
    - leakage_estimate() first calls find_mf()
    - find_mf() first calls simpy.symbols() to return Symbol objs r_temp and f_temp.
    These two are the risk and monitoring fraction. Then it calls csp_leakage(),
    which will call either leakage_csp3() or leakage_csp1() depending on CSP choice.
    - leakage_csp3() calls csp3_partials() to evaluate a list whose elements depend
    on the rt and ft (i.e., Symbols for risk and montoring fraction). This sum of
    the list elements defines a normalisation (the sum of all partial probabilities),
    while the list elements are selected based on a short list of integers which index
    the elements of the monitoring mode. The partail sum of the probabilities indexed
    by these integers is then divided by the total sum of the probabilities. This fraction
    (i.e., monitoring mode proportion) times rt times (1 - ft) is returned from leakage_csp3()
    to csp_leakage() as the leak (a symbolic expression).
    - back in find_md() the derviative of the leak with respect to the rt Symbol
    (i.e., the risk), make a 2-element vector function with the elements
    f = (leak - max_leakage, d_leak / d_rt) and find the solution/root f=0 with respect to
    risk and monit_frac. Then it returns the root (risk and monit_frac) from find_mf()
    to point_estimate_leakege(). This are numeric (float), not Symbols.
    - leakage_estimate() then returns the calculated risk, monitoring fraction
    values and also the associated leakage by calliing the csp_leakage()
    (this time with values for risk and monit_frac, not Symbols)
    - finally, using these values inside make_tables(), inspection_profile() is
    called to return censused, monitored, uninspected values by calling
    csp_leakage() (csp3 or csp1).
    -make_tables prints the rounded root values, the first (risk) as the peak risk
    and the second as the

    - plot_curves() creates only the risk Symbol and calls find_mf() to
    return the monit_frac (f_val), then calls csp_leakage(),
    for the upper and lowe panel plots
    """
    make_tables(CNS, MAX_LEAKAGE, TCN, R_EXPECT, csp3=True)
    print('\n')
    plot_curves(MAX_LEAKAGE, CNS, TCN, csp3=True)
    print('\n')
    plot_curves(MAX_LEAKAGE, CNS, TCN, csp3=False)
