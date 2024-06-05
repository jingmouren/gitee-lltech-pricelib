#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.autocallable import (autocall_demo, fcn_demo, dcn_demo, phoenix_demo, standard_snowball_demo,
                                   earlyprofit_snowball_demo, butterfly_snowball_demo, otm_snowball_demo,
                                   floored_snowball_demo, stepdown_snowball_demo, bothdown_snowball_demo,
                                   parachute_snowball_demo, snowball_plus_demo, paris_snowball_demo)


def test_autocall_demo():
    expected_data = [['AutoCall Note(二元小雪球)', 99.93417566971927, 99.9772462836672, 99.96258810435373],
                     ['AutoPut Note(二元小雪球)', 100.34198687410401, np.nan, 100.36871158225928]]
    expected_df = pd.DataFrame(expected_data, columns=['期权类型', 'MonteCarlo', 'PDE', 'Quadrature'])
    res_df = autocall_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_fcn_demo():
    expected_data = {'结构': 'FCN(Fixed Coupon Note固定派息票据)', 'MonteCarlo': 100.08715717857146,
                     'PDE': 100.09090516454178, 'Quadrature': 100.09185163785885}
    expected_greeks = {'price_pde': 100.09090516454178, 'delta': 0.07668872814773664, 'gamma': -0.011678641376448695,
                       'vega': -0.002439338027768514, 'theta': 0.0001501986084772966, 'rho': -0.0070618801461777994}
    res_df, greeks = fcn_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_dcn_demo():
    expected_data = {'结构': 'DCN(Digital Coupon Note 二元派息票据)', 'MonteCarlo': 100.10975567486145,
                     'PDE': 100.12485463235888, 'Quadrature': 100.12254728843223}
    expected_greeks = {'price_pde': 100.12485463235888, 'delta': 0.12231670400754524, 'gamma': -0.019447885068544224,
                       'vega': -0.0033750251234131667, 'theta': 0.00019488342688944726, 'rho': -0.006366459347127602}
    res_df, greeks = dcn_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_phoenix_demo():
    expected_data = {'结构': 'Phoenix 凤凰票据', 'MonteCarlo': 99.9742330017115, 'PDE': 99.66294528263722}
    expected_greeks = {'price_pde': 99.66294528263722, 'delta': 0.33140941819185343, 'gamma': -0.04166032121553087,
                       'vega': -0.006730509916361882, 'theta': 0.00034238890381871556, 'rho': -0.002892764013483742}
    res_df, greeks = phoenix_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_standard_snowball_demo():
    expected_data = {'结构': '经典雪球(平敲雪球)', 'MonteCarlo': 99.92088092163144, 'PDE': 99.97242502284875,
                     'Quadrature': 100.01021095357333, 'pde_pct': '0.052%', 'quad_pct': '0.089%'}
    expected_greeks = {'price_pde': 99.97242502284875, 'delta': 0.4336714279992435, 'gamma': -0.06160759915984215,
                       'vega': -0.007502448878742313, 'theta': 0.0004425609512557571, 'rho': -0.0019984183877359385}
    res_df, greeks = standard_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_earlyprofit_snowball_demo():
    expected_data = {'结构': '早利雪球', 'MonteCarlo': 99.93429261435013, 'PDE': 100.01918085274494,
                     'Quadrature': 100.04938425723334}
    expected_greeks = {'price_pde': 100.01918085274494, 'delta': 0.7485946436239885, 'gamma': -0.06779574769430496,
                       'vega': -0.004714328707591591, 'theta': 0.00042737904842510944, 'rho': 0.00035195751124447836}
    res_df, greeks = earlyprofit_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_butterfly_snowball_demo():
    expected_data = {'结构': '蝶式雪球', 'MonteCarlo': 100.0799579903268, 'PDE': 100.13361422282733,
                     'Quadrature': 100.11961017487525}
    expected_greeks = {'price_pde': 100.13361422282733, 'delta': 0.6024552484713723, 'gamma': -0.0582534100587111,
                       'vega': -0.006733108352314332, 'theta': 0.0003771607953234479, 'rho': -0.0008582065182750398}
    res_df, greeks = butterfly_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_otm_snowball_demo():
    expected_data = {'结构': 'OTM雪球', 'MonteCarlo': 100.00358496495079, 'PDE': 100.01671991322786,
                     'Quadrature': 100.0695778009394}
    expected_greeks = {'price_pde': 100.01671991322786, 'delta': 0.13769578938424587, 'gamma': -0.0216993090541564,
                       'vega': -0.0038326463263842923, 'theta': 0.00018841550361813118, 'rho': -0.004585777782421445}
    res_df, greeks = otm_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_floored_snowball_demo():
    expected_data = {'结构': '保底雪球(限损雪球，不追保雪球)', 'MonteCarlo': 100.04932829418274,
                     'PDE': 100.05319061780084, 'Quadrature': 100.08646854508989}
    expected_greeks = {'price_pde': 100.05319061780084, 'delta': 0.30476874786146624, 'gamma': -0.041764074317384825,
                       'vega': -0.004927915110993553, 'theta': 0.00028830963019615296, 'rho': -0.002606915995998662}
    res_df, greeks = floored_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_stepdown_snowball_demo():
    expected_data = {'结构': '降敲雪球', 'MonteCarlo': 99.99000080300591, 'PDE': 99.99882667508119,
                     'Quadrature': 100.04927179943412}
    expected_greeks = {'price_pde': 99.99882667508119, 'delta': 0.42893445638513583, 'gamma': -0.05185686827974223,
                       'vega': -0.006690423157686496, 'theta': 0.0003908593488756651, 'rho': -0.0007847959340392663}
    res_df, greeks = stepdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_bothdown_snowball_demo():
    expected_data = {'结构': '双降雪球', 'MonteCarlo': 99.99734493746297, 'PDE': 100.03946242145815,
                     'Quadrature': 100.02280615164219}
    expected_greeks = {'price_pde': 100.03946242145815, 'delta': 0.6031430326171687, 'gamma': -0.05735408333077885,
                       'vega': -0.003797294860883085, 'theta': 0.00043434106369375056, 'rho': 0.00041062551228748134}
    res_df, greeks = bothdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_parachute_snowball_demo():
    expected_data = {'结构': '降落伞雪球', 'MonteCarlo': 99.27530066736863, 'PDE': 99.25876664922352,
                     'Quadrature': 99.30233353299197}
    expected_greeks = {'price_pde': 99.25876664922352, 'delta': 0.4275397355739159, 'gamma': -0.04838673805178928,
                       'vega': -0.006582582110301161, 'theta': 0.00032340023270862163, 'rho': -0.0015798927309687372}

    res_df, greeks = parachute_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_snowball_plus_demo():
    expected_data = {'结构': '看涨雪球(雪球增强，雪球plus)', 'MonteCarlo': 100.00865599002834, 'PDE': 100.02079002662461,
                     'Quadrature': 100.07052911745969}
    expected_greeks = {'price_pde': 100.02079002662461, 'delta': 0.7065034167317137, 'gamma': -0.03865143323940856,
                       'vega': -0.0055950740291760325, 'theta': 0.0002904252766744264, 'rho': -3.429244285797495e-06}

    res_df, greeks = snowball_plus_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_paris_snowball_demo():
    expected_data = {'pv': 100.01742416459761, 'delta': 0.43278297470384075, 'gamma': -0.03718829098156107,
                     'vega': -0.008608643540902107, 'theta': 0.0003964196043583001, 'rho': -0.0008765889543886373,
                     '结构': '巴黎雪球'}
    res = paris_snowball_demo.run()
    assert expected_data == pytest.approx(res, rel=1e-10)
