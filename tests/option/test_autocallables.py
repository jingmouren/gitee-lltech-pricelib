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
    expected_data = [['AutoCall Note(二元小雪球)', 100.02334695138468, 100.04518499837937, 100.04953872339847],
                     ['AutoPut Note(二元小雪球)', 100.27662795636324, np.nan, 100.36285531539146]]
    expected_df = pd.DataFrame(expected_data, columns=['期权类型', 'MonteCarlo', 'PDE', 'Quadrature'])
    res_df = autocall_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_fcn_demo():
    expected_data = {'结构': 'FCN(Fixed Coupon Note固定派息票据)', 'MonteCarlo': 100.02061618526523,
                     'PDE': 100.02949962741458, 'Quadrature': 100.00768789210922}
    expected_greeks = {'price_pde': 100.02949962741458, 'delta': 0.079863375805175, 'gamma': -0.011293522807932277,
                       'theta': 0.00014837744137892627, 'vega': -0.0019338924676306135, 'rho': -0.005553939793139051}
    res_df, greeks = fcn_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_dcn_demo():
    expected_data = {'结构': 'DCN(Digital Coupon Note 二元派息票据)', 'MonteCarlo': 99.98826667251322,
                     'PDE': 100.00645069699783, 'Quadrature': 99.97325117020301}
    expected_greeks = {'price_pde': 100.00645069699783, 'delta': 0.12793388665285477, 'gamma': -0.018703707918120926,
                       'theta': 0.00019132147136446064, 'vega': -0.002689000373779038, 'rho': -0.004974060086765633}
    res_df, greeks = dcn_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_phoenix_demo():
    expected_data = {'结构': 'Phoenix 凤凰票据', 'MonteCarlo': 100.02433288908446, 'PDE': 99.79969615827606}
    expected_greeks = {'price_pde': 99.79969615827606, 'delta': 0.3243718409488139, 'gamma': -0.041285065939064225,
                       'theta': 0.00033965493900424805, 'vega': -0.005205386177546103, 'rho': -0.0021550642327044043}
    res_df, greeks = phoenix_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_standard_snowball_demo():
    expected_data = {'结构': '经典雪球(平敲雪球)', 'MonteCarlo': 100.00658971084546, 'PDE': 99.90910316342752,
                     'Quadrature': 100.07847347363567, 'pde_pct': '-0.097%', 'quad_pct': '0.072%'}
    expected_greeks = {'price_pde': 99.90910316342752, 'delta': 0.369474431864667, 'gamma': -0.058712416510275034,
                       'theta': 0.0003692511667713916, 'vega': -0.006932521782798631, 'rho': -0.00240883815238675}
    res_df, greeks = standard_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_earlyprofit_snowball_demo():
    expected_data = {'结构': '早利雪球', 'MonteCarlo': 99.93429261435013, 'PDE': 100.01918085274494,
                     'Quadrature': 100.08467880934458}
    expected_greeks = {'price_pde': 100.01918085274494, 'delta': 0.7485946436239885, 'gamma': -0.06779574769430496,
                       'theta': 0.00042737904842510944, 'vega': -0.004714328707591591, 'rho': 0.00035195751124447836}
    res_df, greeks = earlyprofit_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_butterfly_snowball_demo():
    expected_data = {'结构': '蝶式雪球', 'MonteCarlo': 100.0799579903268, 'PDE': 100.13361422282733,
                     'Quadrature': 100.20682094918288}
    expected_greeks = {'price_pde': 100.13361422282733, 'delta': 0.6024552484713723, 'gamma': -0.0582534100587111,
                       'theta': 0.0003771607953234479, 'vega': -0.006733108352314332, 'rho': -0.0008582065182750398}
    res_df, greeks = butterfly_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_otm_snowball_demo():
    expected_data = {'结构': 'OTM雪球', 'MonteCarlo': 100.00358496495079, 'PDE': 100.01671991322786,
                     'Quadrature': 100.0386975482511}
    expected_greeks = {'price_pde': 100.01671991322786, 'delta': 0.13769578938424587, 'gamma': -0.0216993090541564,
                       'theta': 0.00018841550361813118, 'vega': -0.0038326463263842923, 'rho': -0.004585777782421445}
    res_df, greeks = otm_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_floored_snowball_demo():
    expected_data = {'结构': '保底雪球(限损雪球，不追保雪球)', 'MonteCarlo': 100.04932829418274,
                     'PDE': 100.05319061780084, 'Quadrature': 100.12952354828286}
    expected_greeks = {'price_pde': 100.05319061780084, 'delta': 0.30476874786146624, 'gamma': -0.041764074317384825,
                       'theta': 0.00028830963019615296, 'vega': -0.004927915110993553, 'rho': -0.002606915995998662}
    res_df, greeks = floored_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_stepdown_snowball_demo():
    expected_data = {'结构': '降敲雪球', 'MonteCarlo': 99.99000080300591, 'PDE': 99.99882667508119,
                     'Quadrature': 100.07343786545685}
    expected_greeks = {'price_pde': 99.99882667508119, 'delta': 0.42893445638513583, 'gamma': -0.05185686827974223,
                       'theta': 0.0003908593488756651, 'vega': -0.006690423157686496, 'rho': -0.0007847959340392663}
    res_df, greeks = stepdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_bothdown_snowball_demo():
    expected_data = {'结构': '双降雪球', 'MonteCarlo': 99.99734493746297, 'PDE': 100.03946242145815,
                     'Quadrature': 100.03162766717074}
    expected_greeks = {'price_pde': 100.03946242145815, 'delta': 0.6031430326171687, 'gamma': -0.05735408333077885,
                       'theta': 0.00043434106369375056, 'vega': -0.003797294860883085, 'rho': 0.00041062551228748134}
    res_df, greeks = bothdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_parachute_snowball_demo():
    expected_data = {'结构': '降落伞雪球', 'MonteCarlo': 99.27530066736863, 'PDE': 99.26716711461428,
                     'Quadrature': 99.27687867564143}
    expected_greeks = {'price_pde': 99.26716711461428, 'delta': 0.4270126473324254, 'gamma': -0.04841506348533642,
                       'theta': 0.0003234177214700651, 'vega': -0.00638494875394386, 'rho': -0.0014908618185643264}

    res_df, greeks = parachute_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_snowball_plus_demo():
    expected_data = {'结构': '看涨雪球(雪球增强，雪球plus)', 'MonteCarlo': 100.00865599002834, 'PDE': 100.02079002662461,
                     'Quadrature': 100.04711084587932}
    expected_greeks = {'price_pde': 100.02079002662461, 'delta': 0.7065034167317137, 'gamma': -0.03865143323940856,
                       'theta': 0.0002904252766744264, 'vega': -0.0055950740291760325, 'rho': -3.429244285797495e-06}

    res_df, greeks = snowball_plus_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_paris_snowball_demo():
    expected_data = {'结构': '巴黎雪球', 'price_mc': 100.03026316804844, 'delta': 0.45528940073680246,
                     'gamma': -0.11358651829419841, 'theta': 2.9542325602847086e-05, 'vega': -0.00928284151707203,
                     'rho': -0.0010586692801366837}
    res = paris_snowball_demo.run()
    assert expected_data == pytest.approx(res, rel=1e-10)
