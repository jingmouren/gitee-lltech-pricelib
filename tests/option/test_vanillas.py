#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.vanilla import (vanilla_demo, portfolio_demo, spread_demo, straddle_demo, strangle_demo, collar_demo,
                              condor_demo, butterfly_demo, risk_reversal_demo, discount_call_demo)


def test_vanilla_demo():
    expected_data = [['欧式Call香草', 2.8017559322677457, 2.792247551316844, 2.801816845708865, 2.8113028654762493,
                      2.79556471527213],
                     ['欧式Put香草', 3.5432046686084036, 3.5266321547209594, 3.543265582049523, 3.554770735356251,
                      3.5370134556210067],
                     ['美式Call香草', 2.8599472704899265, 2.8939856472547505, 2.87264857952708, 2.8753981861636766,
                      2.8571950453058035],
                     ['美式Put香草', 3.5432046748536394, 3.5423058056556673, 3.5565758800631113, 3.554770735356251,
                      3.5370134556210067]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = vanilla_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_portfolio_demo():
    expected_data = [['CallPut.Call', -8.550992791872941, -8.5246737225969, -8.55093092936645, -8.557680911680746,
                      -8.550992791872941],
                     ['CallPut.Put', -8.55099279187288, -8.524673722596901, -8.55093092936645, -8.557680911680746,
                      -8.55099279187288]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = portfolio_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_spread_demo():
    expected_data = [
        ['牛市价差期权', 8.29508304773476, 8.30400985719278, 8.295108797601005, 8.29486715454609, 8.29508304773476],
        ['熊市价差期权', -8.29508304773476, -8.30400985719278, -8.295108797601005, -8.29486715454609,
         -8.29508304773476]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = spread_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_straddle_demo():
    expected_data = [
        ['v跨式', 12.348522194733306, 12.37827359386603, 12.348641801350531, 12.342350400683268, 12.348522194733306],
        ['^跨式', -12.348522194733306, -12.37827359386603, -12.348641801350531, -12.342350400683268,
         -12.348522194733306]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = straddle_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_strangle_demo():
    expected_data = [
        ['v宽跨式', 12.734971944943684, 12.762111463098753, 12.735073847640916, 12.738323279959287, 12.734971944943684],
        ['^宽跨式', -12.734971944943684, -12.762111463098753, -12.735073847640916, -12.738323279959287,
         -12.734971944943684]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = strangle_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_collar_demo():
    expected_data = [['买入领式', 108.94128414427007, 108.94898901831708, 108.94131376505032, 108.94160007194448,
                      108.94128414427007],
                     ['卖出领式', 91.05871585572993, 91.05101098168292, 91.05868623494968, 91.05839992805552,
                      91.05871585572993]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = collar_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_condor_demo():
    expected_data = [
        ['^鹰式', 7.920042408255584, 7.904073532177874, 7.919952801738883, 7.922758997253517, 7.920042408255584],
        ['v鹰式', -7.920042408255584, -7.904073532177874, -7.919952801738883, -7.922758997253517, -7.920042408255584]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = condor_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_butterfly_demo():
    expected_data = [['^蝶式期权', 2.3080606549898057, 2.3011756940223904, 2.308042951069796, 2.3175837840537867,
                      2.3080606549898057],
                     ['∨蝶式期权', -2.3080606549898057, -2.3011756940223904, -2.308042951069796, -2.3175837840537867,
                      -2.3080606549898057]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = butterfly_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_risk_reversal_demo():
    expected_data = [['/风险反转', -8.941284144270075, -8.948989018317075, -8.941313765050332, -8.941600071944478,
                      -8.941284144270075],
                     ['\\风险反转', 8.941284144270075, 8.948989018317075, 8.941313765050332, 8.941600071944478,
                      8.941284144270075]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = risk_reversal_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_discount_call_demo():
    expected_data = [
        ['折价看涨', 15.683130241456592, 15.67385445132041, 15.683077476980642, 15.681250224667934, 15.683130241456592]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    res_df = discount_call_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)
