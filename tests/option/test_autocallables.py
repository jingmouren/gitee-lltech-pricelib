#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import pytest
import datetime
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from ..conftest import init_bsm_process
from pricelib import *
from examples.autocallable import (phoenix_demo, standard_snowball_demo, earlyprofit_snowball_demo,
                                   butterfly_snowball_demo, otm_snowball_demo, floored_snowball_demo,
                                   stepdown_snowball_demo, bothdown_snowball_demo, parachute_snowball_demo,
                                   snowball_plus_demo, paris_snowball_demo)


class TestAutocallNote:
    """测试Autocall Note(二元小雪球)"""

    @staticmethod
    def run(evaluation_date: datetime.date, spot: float):
        process = init_bsm_process(evaluation_date, s=spot, r=0.03, q=0.05, vol=0.2)
        # 3. 定价引擎，包括蒙特卡洛模拟、有限差分、数值积分
        mc_engine = MCAutoCallEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                     antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)
        quad_engine = QuadAutoCallEngine(process, quad_method=QuadMethod.Simpson, n_points=1301)
        pde_engine = FdmSnowBallEngine(process, s_step=800, n_smax=2, fdm_theta=1)
        # 4. 定义产品：Autocall Note(二元小雪球)
        results = pd.DataFrame()
        for callput in CallPut:
            option = AutoCall(s0=100, maturity=2, start_date=datetime.date(2022, 1, 5), lock_term=3,
                              barrier_out=103, coupon_out=0.045, coupon_div=0.02, callput=callput, obs_dates=None,
                              pay_dates=None, margin_lvl=1, trade_calendar=CN_CALENDAR, t_step_per_year=243,
                              engine=None)
            # 5.为产品设置定价引擎
            option.set_pricing_engine(mc_engine)
            price_mc = option.pv_and_greeks()

            option.set_pricing_engine(pde_engine)
            price_pde = option.pv_and_greeks()

            option.set_pricing_engine(quad_engine)
            price_quad = option.pv_and_greeks()

            df = pd.DataFrame([price_mc, price_pde, price_quad], index=['MonteCarlo', 'PDE', 'Quadrature'])
            df.insert(0, "期权类型", str(option))
            results = pd.concat([results, df], axis=0)

        results = results.reset_index()
        return results

    def test_first_day(self):
        """测试期初估值"""
        res_df = self.run(evaluation_date=datetime.date(2022, 1, 5), spot=100)
        expected_data = [{'index': 'MonteCarlo', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 99.93100555332597,
                          'delta': 0.04560718511393702, 'gamma': -0.0017764579370975753, 'vega': 0.006866722182664375,
                          'theta': 0.009587446233922492, 'rho': -0.8369559555542452},
                         {'index': 'PDE', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 99.96033974577038,
                          'delta': 0.04474355948097042, 'gamma': -0.002709507283910284, 'vega': 0.0057905353147731375,
                          'theta': 0.014941219562047081, 'rho': -0.8248703827510013},
                         {'index': 'Quadrature', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 99.96258810435373,
                          'delta': 0.03928482280267076, 'gamma': -0.001757213484637532, 'vega': -0.00032291027079622836,
                          'theta': 0.01507381313852818, 'rho': -0.8221706577589032},
                         {'index': 'MonteCarlo', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 100.33861697900203,
                          'delta': -0.006719327618149862, 'gamma': 0.0009661924013215639,
                          'vega': -0.0011945570559532825, 'theta': 0.010620002727208089, 'rho': -0.5299448769754207},
                         {'index': 'PDE', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': np.nan, 'delta': np.nan,
                          'gamma': np.nan, 'vega': np.nan, 'theta': np.nan, 'rho': np.nan},
                         {'index': 'Quadrature', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 100.36871158225928,
                          'delta': -0.006662976297697298, 'gamma': -0.001045828316065922, 'vega': 9.90501560949042e-05,
                          'theta': 0.013353193558771181, 'rho': -0.5263612203634693}]
        expected_df = pd.DataFrame(expected_data)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_last_day_1(self):
        """测试到期日, 敲出价上方"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=105)
        expected_data = [
            {'index': 'MonteCarlo', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 109.00000000000001, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 109.00000000000001, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'Quadrature', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 109.00000000000001, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'MonteCarlo', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 104.0, 'delta': 0.0, 'gamma': 0.0,
             'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': np.nan, 'delta': np.nan, 'gamma': np.nan,
             'vega': np.nan, 'theta': np.nan, 'rho': np.nan},
            {'index': 'Quadrature', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 104.0, 'delta': 0.0, 'gamma': 0.0,
             'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected_data)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_last_day_2(self):
        """测试到期日, 敲出价下方"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=95)
        expected_data = [
            {'index': 'MonteCarlo', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 104.0, 'delta': 0.0, 'gamma': 0.0,
             'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 104.0, 'delta': 0.0, 'gamma': 0.0,
             'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'Quadrature', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 104.0, 'delta': 0.0, 'gamma': 0.0,
             'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'MonteCarlo', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 109.00000000000001, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': np.nan, 'delta': np.nan, 'gamma': np.nan,
             'vega': np.nan, 'theta': np.nan, 'rho': np.nan},
            {'index': 'Quadrature', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 109.00000000000001, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected_data)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_penultimate_day(self):
        """测试倒数第二天"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 4), spot=102)
        expected_data = [{'index': 'MonteCarlo', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 105.09101202759277,
                          'delta': 1.0911603119050381, 'gamma': 0.5841984253932281, 'vega': 0.05534545087188292,
                          'theta': -1.0910120275927682, 'rho': 0.002070291195806817},
                         {'index': 'PDE', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 104.93900183414304,
                          'delta': 1.2475814729859098, 'gamma': 1.2166670865011868, 'vega': 0.04448723733024451,
                          'theta': -0.9390018341430704, 'rho': -8.250464362902221e-05},
                         {'index': 'Quadrature', '期权类型': 'AutoCall Note(二元小雪球)', 'pv': 105.07928482696474,
                          'delta': 1.0715455441597908, 'gamma': 0.575234196548749, 'vega': 0.045874167127550436,
                          'theta': -1.079284826964738, 'rho': 0.0004103781644744231},
                         {'index': 'MonteCarlo', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 107.89148184253199,
                          'delta': -1.091160311905045, 'gamma': -0.5841984253932144, 'vega': -0.05534545087188292,
                          'theta': 1.1085181574680263, 'rho': -0.007905348081493457},
                         {'index': 'PDE', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': np.nan, 'delta': np.nan,
                          'gamma': np.nan, 'vega': np.nan, 'theta': np.nan, 'rho': np.nan},
                         {'index': 'Quadrature', '期权类型': 'AutoPut Note(二元小雪球)', 'pv': 107.89442050012653,
                          'delta': -1.0715455441598047, 'gamma': -0.5752341965502787, 'vega': -0.04587416700508129,
                          'theta': 1.1055794998734854, 'rho': -0.009174547843997516}]
        expected_df = pd.DataFrame(expected_data)
        assert_frame_equal(res_df, expected_df, check_dtype=True)


class TestFCN:
    """测试FCN"""

    @staticmethod
    def run(evaluation_date: datetime.date, spot: float):
        process = init_bsm_process(evaluation_date, s=spot, r=0.02, q=0.04, vol=0.16)
        # 3. 定价引擎，包括蒙特卡洛模拟、有限差分、数值积分
        mc_engine = MCPhoenixEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                    antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
        quad_engine = QuadFCNEngine(process, quad_method=QuadMethod.Simpson, n_points=1301)
        pde_engine = FdmPhoenixEngine(process, s_step=800, n_smax=2, fdm_theta=1)

        # 4. 定义产品：FCN(Fixed Coupon Note固定派息票据)
        option = FCN(maturity=2, s0=100, start_date=datetime.date(2022, 1, 5), trade_calendar=CN_CALENDAR,
                     barrier_out=103, barrier_in=80, strike_upper=None, coupon=0.00314, lock_term=3,
                     engine=None, status=StatusType.NoTouch, t_step_per_year=243)
        # 5.为产品设置定价引擎
        option.set_pricing_engine(mc_engine)
        price_mc = option.pv_and_greeks()

        option.set_pricing_engine(pde_engine)
        price_pde = option.pv_and_greeks()

        option.set_pricing_engine(quad_engine)
        price_quad = option.pv_and_greeks()

        df = pd.DataFrame([price_mc, price_pde, price_quad], index=['MonteCarlo', 'PDE', 'Quadrature'])
        df.insert(0, "期权类型", str(option))
        df = df.reset_index()
        return df

    def test_first_day(self):
        """测试期初估值"""
        res_df = self.run(evaluation_date=datetime.date(2022, 1, 5), spot=100)
        expected = [{'index': 'MonteCarlo', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.05171787086645,
                     'delta': 0.07635024390848599, 'gamma': -0.015165972870889277, 'vega': -0.2387282542682101,
                     'theta': 0.012701412348405938, 'rho': -0.7136221312761393},
                    {'index': 'PDE', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.00675403778338,
                     'delta': 0.08314976924964412, 'gamma': -0.011802049851411311, 'vega': -0.24197337781235717,
                     'theta': 0.015131337693560454, 'rho': -0.699107384765469},
                    {'index': 'Quadrature', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.00714124444022,
                     'delta': 0.08163928634644435, 'gamma': -0.005477769623979611, 'vega': -0.2443748254342495,
                     'theta': 0.015040570251045438, 'rho': -0.7035455452124779}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_knockout_day(self):
        """测试敲出观察日，敲出线上方"""
        res_df = self.run(evaluation_date=datetime.date(2023, 1, 5), spot=104)
        expected = [{'index': 'MonteCarlo', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314,
                     'delta': -0.16258168039846246, 'gamma': 0.31265707768935086, 'vega': 0.0,
                     'theta': -0.0036999895239517855, 'rho': 0.0},
                    {'index': 'PDE', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314,
                     'delta': -0.019744791561714638, 'gamma': 0.03796955361753901, 'vega': 0.0,
                     'theta': -0.007857451128046478, 'rho': 0.0},
                    {'index': 'Quadrature', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314,
                     'delta': -0.1590626081067329, 'gamma': 0.3058896309744863, 'vega': 0.0,
                     'theta': 0.0027993324027875133, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_last_day_1(self):
        """测试到期日，未敲入未敲出"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=100)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'Quadrature', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.314, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_last_day_2(self):
        """测试到期日，敲入"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=79)
        expected = [{'index': 'MonteCarlo', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 99.314,
                     'delta': 0.9999999999999449, 'gamma': 2.277015656978369e-14, 'vega': 0.0, 'theta': np.nan,
                     'rho': 0.0},
                    {'index': 'PDE', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 99.314,
                     'delta': 0.9999999999999989,
                     'gamma': 2.277015656978369e-14, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
                    {'index': 'Quadrature', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 99.314,
                     'delta': 1.0000000000000078, 'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_penultimate_day(self):
        """测试倒数第二天"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 4), spot=82)
        expected = [{'index': 'MonteCarlo', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.30590069537448,
                     'delta': 0.018144003437236284, 'gamma': -0.0368077052172507, 'vega': -0.0015630135122108868,
                     'theta': 0.008099304625517334, 'rho': -0.0027166507625224767},
                    {'index': 'PDE', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.29562497052231,
                     'delta': 0.022477502654348103, 'gamma': -0.032539236461599265, 'vega': -0.002955125070457143,
                     'theta': 0.018375029477681437, 'rho': -0.003972152545600238},
                    {'index': 'Quadrature', '期权类型': 'FCN(Fixed Coupon Note固定派息票据)', 'pv': 100.30346702051233,
                     'delta': 0.01769893847376848, 'gamma': -0.03663982148852997, 'vega': -0.0013905131321791941,
                     'theta': 0.010532979487663852, 'rho': -0.0041003354883173415}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)


class TestDCN:
    """测试DCN"""

    @staticmethod
    def run(evaluation_date: datetime.date, spot: float):
        process = init_bsm_process(evaluation_date, s=spot, r=0.02, q=0.04, vol=0.16)
        # 3. 定价引擎，包括蒙特卡洛模拟、有限差分、数值积分
        mc_engine = MCPhoenixEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                    antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
        quad_engine = QuadFCNEngine(process, quad_method=QuadMethod.Simpson, n_points=1301)
        pde_engine = FdmPhoenixEngine(process, s_step=800, n_smax=2, fdm_theta=1)

        # 4. 定义产品：DCN(Digital Coupon Note 二元派息票据)
        option = DCN(maturity=2, s0=100, start_date=datetime.date(2022, 1, 5), trade_calendar=CN_CALENDAR,
                     barrier_out=103, barrier_in=80, strike_upper=None, barrier_yield=80, coupon=0.00387, lock_term=3,
                     engine=None, status=StatusType.NoTouch, t_step_per_year=243)
        # 5.为产品设置定价引擎
        option.set_pricing_engine(mc_engine)
        price_mc = option.pv_and_greeks()

        option.set_pricing_engine(pde_engine)
        price_pde = option.pv_and_greeks()

        option.set_pricing_engine(quad_engine)
        price_quad = option.pv_and_greeks()

        df = pd.DataFrame([price_mc, price_pde, price_quad], index=['MonteCarlo', 'PDE', 'Quadrature'])
        df.insert(0, "期权类型", str(option))
        df = df.reset_index()
        return df

    def test_first_day(self):
        """测试期初估值"""
        res_df = self.run(evaluation_date=datetime.date(2022, 1, 5), spot=100)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.03897337641266,
             'delta': 0.12449349817369182, 'gamma': -0.023520650192580206, 'vega': -0.33071415418423555,
             'theta': 0.016330219417440617, 'rho': -0.6463332114506528},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.00429339431805,
             'delta': 0.1290889351727742, 'gamma': -0.019322770749539586, 'vega': -0.3328581754794442,
             'theta': 0.01946855195340902, 'rho': -0.6338018244863406},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.00228931538513,
             'delta': 0.13155965751516874, 'gamma': -0.02444031589277529, 'vega': -0.34232309973504016,
             'theta': 0.017786502542534777, 'rho': -0.6377361521656724}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_knockout_day_1(self):
        """测试敲出观察日，敲出线上方"""
        res_df = self.run(evaluation_date=datetime.date(2023, 1, 5), spot=105)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.387, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': 0.047595868779879424, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.38700000000001,
             'delta': -0.0002048632423709694, 'gamma': 0.0003902042532168839, 'vega': -1.4210854715202004e-14,
             'theta': 0.04883308078795778, 'rho': -2.842170943040401e-14},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.387, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': 0.061642633744000364, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_knockout_day_2(self):
        """测试敲出观察日，派息线上方、敲出线下方"""
        res_df = self.run(evaluation_date=datetime.date(2023, 1, 5), spot=90)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.87701603274787,
             'delta': 0.28609273949720027, 'gamma': -0.04228534469885087, 'vega': -0.3543265633827133,
             'theta': -0.36412160892682266, 'rho': -0.6243317922675118},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.76409482753948,
             'delta': 0.2899013976410537, 'gamma': -0.03844849802346168, 'vega': -0.36342304314723606,
             'theta': -0.3602739223918121, 'rho': -0.6169204411645381},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.75709566815144,
             'delta': 0.29114601692649145, 'gamma': -0.03763138514286106, 'vega': -0.37128410944430357,
             'theta': -0.3634817340838481, 'rho': -0.6181149756510536}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_knockout_day_3(self):
        """测试敲出观察日，派息线下方"""
        res_df = self.run(evaluation_date=datetime.date(2023, 1, 5), spot=75)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 90.52117091053566,
             'delta': 0.814720829067331, 'gamma': -0.022042791467457492, 'vega': -0.2069070484346014,
             'theta': 0.011878827572274986, 'rho': -0.35430189473626683},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 90.3889346797555,
             'delta': 0.8102488972281208, 'gamma': -0.008781525082415302, 'vega': -0.21707380306482094,
             'theta': 0.015030699199257924, 'rho': -0.3562835280030896},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 90.74952510342814,
             'delta': 0.8056141448776847, 'gamma': -0.007238400857608617, 'vega': -0.20251411243781092,
             'theta': -0.37011927564746827, 'rho': -0.357011880397053}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_last_day_1(self):
        """测试到期日，未敲入未敲出"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=100)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.387, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.387, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.387, 'delta': 0.0,
             'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        # assert_frame_equal(res_df, expected_df, check_dtype=True)
        print(res_df.to_dict('records'))  # 将DataFrame转换为字典列表
        pd.set_option('display.max_columns', None)  # 显示所有列
        print(res_df)

    def test_last_day_2(self):
        """测试到期日，敲入"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 5), spot=79)
        expected = [{'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.0,
                     'delta': 0.9999999999999539, 'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
                    {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.0,
                     'delta': 1.0000000000000078, 'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0},
                    {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 99.0,
                     'delta': 1.0000000000000078, 'gamma': 0.0, 'vega': 0.0, 'theta': np.nan, 'rho': 0.0}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)

    def test_penultimate_day(self):
        """测试倒数第二天"""
        res_df = self.run(evaluation_date=datetime.date(2024, 1, 4), spot=82)
        expected = [
            {'index': 'MonteCarlo', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.37526296459806,
             'delta': 0.03658920612816049, 'gamma': -0.07150572954305838, 'vega': -0.0030064444179345173,
             'theta': 0.011737035401935714, 'rho': -0.0026605058436786067},
            {'index': 'PDE', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.36030176776694,
             'delta': 0.04096100061087157, 'gamma': -0.05930565090457058, 'vega': -0.004634434440873747,
             'theta': 0.026698232233030694, 'rho': -0.003876420038238848},
            {'index': 'Quadrature', '期权类型': 'DCN(Digital Coupon Note 二元派息票据)', 'pv': 100.37322979852743,
             'delta': 0.03637979787029794, 'gamma': -0.07302746997014209, 'vega': -0.0027675708252274944,
             'theta': 0.013770201472567578, 'rho': -0.004068063933814869}]
        expected_df = pd.DataFrame(expected)
        assert_frame_equal(res_df, expected_df, check_dtype=True)
        print(res_df.to_dict('records'))  # 将DataFrame转换为字典列表
        pd.set_option('display.max_columns', None)  # 显示所有列
        print(res_df)


def test_phoenix_demo():
    expected_data = {'结构': 'Phoenix 凤凰票据', 'MonteCarlo': 100.0397417322641, 'PDE': 100.0740834978835}
    expected_greeks = {'price_pde': 100.0740834978835, 'delta': 0.30693610759495726, 'gamma': -0.04191889361477763,
                       'vega': -0.6873103876611708, 'theta': 0.03410578625100413, 'rho': -0.3006235127738677}
    res_df, greeks = phoenix_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_standard_snowball_demo():
    expected_data = {'结构': '经典雪球(平敲雪球)', 'MonteCarlo': 99.99237077293571, 'PDE': 99.97242502284875,
                     'Quadrature': 100.01021095357333, 'pde_pct': '-0.020%', 'quad_pct': '0.018%'}
    expected_greeks = {'price_pde': 99.97242502284875, 'delta': 0.4336714279992435, 'gamma': -0.06160759915984215,
                       'vega': -0.7502448878742314, 'theta': 0.04425609512557571, 'rho': -0.19984183877359385}
    res_df, greeks = standard_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_earlyprofit_snowball_demo():
    expected_data = {'结构': '早利雪球', 'MonteCarlo': 99.97019888727107, 'PDE': 100.01918085274494,
                     'Quadrature': 100.04938425723334}
    expected_greeks = {'price_pde': 100.01918085274494, 'delta': 0.7485946436239885, 'gamma': -0.06779574769430496,
                       'vega': -0.4714328707591591, 'theta': 0.04273790484251094, 'rho': 0.03519575112444784}
    res_df, greeks = earlyprofit_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_butterfly_snowball_demo():
    expected_data = {'结构': '蝶式雪球', 'MonteCarlo': 100.00465809608768, 'PDE': 100.0364514590044,
                     'Quadrature': 100.0229173107232}
    expected_greeks = {'price_pde': 100.0364514590044, 'delta': 0.6003979584642778, 'gamma': -0.057819747897752904,
                       'vega': -0.6736305421984952, 'theta': 0.03749840960706763, 'rho': -0.08840143603079298}
    res_df, greeks = butterfly_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_otm_snowball_demo():
    expected_data = {'结构': 'OTM雪球', 'MonteCarlo': 100.00358496495079, 'PDE': 100.01671991322786,
                     'Quadrature': 100.0695778009394}
    expected_greeks = {'price_pde': 100.01671991322786, 'delta': 0.13769578938424587, 'gamma': -0.0216993090541564,
                       'vega': -0.38326463263842925, 'theta': 0.01884155036181312, 'rho': -0.4585777782421445}
    res_df, greeks = otm_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_floored_snowball_demo():
    expected_data = {'结构': '保底雪球(限损雪球，不追保雪球)', 'MonteCarlo': 100.04932829418274,
                     'PDE': 100.05319061780084, 'Quadrature': 100.08646854508989}
    expected_greeks = {'price_pde': 100.05319061780084, 'delta': 0.30476874786146624, 'gamma': -0.041764074317384825,
                       'vega': -0.49279151109935526, 'theta': 0.028830963019615297, 'rho': -0.2606915995998662}
    res_df, greeks = floored_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_stepdown_snowball_demo():
    expected_data = {'结构': '降敲雪球', 'MonteCarlo': 99.99000080300591, 'PDE': 99.99882667508119,
                     'Quadrature': 100.04927179943412}
    expected_greeks = {'price_pde': 99.99882667508119, 'delta': 0.42893445638513583, 'gamma': -0.05185686827974223,
                       'vega': -0.6690423157686496, 'theta': 0.03908593488756651, 'rho': -0.07847959340392663}
    res_df, greeks = stepdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_bothdown_snowball_demo():
    expected_data = {'结构': '双降雪球', 'MonteCarlo': 99.99734493746297, 'PDE': 100.03946242145815,
                     'Quadrature': 100.02280615164219}
    expected_greeks = {'price_pde': 100.03946242145815, 'delta': 0.6031430326171687, 'gamma': -0.05735408333077885,
                       'vega': -0.3797294860883085, 'theta': 0.04343410636937506, 'rho': 0.041062551228748134}
    res_df, greeks = bothdown_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_parachute_snowball_demo():
    expected_data = {'结构': '降落伞雪球', 'MonteCarlo': 100.00140677547931, 'PDE': 99.98244385775044,
                     'Quadrature': 100.0196889879925}
    expected_greeks = {'price_pde': 99.98244385775044, 'delta': 0.40278176156137846, 'gamma': -0.05003004622639651,
                       'vega': -0.6793836270392006, 'theta': 0.03305856903973847, 'rho': -0.1495028896361532}

    res_df, greeks = parachute_snowball_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_snowball_plus_demo():
    expected_data = {'结构': '看涨雪球(雪球增强，雪球plus)', 'MonteCarlo': 99.98093281705195, 'PDE': 99.99901087665904,
                     'Quadrature': 100.05768904941405}
    expected_greeks = {'price_pde': 99.99901087665904, 'delta': 0.654490097031001, 'gamma': -0.04116324325255505,
                       'vega': -0.605280941062901, 'theta': 0.029991508381016274, 'rho': -0.01271800889799124}

    res_df, greeks = snowball_plus_demo.run()
    assert expected_data == pytest.approx(res_df, rel=1e-10)
    assert expected_greeks == pytest.approx(greeks, rel=1e-10)


def test_paris_snowball_demo():
    expected_data = {'pv': 100.01742416459761, 'delta': 0.43278297470384075, 'gamma': -0.03718829098156107,
                     'vega': -0.8608643540902108, 'theta': 0.039641960435830015, 'rho': -0.08765889543886374,
                     '结构': '巴黎雪球'}
    res = paris_snowball_demo.run()
    assert expected_data == pytest.approx(res, rel=1e-10)
