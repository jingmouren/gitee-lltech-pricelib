#!/user/bin/env python
# -*- coding: utf-8 -*-
""" Autocall Note(二元小雪球) 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import numpy as np
import pandas as pd
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 4))
    t_step_per_year = 243
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.03, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.2, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    mc_engine = MCAutoCallEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                 antithetic_variate=False, ld_method=LdMethod.Halton, seed=0)
    quad_engine = QuadAutoCallEngine(process, quad_method=QuadMethod.Simpson, n_points=1001)
    pde_engine = FdmSnowBallEngine(process, s_step=800, n_smax=2, fdm_theta=1)
    # 4. 定义产品：Autocall Note(二元小雪球)
    results = []
    for callput in CallPut:
        option = AutoCall(s0=100, maturity=2, start_date=datetime.date(2022, 1, 4), lock_term=3,
                          trade_calendar=CN_CALENDAR, barrier_out=100, coupon_out=0.044, coupon_div=0.02,
                          callput=callput, engine=None,
                          obs_dates=None, pay_dates=None, margin_lvl=1, t_step_per_year=t_step_per_year)
        # 5.为产品设置定价引擎
        option.set_pricing_engine(mc_engine)
        price_mc = option.price()

        option.set_pricing_engine(pde_engine)
        price_pde = option.price() if callput == CallPut.Call else np.nan

        option.set_pricing_engine(quad_engine)
        price_quad = option.price()

        results.append([str(option), price_mc, price_pde, price_quad])

    df = pd.DataFrame(results, columns=['期权类型', 'MonteCarlo', 'PDE', 'Quadrature'])
    return df


if __name__ == '__main__':
    df1 = run()
    print(df1)
