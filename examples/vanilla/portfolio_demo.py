#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 自定义香草期权组合 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import pandas as pd
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    set_evaluation_date(datetime.date(2024, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.03, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、二叉树、有限差分、数值积分
    an_engine = AnalyticPortfolioEngine(process)
    mc_engine = MCPortfolioEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                  antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)
    quad_engine = QuadPortfolioEngine(process, quad_method=QuadMethod.Simpson, n_points=401)
    bitree_engine = BiTreePortfolioEngine(process, tree_branches=500)
    pde_engine = FdmVanillaEngine(process, s_step=400, n_smax=4, fdm_theta=0.5)

    # 4. 定义产品：自定义香草期权组合
    t_step_per_year = 243
    results = []
    for callput in CallPut:
        port = VanillaPortfolio(trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                                maturity=1, start_date=datetime.date(2024, 1, 5))

        # 自定义香草组合：蝶式价差
        port.add_vanilla(strike=100, maturity=1, start_date=datetime.date(2024, 1, 5), callput=callput, position=2)
        port.add_vanilla(strike=120, maturity=1, start_date=datetime.date(2024, 1, 5), callput=callput, position=-1)
        port.add_vanilla(strike=80, maturity=1, start_date=datetime.date(2024, 1, 5), callput=callput, position=-1)

        # 5.为产品设置定价引擎
        port.set_pricing_engine(an_engine)
        price_an = port.price()

        port.set_pricing_engine(mc_engine)
        price_mc = port.price()

        port.set_pricing_engine(quad_engine)
        price_quad = port.price()

        port.set_pricing_engine(bitree_engine)
        price_tree = port.price()

        port.set_pricing_engine(pde_engine)
        price_pde = port.price()

        # 将结果添加到列表中
        results.append([f"{callput}", price_an, price_mc, price_quad, price_tree, price_pde])

    df = pd.DataFrame(results, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    return df


if __name__ == '__main__':
    df1 = run()
    print(df1)
