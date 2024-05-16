#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0

香草期权 示例
    包括 - 欧式/美式，看涨/看跌
"""
import pandas as pd
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、二叉树、有限差分、数值积分
    an_Eu_engine = AnalyticVanillaEuEngine(process)
    an_Am_engine = AnalyticVanillaAmEngine(process, an_method="BAW")  # an_method="Bjerksund&Stensland2002", "BAW"
    mc_engine = MCVanillaEngine(process, n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                                antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
    quad_engine = QuadVanillaEngine(process, quad_method=QuadMethod.Simpson, n_points=801, n_max=4)
    bitree_engine = BiTreeVanillaEngine(process, tree_branches=500)
    pde_engine = FdmVanillaEngine(process, s_step=400, n_smax=4, fdm_theta=0.5)

    # 4. 定义产品：香草期权，包括欧式、美式
    t_step_per_year = 243
    results = []
    for exercise_type in [ExerciseType.European, ExerciseType.American]:
        for callput in CallPut:
            option = VanillaOption(callput=callput, exercise_type=exercise_type, strike=100, maturity=0.25,
                                   t_step_per_year=t_step_per_year)
            # 5.为产品设置定价引擎
            if exercise_type == ExerciseType.European:
                option.set_pricing_engine(an_Eu_engine)
            elif exercise_type == ExerciseType.American:
                option.set_pricing_engine(an_Am_engine)
            price_an = option.price()

            option.set_pricing_engine(mc_engine)
            price_mc = option.price()

            option.set_pricing_engine(quad_engine)
            price_quad = option.price()

            option.set_pricing_engine(bitree_engine)
            price_tree = option.price()

            option.set_pricing_engine(pde_engine)
            price_pde = option.price()

            # 将结果添加到列表中
            results.append(
                [f"{option.exercise_type.value}{option.callput.name}香草", price_an, price_mc, price_quad, price_tree,
                 price_pde])

    df = pd.DataFrame(results, columns=['类型', '闭式解', 'MonteCarlo', 'Quadrature', '二叉树', 'PDE'])
    return df


if __name__ == '__main__':
    res = run()
    print(res)
