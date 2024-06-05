#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 双鲨结构 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import pandas as pd
from pricelib import *


def lite():
    """简易定价接口"""
    option = DoubleShark(maturity=1, strike=(90, 110), bound=(80, 120), rebate=(3, 3), parti=(0.5, 0.5),
                         payment_type=PaymentType.Expire, s=100, r=0.03, q=0.03, vol=0.2)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2021, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.03, name="无风险利率")
    dividend = ConstantRate(value=0.03, name="中证1000贴水率")
    volatility = BlackConstVol(0.2, name="中证1000波动率")

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    Ikeda_Kunitomo_1992_engine = AnalyticDoubleSharkEngine(process, formula_type="Ikeda&Kunitomo1992",
                                                           series_num=10, delta1=0, delta2=0)
    Haug_1998_engine = AnalyticDoubleSharkEngine(process, formula_type="Haug1998")
    mc_engine = MCDoubleSharkEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                    antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)

    pde_engine = FdmDoubleSharkEngine(process, s_step=800, n_smax=2, fdm_theta=1)
    quad_engine = QuadDoubleSharkEngine(process, quad_method=QuadMethod.Trapezoid, n_points=2001)

    # 4. 定义产品：双鲨结构
    t_step_per_year = 243
    results_discrete = []
    results_continuous = []

    # 离散观察(每日观察)
    option_discrete = DoubleShark(maturity=1, strike=(90, 110), start_date=datetime.date(2021, 1, 5),
                                  bound=(80, 120), rebate=(3, 3), window=(None, None), parti=(0.5, 0.5),
                                  discrete_obs_interval=1 / t_step_per_year, exercise_type=ExerciseType.American,
                                  payment_type=PaymentType.Expire, t_step_per_year=t_step_per_year)
    # 连续观察
    option_continuous = DoubleShark(maturity=1, strike=(90, 110), start_date=datetime.date(2021, 1, 5),
                                    bound=(80, 120), rebate=(3, 3), window=(None, None), parti=(0.5, 0.5),
                                    discrete_obs_interval=None, exercise_type=ExerciseType.American,
                                    payment_type=PaymentType.Expire, t_step_per_year=t_step_per_year)
    # 5.为产品设置定价引擎
    # 离散观察(每日观察)
    option_discrete.set_pricing_engine(Ikeda_Kunitomo_1992_engine)
    Ikeda_Kunitomo_1992_discrete = option_discrete.price()

    option_discrete.set_pricing_engine(Haug_1998_engine)
    Haug_1998_discrete = option_discrete.price()

    option_discrete.set_pricing_engine(mc_engine)
    price_mc = option_discrete.price()

    option_discrete.set_pricing_engine(pde_engine)
    price_pde_discrete = option_discrete.price()

    option_discrete.set_pricing_engine(quad_engine)
    price_quad = option_discrete.price()

    # 连续观察
    option_continuous.set_pricing_engine(Ikeda_Kunitomo_1992_engine)
    Ikeda_Kunitomo_1992_continuous = option_continuous.price()

    option_continuous.set_pricing_engine(Haug_1998_engine)
    Haug_1998_continuous = option_continuous.price()

    option_continuous.set_pricing_engine(pde_engine)
    price_pde_continuous = option_continuous.price()

    # 将结果添加到列表中
    results_discrete.append([str(option_discrete), Ikeda_Kunitomo_1992_discrete, Haug_1998_discrete,
                             price_mc, price_pde_discrete, price_quad])
    results_continuous.append([str(option_continuous), Ikeda_Kunitomo_1992_continuous, Haug_1998_continuous,
                               price_pde_continuous])

    df1 = pd.DataFrame(results_discrete,
                       columns=['双鲨期权(每日观察)', 'I&K1992', 'Haug1998', 'MonteCarlo', 'PDE', 'Quad'])
    df2 = pd.DataFrame(results_continuous, columns=['双鲨期权(连续观察)', 'I&K1992', 'Haug1998', 'PDE', ])
    return df1, df2


if __name__ == '__main__':
    print(lite())
    pd.set_option('display.max_columns', None)
    dfa, dfb = run()
    print(dfa)
    print(dfb)
