#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 安全气囊 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import pandas as pd
from pricelib import *


def lite():
    """简易定价接口"""
    option = Airbag(maturity=1, strike=100, barrier=70, knockin_parti=1., call_parti=0.7, reset_call_parti=1.,
                    s=100, r=0.03, q=0.05, vol=0.2)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    pd.set_option('display.max_columns', None)
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2021, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.03, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.2, name="中证1000波动率")

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    an_engine = AnalyticAirbagEngine(process)
    mc_engine = MCAirbagEngine(process, n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                               antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
    pde_engine = FdmAirbagEngine(process, s_step=1600, n_smax=2, fdm_theta=1)

    # 4. 定义产品：安全气囊
    t_step_per_year = 243
    results_discrete = []
    results_continuous = []

    # 离散观察(每日观察)
    option_discrete = Airbag(maturity=1, start_date=datetime.date(2021, 1, 5), strike=100, barrier=70,
                             knockin_parti=1., call_parti=0.7, reset_call_parti=1.,
                             discrete_obs_interval=1 / t_step_per_year, t_step_per_year=t_step_per_year)
    # 连续观察
    option_continuous = Airbag(maturity=1, start_date=datetime.date(2021, 1, 5), strike=100, barrier=70,
                               knockin_parti=1., call_parti=0.7, reset_call_parti=1,
                               discrete_obs_interval=None, t_step_per_year=t_step_per_year)
    # 5.为产品设置定价引擎
    # # 离散观察(每日观察)
    option_discrete.set_pricing_engine(an_engine)
    price_an_discrete = option_discrete.price()

    option_discrete.set_pricing_engine(mc_engine)
    price_mc = option_discrete.price()

    option_discrete.set_pricing_engine(pde_engine)
    price_pde_discrete = option_discrete.price()

    # 连续观察
    option_continuous.set_pricing_engine(an_engine)
    price_an_continuous = option_continuous.price()

    option_continuous.set_pricing_engine(pde_engine)
    price_pde_continuous = option_continuous.price()

    # 将结果添加到列表中
    results_discrete.append([str(option_discrete), price_an_discrete, price_mc, price_pde_discrete])
    results_continuous.append([str(option_continuous), price_an_continuous, price_pde_continuous])

    # 6. 返回结果
    df1 = pd.DataFrame(results_discrete,
                       columns=['安全气囊', '闭式解(每日观察)', 'MonteCarlo', 'PDE'])
    df2 = pd.DataFrame(results_continuous, columns=['安全气囊', '闭式解(连续观察)', 'PDE'])
    return df1, df2


if __name__ == '__main__':
    dfa, dfb = run()
    print(dfa)
    print(dfb)
    print(lite())
