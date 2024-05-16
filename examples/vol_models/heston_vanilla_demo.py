#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
import pandas as pd
from pricelib import *

if __name__ == '__main__':
    # 1. 从csv文件中读取div数据
    div = pd.read_csv("../../tests/div.csv")
    # 2. 市场数据，包括标的物价格、无风险利率、分红率
    spot_price = SimpleQuote(value=2.367, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = RateTermStructure.from_array(div['maturity'].values, div['q'].values)

    # 2. 随机过程，Heston价格动态
    process = HestonProcess(spot=spot_price, interest=riskfree, div=dividend, v0=0.02519171, var_theta=0.0216701,
                            var_kappa=4.010982, var_vol=0.1, var_rho=-0.29376858)

    # 3. 定价引擎，Heston半闭式解、蒙特卡洛模拟
    an_engine = AnalyticHestonVanillaEngine(process)
    mc_engine = MCVanillaEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)

    # 4. 定义产品：香草欧式期权
    t_step_per_year = 243
    results = []
    exercise_type = ExerciseType.European
    strike_arr = np.load('../../tests/strike.npy')
    maturity_arr = np.load('../../tests/maturity.npy')
    for callput in CallPut:
        for maturity in maturity_arr:
            for strike in strike_arr:
                option = VanillaOption(callput=callput, exercise_type=exercise_type, strike=strike, maturity=maturity, t_step_per_year=t_step_per_year)
                # 5.为产品设置定价引擎
                option.set_pricing_engine(an_engine)
                price_an = option.price()
                option.set_pricing_engine(mc_engine)
                price_mc = option.price()

                # 将结果添加到列表中
                results.append([f"{option.callput.name}", f"{maturity}", f"{strike}", price_an, price_mc])

    res_df = pd.DataFrame(results, columns=['类型', '到期日', '行权价', 'Heston_Analytical', 'Heston_MC'])
    print(res_df)
