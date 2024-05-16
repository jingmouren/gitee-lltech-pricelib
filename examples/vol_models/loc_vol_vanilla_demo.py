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
    # 1. 从csv文件中读取div和局部波动率loc_vol数据
    div = pd.read_csv("../../tests/div.csv")
    loc_vol_df = pd.read_csv('../../tests/loc_vol.csv', index_col=0)
    # 2. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    spot_price = SimpleQuote(value=2.367, name="510050.SH")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = RateTermStructure.from_array(div['maturity'].values, div['q'].values)
    expirations = loc_vol_df.index.values
    strikes = loc_vol_df.columns.values.astype(float)
    volval = loc_vol_df.values
    volatility = LocalVolSurface(expirations=expirations, strikes=strikes, volval=volval)

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，蒙特卡洛模拟
    mc_engine = MCVanillaEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)
    pde_engine = FdmVanillaEngine(process, s_step=400, n_smax=4, fdm_theta=0.5)

    # 4. 定义产品：香草欧式期权
    t_step_per_year = 243
    results = []
    exercise_type = ExerciseType.European
    strike_arr = np.load('../../tests/strike.npy')
    maturity_arr = np.load('../../tests/maturity.npy')
    for callput in CallPut:
        for maturity in maturity_arr:
            for strike in strike_arr:
                option = VanillaOption(callput=callput, exercise_type=exercise_type, strike=strike, maturity=maturity,
                                       t_step_per_year=t_step_per_year, annual_days=AnnualDays.N365,
                                       trade_calendar=CN_CALENDAR)
                # 5.为产品设置定价引擎
                option.set_pricing_engine(mc_engine)
                price_mc = option.price()

                option.set_pricing_engine(pde_engine)
                price_pde = option.price()

                # 将结果添加到列表中
                results.append([f"{option.callput.name}", f"{maturity}", f"{strike}", price_mc, price_pde])

    df = pd.DataFrame(results, columns=['类型', '到期日', '行权价', 'LV_MC', 'LV_PDE'])
    df['MKT'] = pd.read_csv('../../tests/mkt_vanilla_price.csv')['close'].values
    print(df)
