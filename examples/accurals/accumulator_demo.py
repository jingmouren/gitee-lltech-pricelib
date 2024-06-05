#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 累购、累沽 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import pandas as pd
from pricelib import *


def lite():
    """简易定价接口"""
    option = Accumulator(s0=100, maturity=2, barrier_out=110, strike=87.14, leverage_ratio=2,
                         s=100, r=0.02, q=0.04, vol=0.16)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    # 2&3. 定价引擎，随机过程使用默认的BSM
    mc_engine = MCAccumulatorEngine(n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                                    antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0,
                                    s=100, r=0.02, q=0.04, vol=0.16)
    # 4. 定义产品：累购、累沽
    accumulator_option = Accumulator(s0=100, maturity=2, start_date=datetime.date(2022, 1, 5),
                                    trade_calendar=CN_CALENDAR, barrier_out=110, strike=87.14,
                                    leverage_ratio=2, margin_lvl=0.2, obs_dates=None)
    decumulator_option = Accumulator(s0=100, maturity=2, start_date=datetime.date(2022, 1, 5),
                                     trade_calendar=CN_CALENDAR, barrier_out=90, strike=107.63,
                                     leverage_ratio=2, margin_lvl=0.2, obs_dates=None)
    result = []
    # 5.为产品设置定价引擎
    accumulator_option.set_pricing_engine(mc_engine)
    price_mc_accumulator = accumulator_option.price()

    decumulator_option.set_pricing_engine(mc_engine)
    price_mc_deccumulator = decumulator_option.price()

    result.append([str(accumulator_option), price_mc_accumulator])
    result.append([str(decumulator_option), price_mc_deccumulator])
    df = pd.DataFrame(result, columns=['期权类型', 'MonteCarlo'])
    return df


if __name__ == '__main__':
    print(lite())
    df1 = run()
    print(df1)
