#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 区间累计 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import pandas as pd
from pricelib import *


def lite():
    """简易定价接口"""
    option = RangeAccural(s0=100, maturity=2, upper_strike=110, lower_strike=90, payment=0.1,
                          s=100, r=0.02, q=0.04, vol=0.16)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    # 2&3. 定价引擎，随机过程使用默认的BSM
    mc_engine = MCRangeAccuralEngine(n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                                     antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0,
                                     s=100, r=0.02, q=0.04, vol=0.16)
    # 4. 定义产品：区间累计
    option = RangeAccural(s0=100, maturity=2, start_date=datetime.date(2022, 1, 5),
                          trade_calendar=CN_CALENDAR, upper_strike=110, lower_strike=90, payment=0.1)

    # 5.为产品设置定价引擎
    option.set_pricing_engine(mc_engine)
    price_mc = option.price()

    result = {"结构": str(option), "MonteCarlo": price_mc}
    return result


if __name__ == '__main__':
    print(lite())
    res = run()
    print(res)
