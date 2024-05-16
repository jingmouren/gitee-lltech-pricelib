# !/user/bin/env python
# -*- coding: utf-8 -*-
""" 巴黎雪球 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    # 2&3. 定价引擎，蒙特卡洛模拟，使用默认的BSM模型
    mc_engine = MCParisSnowballEngine(n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                      antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0,
                                      s=100, r=0.02, q=0.06, vol=0.16)
    # 4. 定义产品：巴黎雪球
    option = ParisSnowball(maturity=2, s0=100, start_date=datetime.date(2022, 1, 5),
                           trade_calendar=CN_CALENDAR, barrier_out=103, barrier_in=75,
                           coupon_out=0.129, coupon_div=None, lock_term=12, engine=None, knock_in_times=2,
                           knockout_freq="w", status=StatusType.NoTouch, t_step_per_year=243, )
    # 5.为产品设置定价引擎
    option.set_pricing_engine(mc_engine)
    price_mc = option.price()
    return {"结构": str(option), "price_mc": price_mc, "delta": option.delta(), "gamma": option.gamma(),
            "theta": option.theta(), "vega": option.vega(), "rho": option.rho()}


if __name__ == '__main__':
    print(run())
