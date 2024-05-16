#!/user/bin/env python
# -*- coding: utf-8 -*-
""" Phoenix(凤凰票据) 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括蒙特卡洛模拟、有限差分
    mc_engine = MCPhoenixEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
    pde_engine = FdmPhoenixEngine(process, s_step=800, n_smax=2, fdm_theta=1)

    # 4. 定义产品：Phoenix(凤凰票据)
    option = Phoenix(maturity=2, s0=100, start_date=datetime.date(2022, 1, 5), trade_calendar=CN_CALENDAR,
                     barrier_out=100, barrier_in=75, barrier_yield=75, coupon=0.00745, lock_term=3, engine=None,
                     status=StatusType.NoTouch)

    # 5.为产品设置定价引擎
    option.set_pricing_engine(mc_engine)
    price_mc = option.price()

    option.set_pricing_engine(pde_engine)
    price_pde = option.price()
    pde_greeks = {"price_pde": price_pde, "delta": option.delta(), "gamma": option.gamma(), "theta": option.theta(),
                  "vega": option.vega(), "rho": option.rho()}

    results = {"结构": str(option), "MonteCarlo": price_mc, "PDE": price_pde}
    return results, pde_greeks


if __name__ == '__main__':
    res, greeks = run()
    for k, v in res.items():
        print(f'{k}: {v}')
    print(greeks)

