#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 经典雪球 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import *


def run():
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    t_step_per_year = 243
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.03, name="中证1000贴水率")
    volatility = BlackConstVol(0.15, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    mc_engine = MCAutoCallableEngine(process, n_path=100000,
                                     rands_method=RandsMethod.Pseudorandom,
                                     antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
    quad_engine = QuadSnowballEngine(process, quad_method=QuadMethod.Simpson, n_points=601, n_max=2)
    pde_engine = FdmSnowBallEngine(process, s_step=200, n_smax=2, fdm_theta=1)

    # 4. 定义产品：经典雪球
    barrier_in = 80
    option = StandardSnowball(maturity=1, s0=100, start_date=datetime.date(2022, 1, 5),
                              trade_calendar=CN_CALENDAR, barrier_out=103, barrier_in=barrier_in,
                              coupon_out=0.0884, coupon_div=None, lock_term=3, engine=None,
                              status=StatusType.NoTouch, t_step_per_year=t_step_per_year, )
    # 5.为产品设置定价引擎
    option.set_pricing_engine(pde_engine)
    price_pde = option.price()
    pde_greeks = {"price_pde": price_pde, "delta": option.delta(), "gamma": option.gamma(), "theta": option.theta(),
                  "vega": option.vega(), "rho": option.rho()}

    option.set_pricing_engine(mc_engine)
    price_mc = option.price()

    option.set_pricing_engine(quad_engine)
    price_quad = option.price()

    results = {"结构": str(option), "MonteCarlo": price_mc, "PDE": price_pde, "Quadrature": price_quad,
               "pde_pct": f"{price_pde / price_mc - 1:.3%}", "quad_pct": f"{price_quad / price_mc - 1:.3%}"}
    return results, pde_greeks


if __name__ == '__main__':
    res, greeks = run()
    for k, v in res.items():
        print(f'{k}: {v}')
    print(greeks)
