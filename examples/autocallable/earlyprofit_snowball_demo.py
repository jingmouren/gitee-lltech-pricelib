#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 早利雪球 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import *


def lite():
    """简易定价接口"""
    option = EarlyProfitSnowball(maturity=2, lock_term=3, s0=100, barrier_out=103, barrier_in=80,
                                 coupon_out1=0.231, coupon_out2=0.05, coupon_stair_ends=(12, 24),
                                 s=100, r=0.02, q=0.03, vol=0.15)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.03, name="中证1000贴水率")
    volatility = BlackConstVol(0.15, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括蒙特卡洛模拟、有限差分、数值积分
    mc_engine = MCAutoCallableEngine(process, n_path=200000, rands_method=RandsMethod.Pseudorandom,
                                     antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)
    quad_engine = QuadSnowballEngine(process, quad_method=QuadMethod.Simpson, n_points=2001)
    pde_engine = FdmSnowBallEngine(process, s_step=800, n_smax=2, fdm_theta=1)

    # 4. 定义产品：早利雪球
    option = EarlyProfitSnowball(maturity=2, s0=100, start_date=datetime.date(2022, 1, 5),
                                 trade_calendar=CN_CALENDAR, barrier_out=103, barrier_in=80, coupon_out1=0.231,
                                 coupon_out2=0.05, coupon_div=None, lock_term=3,
                                 coupon_stair_ends=(12, 24), pay_dates=None, engine=None,
                                 status=StatusType.NoTouch, t_step_per_year=243)
    # 5.为产品设置定价引擎
    option.set_pricing_engine(mc_engine)
    price_mc = option.price()

    option.set_pricing_engine(pde_engine)
    price_pde = option.price()
    pde_greeks = {"price_pde": price_pde, "delta": option.delta(), "gamma": option.gamma(), "vega": option.vega(),
                  "theta": option.theta(), "rho": option.rho()}

    option.set_pricing_engine(quad_engine)
    price_quad = option.price()

    results = {"结构": str(option), "MonteCarlo": price_mc, "PDE": price_pde, "Quadrature": price_quad}
    return results, pde_greeks


if __name__ == '__main__':
    print(lite())
    res, greeks = run()
    for k, v in res.items():
        print(f'{k}: {v}')
    print(greeks)
