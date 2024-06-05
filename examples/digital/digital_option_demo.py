#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0

单边二元(数字)期权-现金或无(cash or nothing) 示例
    包括欧式二元(现金或无)、美式二元(立即支付)、一触即付(到期支付)
"""
import datetime
import pandas as pd
from pricelib import *


def lite():
    """简易定价接口"""
    option = DigitalOption(maturity=1, strike=120, rebate=10, callput=CallPut.Call,
                           exercise_type=ExerciseType.European, payment_type=PaymentType.Expire,
                           s=100, r=0.02, q=0.05, vol=0.16)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2021, 1, 5))
    t_step_per_year = 243
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    an_engine = AnalyticCashOrNothingEngine(process)
    mc_engine = MCDigitalEngine(process, n_path=100000,
                                rands_method=RandsMethod.Pseudorandom,
                                antithetic_variate=False, ld_method=LdMethod.Halton, seed=0)
    quad_engine = QuadDigitalEngine(process, quad_method=QuadMethod.Simpson, n_points=2001)
    bitree_engine = BiTreeDigitalEngine(process, tree_branches=500)
    pde_engine = FdmDigitalEngine(process, s_step=800, n_smax=2, fdm_theta=1)

    # 4. 定义产品：二元期权
    results_discrete = []
    results_continuous = []
    for exercise in [ExerciseType.European, ExerciseType.American]:
        for payment in PaymentType:
            if exercise == ExerciseType.European and payment == PaymentType.Hit:
                continue
            for callput in CallPut:
                # 离散观察(每日观察)
                option_discrete = DigitalOption(maturity=1, start_date=datetime.date(2021, 1, 5),
                                                strike=100 + callput.value * 20, rebate=10,
                                                exercise_type=exercise, payment_type=payment,
                                                callput=callput, discrete_obs_interval=1 / t_step_per_year,
                                                t_step_per_year=t_step_per_year)
                # 连续观察
                option_continuous = DigitalOption(maturity=1, start_date=datetime.date(2021, 1, 5),
                                                  strike=100 + callput.value * 20, rebate=10,
                                                  exercise_type=exercise, payment_type=payment,
                                                  callput=callput, discrete_obs_interval=None,
                                                  t_step_per_year=t_step_per_year)
                # 5.为产品设置定价引擎
                # 离散观察(每日观察)
                option_discrete.set_pricing_engine(an_engine)
                price_an_discrete = option_discrete.price()

                option_discrete.set_pricing_engine(mc_engine)
                price_mc = option_discrete.price()

                option_discrete.set_pricing_engine(quad_engine)
                price_quad = option_discrete.price()

                option_discrete.set_pricing_engine(bitree_engine)
                price_tree = option_discrete.price()

                option_discrete.set_pricing_engine(pde_engine)
                price_pde_discrete = option_discrete.price()

                # 连续观察
                option_continuous.set_pricing_engine(an_engine)
                price_an_continuous = option_continuous.price()

                option_continuous.set_pricing_engine(pde_engine)
                price_pde_continuous = option_continuous.price()

                # 将结果添加到列表中
                results_discrete.append(
                    [str(option_discrete), price_an_discrete, price_mc, price_quad, price_tree, price_pde_discrete])
                results_continuous.append([str(option_continuous), price_an_continuous, price_pde_continuous])
    # 6. 结果输出
    discrete_df = pd.DataFrame(results_discrete,
                               columns=['期权类型    ', '闭式解(每日观察)', 'MonteCarlo', 'Quadrature', '二叉树',
                                        'PDE'])
    continuous_df = pd.DataFrame(results_continuous, columns=['期权类型    ', '闭式解(连续观察)', 'PDE'])
    return discrete_df, continuous_df


if __name__ == '__main__':
    print(lite())
    df1, df2 = run()
    print(df1)
    print(df2)
