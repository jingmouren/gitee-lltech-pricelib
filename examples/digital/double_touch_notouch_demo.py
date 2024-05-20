#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0

双边二元期权 示例
    欧式 - 二元凸式/二元凹式
    美式 - 美式双接触/美式双不接触
"""
import datetime
import pandas as pd
import numpy as np
from pricelib import *


def lite():
    """简易定价接口"""
    option = DoubleDigitalOption(maturity=1, bound=(80, 120), rebate=(10, 10), touch_type=TouchType.Touch,
                                 exercise_type=ExerciseType.American, payment_type=PaymentType.Expire,
                                 s=100, r=0.02, q=0.05, vol=0.2)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    set_evaluation_date(datetime.date(2021, 1, 5))
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.2, name="中证1000波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、有限差分、数值积分
    an_engine = AnalyticDoubleDigitalEngine(process, series_num=10)
    mc_engine = MCDoubleDigitalEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                                      antithetic_variate=False, ld_method=LdMethod.Halton, seed=0)
    pde_engine = FdmDigitalEngine(process, s_step=800, n_smax=2, fdm_theta=1)

    # 4. 定义产品：双边二元期权 - 欧式：二元凸式/二元凹式；美式：美式双接触/美式双不接触
    t_step_per_year = 243
    results_discrete = []
    results_continuous = []
    for exercise in [ExerciseType.European, ExerciseType.American]:
        for payment in PaymentType:
            if exercise == ExerciseType.European and payment == PaymentType.Hit:
                continue
            for touch_type in TouchType:
                # 离散观察(每日观察)
                option_discrete = DoubleDigitalOption(maturity=1, start_date=datetime.date(2021, 1, 5),
                                                      bound=(80, 120), rebate=(10, 10),
                                                      exercise_type=exercise, payment_type=payment,
                                                      touch_type=touch_type, discrete_obs_interval=1 / t_step_per_year,
                                                      t_step_per_year=t_step_per_year)
                # 连续观察
                option_continuous = DoubleDigitalOption(maturity=1, start_date=datetime.date(2021, 1, 5),
                                                        bound=(80, 120), rebate=(10, 10),
                                                        exercise_type=exercise, payment_type=payment,
                                                        touch_type=touch_type, discrete_obs_interval=None,
                                                        t_step_per_year=t_step_per_year)
                # 5.为产品设置定价引擎
                # 离散观察(每日观察)
                option_discrete.set_pricing_engine(an_engine)
                try:
                    price_an_discrete = option_discrete.price()
                except Exception as e:
                    price_an_discrete = np.nan
                    print(e)

                option_discrete.set_pricing_engine(mc_engine)
                price_mc = option_discrete.price()

                option_discrete.set_pricing_engine(pde_engine)
                price_pde_discrete = option_discrete.price()

                # 连续观察
                option_continuous.set_pricing_engine(an_engine)
                try:
                    price_an_continuous = option_continuous.price()
                except Exception as e:
                    price_an_continuous = np.nan
                    print(e)

                option_continuous.set_pricing_engine(pde_engine)
                price_pde_continuous = option_continuous.price()

                # 将结果添加到列表中
                results_discrete.append([str(option_discrete), price_an_discrete, price_mc, price_pde_discrete])
                results_continuous.append([str(option_continuous), price_an_continuous, price_pde_continuous])

    df1 = pd.DataFrame(results_discrete,
                       columns=['期权类型    ', '闭式解(每日观察)', 'MonteCarlo', 'PDE'])
    df2 = pd.DataFrame(results_continuous, columns=['期权类型    ', '闭式解(连续观察)', 'PDE'])
    return df1, df2


if __name__ == '__main__':
    dfa, dfb = run()
    print(dfa)
    print(dfb)
    print(lite())
