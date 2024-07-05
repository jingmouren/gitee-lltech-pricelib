#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0

亚式期权：几何平均/算术平均；替代标的资产价格/替代执行价；增强亚式
    解析解只支持平均结算价，无增强亚式
    蒙特卡洛支持几何平均/算术平均，支持增强亚式，支持替代标的资产价格/替代执行价
    二叉树支持几何平均/算术平均，支持增强亚式，支持替代标的资产价格，不支持替代执行价
"""
import datetime
import pandas as pd
import numpy as np
from pricelib import *


def lite():
    """简易定价接口"""
    option = AsianOption(strike=100, maturity=1, callput=CallPut.Call, ave_method=AverageMethod.Geometric,
                         substitute=AsianAveSubstitution.Underlying, enhanced=False, limited_price=None,
                         s=100, r=0.02, q=0.05, vol=0.16)
    return option.pv_and_greeks()


def run():
    """自行配置定价引擎 """
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    # 设置全局估值日
    set_evaluation_date(datetime.date(2023, 1, 5))
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = ConstantRate(value=0.02, name="无风险利率")
    dividend = ConstantRate(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")

    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    # 3. 定价引擎，包括解析解、蒙特卡洛模拟、二叉树
    an_engine = AnalyticAsianEngine(process)
    mc_engine = MCAsianEngine(process, n_path=100000, rands_method=RandsMethod.Pseudorandom,
                              antithetic_variate=True, ld_method=LdMethod.Halton, seed=0)
    bitree_engine = BiTreeAsianEngine(process, tree_branches=100, n_samples=200)

    # 4. 定义产品：亚式期权 - 算术平均/几何平均，增强亚式，替代标的资产价格
    results = []
    for substitute in AsianAveSubstitution:  # AsianAveSubstitution.Underlying替代标的结算价；AsianAveSubstitution.Strike替代执行价
        for ave_method in AverageMethod:  # AverageMethod.Geometric几何平均；AverageMethod.Arithmetic算术平均
            for callput in CallPut:  # 看涨/看跌
                for enhanced in [False, True]:  # 是否是增强亚式
                    if substitute == AsianAveSubstitution.Strike and enhanced:
                        continue
                    option = AsianOption(callput=callput, ave_method=ave_method, strike=100, maturity=1,
                                         substitute=substitute, enhanced=enhanced, limited_price=100,
                                         start_date=datetime.date(2023, 1, 5),
                                         end_date=None, obs_start=None, obs_end=None)
                    # 5.为产品设置定价引擎
                    if not enhanced and substitute == AsianAveSubstitution.Underlying:
                        option.set_pricing_engine(an_engine)
                        price_an = option.price()
                    else:
                        price_an = np.nan

                    option.set_pricing_engine(mc_engine)
                    price_mc = option.price()

                    if (not enhanced and ave_method == AverageMethod.Arithmetic
                            and substitute == AsianAveSubstitution.Underlying):
                        option.set_pricing_engine(bitree_engine)
                        price_tree = option.price()
                    else:
                        price_tree = np.nan

                    # 将结果添加到列表中
                    results.append([str(option), price_an, price_mc, price_tree])

    df = pd.DataFrame(results, columns=['类型', '闭式解', 'MonteCarlo', '二叉树'])
    return df


if __name__ == '__main__':
    print(lite())
    res_df = run()
    print(res_df)
