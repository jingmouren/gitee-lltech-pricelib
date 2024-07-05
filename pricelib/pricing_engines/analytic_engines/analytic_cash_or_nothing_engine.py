#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from scipy.stats import norm
from pricelib.common.utilities.enums import CallPut, ExerciseType, PaymentType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticCashOrNothingEngine(AnalyticEngine):
    """二元(数字)期权-现金或无(cash or nothing)-闭式解定价引擎
    Reiner and Rubinstein(1991b) 支持欧式二元(现金或无)、美式二元(立即支付)、一触即付(到期支付)，
    支持M. Broadie, P. Glasserman, S.G. Kou(1999)离散观察调整"""

    # pylint: disable=invalid-name
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)

        v = r - q - vol ** 2 / 2
        mu = (r - q) / vol ** 2 - 0.5
        K_S = prod.strike / spot
        denominator = vol * np.sqrt(tau)
        # 欧式
        if prod.exercise_type == ExerciseType.European:
            if tau == 0:  # 如果估值日是到期日
                if (prod.callput == CallPut.Call and spot >= prod.strike) or (
                        prod.callput == CallPut.Put and spot <= prod.strike):
                    return prod.rebate
            return prod.rebate * np.exp(-r * tau) * norm.cdf(prod.callput.value * (v * tau - np.log(K_S)) / denominator)
        if prod.exercise_type == ExerciseType.American:
            # 如果估值时的标的价格已经触碰行权价，直接返回行权收益的现值
            if (prod.callput == CallPut.Call and spot >= prod.strike) or (
                    prod.callput == CallPut.Put and spot <= prod.strike):
                if prod.payment_type == PaymentType.Hit:
                    rebate_v = prod.rebate
                elif prod.payment_type == PaymentType.Expire:
                    rebate_v = prod.rebate * math.exp(-r * tau)
                else:
                    raise ValueError("Invalid payment type")
                return rebate_v

            # 离散观察期权，M. Broadie, P. Glasserman, S.G. Kou(1999) 在连续观察期权解析解上加调整项
            if prod.discrete_obs_interval is not None:
                if prod.callput == CallPut.Call:  # 向上
                    strike = prod.strike * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
                elif prod.callput == CallPut.Put:  # 向下
                    strike = prod.strike * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
                else:
                    raise ValueError("不支持的看涨看跌类型")
            else:
                strike = prod.strike
            K_S = strike / spot
            # 美式，立即支付
            if prod.payment_type == PaymentType.Hit:
                lambda_ = np.sqrt(mu ** 2 + 2 * r / vol ** 2)
                return prod.rebate * (K_S ** (mu + lambda_) * norm.cdf(
                    -prod.callput.value * (np.log(K_S) / denominator + lambda_ * denominator)) +
                                      K_S ** (mu - lambda_) * norm.cdf(
                            -prod.callput.value * (np.log(K_S) / denominator - lambda_ * denominator)))
            # 美式，到期支付
            if prod.payment_type == PaymentType.Expire:
                return prod.rebate * np.exp(-r * tau) * (
                        norm.cdf((v * tau - np.log(K_S)) / denominator * prod.callput.value) +
                        K_S ** (2 * mu) * norm.cdf(
                            -(v * tau + np.log(K_S)) / denominator * prod.callput.value))
            raise ValueError("不支持的Payment type")
