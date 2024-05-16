#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from scipy.stats import norm
from pricelib.common.utilities.enums import CallPut, PaymentType, ExerciseType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticAssetOrNothingEngine(AnalyticEngine):
    """二元(数字)期权-资产或无(asset or nothing)-闭式解定价引擎
    由于到期获得的payoff是标的资产多头，因此rebate固定为0"""

    # pylint: disable=invalid-name
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert prod.rebate == 0, "资产或无期权(asset or nothing)的rebate必须为0"
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)

        numerator = r - q + vol ** 2 / 2
        denominator = vol * np.sqrt(tau)
        mu = (r - q) / vol ** 2 - 0.5
        # 欧式
        if prod.exercise_type == ExerciseType.European:
            K_S = prod.strike / spot
            return spot * np.exp(-q * tau) * norm.cdf(
                prod.callput.value * (numerator * tau - np.log(K_S) / denominator))
        if prod.exercise_type == ExerciseType.American:
            if prod.discrete_obs_interval is not None:
                # 均匀离散观察障碍期权，M. Broadie, P. Glasserman, S.G. Kou(1997) 在连续障碍期权解析解上加调整项，调整障碍价格水平
                # 指数上的beta = -zeta(1/2) / sqrt(2*pi) = 0.5826, 其中zeta是黎曼zeta函数
                if prod.callput == CallPut.Call:  # 向上
                    strike = prod.strike * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
                else:  # self.callput == CallPut.Put:  # 向下
                    strike = prod.strike * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            else:  # 连续观察
                strike = prod.strike
            K_S = strike / spot
            # 美式，立即支付
            if prod.payment_type == PaymentType.Hit:
                lambda_ = np.sqrt(mu ** 2 + 2 * r / vol ** 2)
                return strike * (K_S ** (mu + lambda_) * norm.cdf(
                    -prod.callput.value * (np.log(K_S) / denominator + lambda_ * denominator)) + K_S ** (
                                         mu - lambda_) * norm.cdf(
                    -prod.callput.value * (np.log(K_S) / denominator - lambda_ * denominator)))
            # 美式，到期支付
            # else:  self.payment_type == PaymentType.Expire:
            return spot * np.exp(-q * tau) * (
                    norm.cdf((numerator * tau - np.log(K_S)) / denominator * prod.callput.value) +
                    K_S ** (2 * mu + 2) * norm.cdf(
                        -(numerator * tau + np.log(K_S)) / denominator * prod.callput.value))
        # 观察方式既不是欧式也不是美式
        raise NotImplementedError(
            f"ExerciseType {prod.exercise_type.value}不支持的观察方式，仅支持欧式观察或美式观察.")
