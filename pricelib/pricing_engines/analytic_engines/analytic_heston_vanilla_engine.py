#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import CallPut, ProcessType, ExerciseType
from pricelib.common.processes import StochProcessBase
from pricelib.common.pricing_engine_base import AnalyticEngine
from pricelib.common.time import global_evaluation_date


class AnalyticHestonVanillaEngine(AnalyticEngine):
    """Heston半闭式解定价引擎"""

    def __init__(self, stoch_process: StochProcessBase):
        """
        初始化Heston半闭式解定价引擎
        Args:
            stoch_process: HestonProcess1D随机过程
        """
        assert stoch_process.process_type == ProcessType.Heston, "Error: Heston半闭式解定价引擎只能使用HestonProcess1D随机过程"
        super().__init__(stoch_process)
        # 以下属性指向需要定价的产品，由calc_present_value方法设置
        self.prod = None

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert prod.exercise_type == ExerciseType.European, "Error: 欧式期权的行权类型只能是European"
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        tau = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        q = self.process.div(tau)
        p_value1 = self.cal_p_value(prod.strike, tau, spot, r, q, self.process.v0, self.process.var_theta,
                                    self.process.var_rho, self.process.var_kappa, self.process.var_vol, 1)
        p_value2 = self.cal_p_value(prod.strike, tau, spot, r, q, self.process.v0, self.process.var_theta,
                                    self.process.var_rho, self.process.var_kappa, self.process.var_vol, 2)
        if prod.callput == CallPut.Call:
            price = spot * np.exp(-q * tau) * p_value1 - prod.strike * np.exp(-r * tau) * p_value2
        elif prod.callput == CallPut.Put:
            price = prod.strike * np.exp(-r * tau) * (1 - p_value2) - spot * np.exp(-q * tau) * (1 - p_value1)
        return price

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    def cal_p_value(self, K, t, s0, r, q, v0, theta, rho, kappa, sigma, j_type, deg=64, m=250):
        """
        Heston猜解Pj，j=1或2，对应BSM解析解中的N(d1)或N(d2)，通过数值积分计算
        """

        def integral_func(phi):
            """被积函数(复数)"""
            return self.char_fun(t, s0, r, q, v0, theta, rho, kappa, sigma, j_type, phi) * np.exp(
                -1j * phi * np.log(K)) / (1j * phi)
            # 初始化高斯-勒让德积分，deg是积分划分节点，愈多精度愈高。x是高斯点，w是高斯点对应的权重

        x, w = np.polynomial.legendre.leggauss(deg)
        x = (x + 1) * 0.5 * m
        return 0.5 + (1 / np.pi) * np.sum(w * integral_func(x)).real * 0.5 * m  # 计算积分

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    @staticmethod
    def char_fun(t, s0, r, q, v0, theta, rho, kappa, sigma, j_type, phi):
        """Heston特征函数"""
        if j_type == 1:
            u = 0.5
            b = kappa - rho * sigma
        elif j_type == 2:
            u = -0.5
            b = kappa
        else:
            raise ValueError(f'heston模型半闭式解中的j_type只能是1或2，当前为j_type={j_type}')
        a = kappa * theta
        x = np.log(s0)
        d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
        capitalC = (r - q) * phi * 1j * t + (a / sigma ** 2) * (
                (b - rho * sigma * phi * 1j + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
        capitalD = ((b - rho * sigma * phi * 1j + d) / sigma ** 2) * (1 - np.exp(d * t)) / (1 - g * np.exp(d * t))
        return np.exp(capitalC + capitalD * v0 + 1j * phi * x)
