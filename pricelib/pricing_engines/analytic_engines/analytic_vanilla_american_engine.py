#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from scipy.stats import norm, multivariate_normal
from pricelib.common.utilities.enums import CallPut, ExerciseType
from pricelib.common.processes import StochProcessBase
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticVanillaAmEngine(AnalyticEngine):
    """美式香草期权-解析解定价引擎
    1. BAW - Barone-Adesi and Whaley(1987)方法
    2. Bjerksund&Stensland2002 - Bjerksund and Stensland(2002)方法"""

    # pylint: disable=invalid-name
    def __init__(self, stoch_process: StochProcessBase = None, an_method="BAW", *,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: 随机过程对象
            an_method: 美式期权解析解方法："BAW",或 "Bjerksund&Stensland2002"
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.an_method = an_method
        # 以下属性指向需要定价的产品，由calc_present_value方法设置
        self.prod = None
        # 以下为计算过程的中间变量
        self.spot = None
        self.T = None
        self.vol = None
        self.r = 0
        self.q = 0
        self.q_star = None
        self.beta = 1
        self.b = 0
        self.strike = None  # Bjerksund and Stensland(2002)方法计算Put时需要颠倒strike和spot，此变量临时存储strike

    # pylint: disable=invalid-name
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert prod.exercise_type == ExerciseType.American, "Error: 美式期权的行权类型只能是American"
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        self.T = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        self.spot = spot
        self.r = self.process.interest(self.T)
        self.q = self.process.div(self.T)
        self.vol = self.process.vol(self.T, spot)
        if self.q == 0 and prod.callput == CallPut.Call:
            # 如果没有分红，看涨期权的美式期权价值等于欧式期权价值
            return self.BSM_value(self.spot)
        if self.an_method == "Bjerksund&Stensland2002":
            price = self._Bjerksund_Stensland_2002()
        elif self.an_method == "BAW":
            price = self._BAW_1987()
        else:
            raise ValueError(f"{self.an_method} - 不支持的美式期权解析解方法")

        price = max(price, self.BSM_value(self.spot))
        return price

    # pylint: disable=invalid-name
    def d1(self, S):
        """BSM d1   S: 标的价格"""
        d1_value = (np.log(S / self.prod.strike) + (self.r - self.q + 0.5 * self.vol ** 2) * self.T) / (
                self.vol * np.sqrt(self.T))
        return d1_value

    # pylint: disable=invalid-name
    def d2(self, S):
        """BSM d2   S: 标的价格"""
        d2_value = self.d1(S) - self.vol * np.sqrt(self.T)
        return d2_value

    # pylint: disable=invalid-name
    def BSM_value(self, S):
        """BSM公式欧式香草期权现值    S: 标的价格"""
        return self.prod.callput.value * (
                S * math.exp(-self.q * self.T) * norm.cdf(self.prod.callput.value * self.d1(S))
                - self.prod.strike * math.exp(-self.r * self.T) * norm.cdf(
            self.prod.callput.value * self.d2(S)))

    # pylint: disable=invalid-name, missing-docstring
    def _BAW_1987(self):
        """Barone-Adesi and Whaley (1987)近似解"""
        if self.T == 0:  # 如果估值日就是到期日
            return max(self.prod.callput.value * (self.spot - self.prod.strike), 0)

        def A(S_star):
            return self.prod.callput.value * (S_star / self.q_star * (
                    1 - math.exp(-self.q * self.T) * norm.cdf(self.prod.callput.value * self.d1(S_star))))

        def cal_S_star(S):
            LHS = self.prod.callput.value * (S - self.prod.strike)
            RHS = self.BSM_value(S) + A(S)
            n_iter = 0
            while abs(LHS - RHS) / self.prod.strike > 1e-6 and n_iter < 100:
                nom_b_1 = self.prod.callput.value * math.exp(-self.q * self.T) * norm.cdf(
                    self.prod.callput.value * self.d1(S)) * (1 - 1 / self.q_star)
                nom_b_2 = (1 - self.prod.callput.value * math.exp(-self.q * self.T) * norm.cdf(
                    self.prod.callput.value * self.d1(S)) / (
                                   self.vol * math.sqrt(self.T))) / self.q_star
                b = nom_b_1 + self.prod.callput.value * nom_b_2
                S = (self.prod.strike + self.prod.callput.value * RHS - self.prod.callput.value * b * S) / (
                        1 - self.prod.callput.value * b)
                LHS = self.prod.callput.value * (S - self.prod.strike)
                RHS = self.BSM_value(S) + A(S)
                n_iter += 1
            return S

        M = 2 * self.r / self.vol ** 2
        N = 2 * (self.r - self.q) / self.vol ** 2
        K = 1 - math.exp(-self.r * self.T)
        self.q_star = (-(N - 1) + self.prod.callput.value * math.sqrt((N - 1) ** 2 + 4 * M / K)) * 0.5

        s_inf = self.prod.strike / (1 - 2 / (N + 1 + self.prod.callput.value * ((N - 1) ** 2 + 4 * M) ** 0.5))
        h = -((self.r - self.q) * self.T + self.prod.callput.value * 2 * self.vol * self.T ** 0.5) * (
                self.prod.strike / (s_inf - self.prod.strike))

        if self.prod.callput == CallPut.Call:
            S_star = self.prod.strike + (s_inf - self.prod.strike) * (1 - math.e ** h)
            S_star = cal_S_star(S_star)
            if self.spot >= S_star:
                return self.spot - self.prod.strike
            # else:
            return self.BSM_value(self.spot) + A(S_star) * (self.spot / S_star) ** self.q_star
        if self.prod.callput == CallPut.Put:
            S_star = s_inf - (s_inf - self.prod.strike) * (1 - math.e ** h)
            S_star = cal_S_star(S_star)
            if self.spot <= S_star:
                return self.prod.strike - self.spot
            # else:
            return self.BSM_value(self.spot) + A(S_star) * (self.spot / S_star) ** self.q_star
        raise ValueError("Error: CallPut类型错误！")

    def _Bjerksund_Stensland_2002(self):
        """Bjerksund Stensland (2002)的美式期权近似解
            看涨期权，直接计算；
            看跌期权，公式中需要调换标的价格和执行价的值"""
        self.b = self.r - self.q
        assert self.b != 0, "Error: Bjerksund_Stensland_2002不支持 r-q=0 的情形！"
        if self.prod.callput is CallPut.Call:
            self.strike = self.prod.strike
            price = self._Bjerksund_Stensland_2002_am_call_value(self.r, self.b)
        else:  # self.callput is CallPut.Put:
            self.strike, self.spot = self.spot, self.prod.strike
            price = self._Bjerksund_Stensland_2002_am_call_value(self.q, -self.b)
            self.strike, self.spot = self.prod.strike, self.strike
        return price

    # pylint: disable=invalid-name, too-many-locals, missing-docstring
    def _Bjerksund_Stensland_2002_am_call_value(self, r, b):
        """
        Bjerksund and Stensland(2002)近似解 - 美式看涨期权
        Args:
            r: 无风险利率
            b: r-q，无风险利率-分红率
        Returns: 美式期权现值
        """

        def B_inf():
            return self.beta / (self.beta - 1) * self.strike

        def B0():
            return np.maximum(self.strike, r / (r - b) * self.strike)

        def M(a, b, rho):  # 标准二元分布函数的累积分布函数
            mean = np.array([0, 0])
            cov = np.array([[1, rho], [rho, 1]])
            return multivariate_normal(mean, cov).cdf([a, b])

        def hold_time(t):
            return -(b * t + 2 * self.vol * math.sqrt(t)) * (self.strike ** 2 / ((B_inf() - B0()) * B0()))

        def barrier_price(t):
            return B0() + (B_inf() - B0()) * (1 - np.exp(hold_time(t)))

        def alpha(x):
            return (x - self.strike) * x ** -self.beta

        def phi(S, T, gamma, H, X):
            d1 = -(np.log(S / H) + (b + (gamma - 0.5) * (self.vol ** 2)) * T) / (self.vol * np.sqrt(T))
            d2 = -(np.log(X ** 2 / (S * H)) + (b + (gamma - 0.5) * (self.vol ** 2)) * T) / (self.vol * np.sqrt(T))
            lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * (self.vol ** 2)
            kappa = (2 * b) / (self.vol ** 2) + (2 * gamma - 1)
            return np.exp(lamb * T) * (S ** gamma) * (norm.cdf(d1) - ((X / S) ** kappa) * norm.cdf(d2))

        def psi(S, big_T, gamma, H, big_X, small_X, small_T):
            lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * (self.vol ** 2)
            kappa = 2 * b / (self.vol ** 2) + (2 * gamma - 1)

            small_d1 = -(np.log(S / small_X) + (b + (gamma - 0.5) * (self.vol ** 2)) * small_T) / (
                    self.vol * np.sqrt(small_T))
            small_d2 = -(np.log((big_X ** 2) / (S * small_X)) + (b + (gamma - 0.5) * (self.vol ** 2)) * small_T) / (
                    self.vol * np.sqrt(small_T))
            small_d3 = -(np.log(S / small_X) - (b + (gamma - 0.5) * (self.vol ** 2)) * small_T) / (
                    self.vol * np.sqrt(small_T))
            small_d4 = -(np.log((big_X ** 2) / (S * small_X)) - (b + (gamma - 0.5) * (self.vol ** 2)) * small_T) / (
                    self.vol * np.sqrt(small_T))
            big_d1 = -(np.log(S / H) + (b + (gamma - 0.5) * (self.vol ** 2)) * big_T) / (self.vol * np.sqrt(big_T))
            big_d2 = -(np.log((big_X ** 2) / (S * H)) + (b + (gamma - 0.5) * (self.vol ** 2)) * big_T) / (
                    self.vol * np.sqrt(big_T))
            big_d3 = -(np.log((small_X ** 2) / (S * H)) + (b + (gamma - 0.5) * (self.vol ** 2)) * big_T) / (
                    self.vol * np.sqrt(big_T))
            big_d4 = -(np.log((S * (small_X ** 2) / (H * (big_X ** 2)))) + (
                    b + (gamma - 0.5) * (self.vol ** 2)) * big_T) / (self.vol * np.sqrt(big_T))

            return np.exp(lamb * big_T) * (S ** gamma) * (M(small_d1, big_d1, np.sqrt(small_T / big_T))
                                                          - (((big_X / S) ** kappa) * M(small_d2, big_d2,
                                                                                        np.sqrt(small_T / big_T)))
                                                          - ((small_X / S) ** kappa) * M(small_d3, big_d3,
                                                                                         -np.sqrt(small_T / big_T))
                                                          + ((small_X / big_X) ** kappa) * M(small_d4, big_d4,
                                                                                             -np.sqrt(small_T / big_T)))

        self.beta = (0.5 - b / self.vol ** 2) + math.sqrt((b / self.vol ** 2 - 0.5) ** 2 + 2 * r / self.vol ** 2)

        t_1 = 0.5 * (math.sqrt(5) - 1) * self.T
        barrier_1 = barrier_price(self.T - t_1)
        barrier_2 = barrier_price(self.T)

        if self.spot > barrier_2:
            return self.spot - self.strike
        # else:
        div1 = (alpha(barrier_2) * self.spot ** self.beta
                - alpha(barrier_2) * phi(self.spot, t_1, self.beta, barrier_2, barrier_2))
        option1 = (phi(self.spot, t_1, 1.0, barrier_2, barrier_2)
                   - phi(self.spot, t_1, 1.0, barrier_1, barrier_2)
                   - self.strike * phi(self.spot, t_1, 0.0, barrier_2, barrier_2)
                   + self.strike * phi(self.spot, t_1, 0.0, barrier_1, barrier_2))
        div2 = (alpha(barrier_1) * phi(self.spot, t_1, self.beta, barrier_1, barrier_2)
                - alpha(barrier_1) * psi(self.spot, self.T, self.beta, barrier_1, barrier_2, barrier_1, t_1))
        option2 = (psi(self.spot, self.T, 1.0, barrier_1, barrier_2, barrier_1, t_1)
                   - psi(self.spot, self.T, 1.0, self.strike, barrier_2, barrier_1, t_1)
                   - self.strike * psi(self.spot, self.T, 0.0, barrier_1, barrier_2, barrier_1, t_1)
                   + self.strike * psi(self.spot, self.T, 0.0, self.strike, barrier_2, barrier_1, t_1))
        return div1 + option1 + div2 + option2
