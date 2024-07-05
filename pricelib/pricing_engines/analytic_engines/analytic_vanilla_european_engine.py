#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from scipy.stats import norm
from pricelib.common.processes import StochProcessBase
from pricelib.common.utilities.enums import ExerciseType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine


def bs_d1(S, K, T, r, sigma, q=0):
    """欧式股票期权BSM公式中的d1"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1


def bs_formula(S, K, T, r, sigma, sign=1, q=0):
    """欧式股票期权BSM公式, 针对现货（股票）期权的定价模型。认购期权的sign为1, 认沽期权的sign为-1
    若股票有稳定的连续分红率q，请传入参数q。股票在0时刻价格 = S(0)exp(qT)，股票在T时间中的分红 = S(0) * [exp(qT) - 1]"""
    if T == 0:  # 如果估值日就是到期日，直接返回payoff
        return max(sign * (S - K), 0)
    d1 = bs_d1(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return sign * np.exp(-q * T) * S * norm.cdf(sign * d1) - sign * np.exp(-r * T) * K * norm.cdf(sign * d2)


class AnalyticVanillaEuEngine(AnalyticEngine):
    """欧式期权BSM解析解定价引擎"""

    def __init__(self, stoch_process: StochProcessBase = None, *, s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: 随机过程StochProcessBase对象
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        # 以下属性指向需要定价的产品，由calc_present_value方法设置
        self.prod = None

    def d1(self, tau, spot):
        """BSM公式d1"""
        d1_value = bs_d1(S=spot, K=self.prod.strike, T=tau, r=self.process.interest(tau), q=self.process.div(tau),
                         sigma=self.process.diffusion(tau, spot))
        return d1_value

    def d2(self, tau, spot):
        """BSM公式d2, d2 = d1 - sigma * sqrt(t)"""
        d2_value = self.d1(tau, spot) - self.process.diffusion(tau, spot) * np.sqrt(tau)
        return d2_value

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert prod.exercise_type == ExerciseType.European, "Error: 欧式期权解析解只支持European行权类型."
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        tau = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()

        price = bs_formula(S=spot, K=prod.strike, T=tau, sign=prod.callput.value, r=self.process.interest(tau),
                           q=self.process.div(tau), sigma=self.process.diffusion(tau, spot))
        return price

    def delta(self, prod, spot=None, *args, **kwargs):
        """∂V/∂S"""
        self.prod = prod
        tau = (prod.end_date - global_evaluation_date()).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        q = self.process.div(tau)
        delta = self.prod.callput.value * math.exp(-q * tau) * norm.cdf(self.prod.callput.value * self.d1(tau, spot))
        return delta

    def gamma(self, prod, spot=None, *args, **kwargs):
        """∂2V/∂S2"""
        self.prod = prod
        tau = (prod.end_date - global_evaluation_date()).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        gamma = math.exp(-q * tau) * norm.pdf(self.d1(tau, spot)) / (spot * vol * np.sqrt(tau))
        return gamma

    def vega(self, prod, spot=None, *args, **kwargs):
        """∂V/∂σ"""
        self.prod = prod
        tau = (prod.end_date - global_evaluation_date()).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        q = self.process.div(tau)
        vega = math.exp(-q * tau) * spot * norm.pdf(self.d1(tau, spot)) * np.sqrt(tau)
        # 业内习惯把Vega理解为波动率变动1个百分点对应期权价值的变化量，所以根据BS公式计算出的vega还需再除以100
        return vega * 0.01

    def rho(self, prod, spot=None, *args, **kwargs):
        """∂V/∂r"""
        self.prod = prod
        tau = (prod.end_date - global_evaluation_date()).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        rho = self.prod.callput.value * self.prod.strike * tau * np.exp(-r * tau) * norm.cdf(
            self.prod.callput.value * self.d2(tau, spot))
        # 业内习惯把Rho理解为波动率变动1个百分点对应期权价值的变化量，所以根据BS公式计算出的rho还需再除以100
        return rho * 0.01

    def theta(self, prod, spot=None, *args, **kwargs):
        """∂V/∂t
        注意，这里B-S公式的theta计算结果在周五或节假日的前一天计算时，与差分法有所不同。因为差分法是交易日+1天，这里是自然日+1天。"""
        self.prod = prod
        tau = (prod.end_date - global_evaluation_date()).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        sign = self.prod.callput.value
        theta = (-1 * spot * math.exp(-q * tau) * norm.pdf(self.d1(tau, spot)) * vol / (2 * np.sqrt(tau))
                 - sign * r * self.prod.strike * np.exp(-r * tau) * norm.cdf(sign * self.d2(tau, spot))
                 + sign * q * spot * np.exp(-q * tau) * norm.cdf(sign * self.d1(tau, spot)))
        # 业内习惯Theta用来衡量每日的time decay，而BS Model中的时间单位是年，所以按此公式算出来的Theta需要再除以365
        return theta / prod.annual_days.value
