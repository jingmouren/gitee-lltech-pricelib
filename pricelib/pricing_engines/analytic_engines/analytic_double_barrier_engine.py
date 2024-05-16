#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from scipy.stats import norm

from pricelib.common.utilities.enums import CallPut, InOut, UpDown
from pricelib.common.processes import StochProcessBase
from pricelib.common.pricing_engine_base import AnalyticEngine
from pricelib.common.time import global_evaluation_date
from ...products.vanilla.vanilla_option import VanillaOption
from ...products.barrier.barrier_option import BarrierOption
from .analytic_vanilla_european_engine import AnalyticVanillaEuEngine
from .analytic_barrier_engine import AnalyticBarrierEngine


class AnalyticDoubleBarrierEngine(AnalyticEngine):
    """双边障碍期权闭式解定价引擎
    Haug(1998) Put-Call Barrier Transformations 对于标的持有成本为0的期权(期货期权)，利用认购-认沽障碍期权对称性，
                                                利用单边障碍期权解析解的组合给双障碍期权定价。
    Ikeda and Kunitomo(1992)将双障碍期权表示为加权正态分布函数的无限集，但是仅当行权价在障碍范围内时，公式才是成立的。
    Haug(1998)的解析解与Ikeda and Kunitomo(1992)的结果几乎一致，两者对比，
        Ikeda and Kunitomo(1992)适用于标的持有成本不为0的情形, Haug(1998)适用于行权价在障碍范围之外的情形。
    """

    def __init__(self, stoch_process: StochProcessBase = None, formula_type="Ikeda&Kunitomo1992",
                 series_num=10, delta1=0, delta2=0, *,
                 s=None, r=None, q=None, vol=None):
        """
        Args:
            stoch_process: 随机过程StochProcessBase对象
            self.formula_type: 解析解类型，Ikeda&Kunitomo1992或Haug1998
        其他三个参数是Ikeda&Kunitomo1992的参数：
            self.series_num: 用于计算近似解的级数项 默认 i = -10 ~ 10
            self.delta1: 上边界的曲率
            self.delta2: 下边界的曲率，
                当delta1 = delta2 = 0时，对应两条平行边界
                当delta1 < 0 < delta2 时，对应下边界随时间呈指数增长，而上百年姐随时间呈指数衰减
                当delta1 > 0 > delta2 时，对应一个向下凸的下边界和一个向上凸的上边界
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.formula_type = formula_type
        self.series_num = series_num
        self.i_vec = np.array(range(- series_num, series_num + 1)).astype(float)  # 用于计算近似解的级数项
        self.delta1 = delta1
        self.delta2 = delta2
        # 以下是计算过程的中间变量
        self.prod = None  # Product产品对象

    # pylint: disable=invalid-name, too-many-locals
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)

        vanilla_option = VanillaOption(maturity=tau, strike=prod.strike, callput=prod.callput,
                                       engine=AnalyticVanillaEuEngine(self.process))

        if prod.discrete_obs_interval is not None:
            # 均匀离散观察，M. Broadie, P. Glasserman, S.G. Kou(1997) 在连续障碍期权解析解上加调整项，调整障碍价格水平
            # 指数上的beta = -zeta(1/2) / sqrt(2*pi) = 0.5826, 其中zeta是黎曼zeta函数
            bound = [None, None]
            bound[1] = prod.bound[1] * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            bound[0] = prod.bound[0] * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
        else:  # 连续观察
            bound = prod.bound

        if self.formula_type == "Ikeda&Kunitomo1992":
            assert (bound[0] < prod.strike < bound[1]), "Error: Ikeda and Kunitomo(1992)的解析解仅当行权价在障碍范围内时，公式才是成立的。"
            return self.Ikeda_Kunitomo1992_pv(spot, r, q, vol, tau, bound, vanilla_option)
        if self.formula_type == "Haug1998":
            assert (r - q == 0), "Error: Haug(1998)的解析解仅当标的持有成本为0时(期货期权，或r-q=0)，公式才是准确的。"
            return self.Haug1998_pv(tau, bound, vanilla_option)
        raise NotImplementedError(f"{self.formula_type}不支持的解析解类型, 仅支持Ikeda&Kunitomo1992和Haug1998")

    # pylint: disable=invalid-name, too-many-locals
    def Ikeda_Kunitomo1992_pv(self, spot, r, q, vol, tau, bound, vanilla_option):
        """Ikeda and Kunitomo(1992)级数解，仅当行权价在障碍范围内时，公式才是成立的"""
        if self.prod.callput == CallPut.Call:
            E = bound[1] * math.exp(self.delta1 * tau)
        else:
            E = bound[0] * math.exp(self.delta2 * tau)
        numerator = (r - q + 0.5 * vol ** 2) * tau
        denominator = vol * math.sqrt(tau)
        d1 = (np.log(
            spot * np.power(bound[1], 2 * self.i_vec) / (self.prod.strike * np.power(bound[0], 2 * self.i_vec)))
              + numerator) / denominator
        d2 = (np.log(spot * np.power(bound[1], 2 * self.i_vec) / (E * np.power(bound[0], 2 * self.i_vec)))
              + numerator) / denominator
        d3 = (np.log(
            np.power(bound[0], 2 * self.i_vec + 2) / (self.prod.strike * spot * np.power(bound[1], 2 * self.i_vec)))
              + numerator) / denominator
        d4 = (np.log(
            np.power(bound[0], 2 * self.i_vec + 2) / (E * spot * np.power(bound[1], 2 * self.i_vec)))
              + numerator) / denominator
        mu1 = 2 * (r - q - self.delta2 - self.i_vec * (self.delta1 - self.delta2)) / vol ** 2 + 1
        mu2 = 2 * self.i_vec * (self.delta1 - self.delta2) / vol ** 2
        mu3 = 2 * (r - q - self.delta2 + self.i_vec * (self.delta1 - self.delta2)) / vol ** 2 + 1

        Nd1a = (np.power(np.power(bound[1], self.i_vec) / np.power(bound[0], self.i_vec), mu1) * (
                bound[0] / spot) ** mu2 * (norm.cdf(d1) - norm.cdf(d2)) * self.prod.callput.value)
        Nd1b = (np.power(np.power(bound[0], self.i_vec + 1) / (np.power(bound[1], self.i_vec) * spot), mu3) * (
                norm.cdf(d3) - norm.cdf(d4)) * self.prod.callput.value)
        Nd2a = (np.power(np.power(bound[1], self.i_vec) / np.power(bound[0], self.i_vec), mu1 - 2) * (
                bound[0] / spot) ** mu2 * self.prod.callput.value * (
                        norm.cdf(d1 - denominator) - norm.cdf(d2 - denominator)))
        Nd2b = (np.power(np.power(bound[0], self.i_vec + 1) / (np.power(bound[1], self.i_vec) * spot), mu3 - 2) * (
                norm.cdf(d3 - denominator) - norm.cdf(d4 - denominator)) * self.prod.callput.value)

        double_out = (self.prod.callput.value * spot * math.exp(-q * tau) * np.sum(Nd1a - Nd1b) -
                      self.prod.callput.value * self.prod.strike * math.exp(-r * tau) * np.sum(Nd2a - Nd2b))
        if self.prod.inout == InOut.Out:
            return double_out * self.prod.parti
        # else: self.inout == InOut.In
        double_in = vanilla_option.price() - double_out
        return double_in * self.prod.parti

    # pylint: disable=too-many-locals
    def Haug1998_pv(self, tau, bound, vanilla_option):
        """Haug(1998)的解析解，仅当标的持有成本为0时(期货期权，或r-q=0)，公式才是准确的"""
        value = 0
        if self.prod.callput == CallPut.Call:
            opposite_callput = CallPut.Put
        else:
            opposite_callput = CallPut.Call

        up_barrier = BarrierOption(maturity=tau, strike=self.prod.strike, barrier=bound[1],
                                   updown=UpDown.Up, callput=self.prod.callput, inout=InOut.In, rebate=0,
                                   discrete_obs_interval=None,
                                   engine=AnalyticBarrierEngine(self.process))
        down_barrier = BarrierOption(maturity=tau, strike=self.prod.strike, barrier=bound[0],
                                     updown=UpDown.Down, callput=self.prod.callput, inout=InOut.In, rebate=0,
                                     discrete_obs_interval=None,
                                     engine=AnalyticBarrierEngine(self.process))
        up_barrier_opposite = BarrierOption(strike=bound[1] ** 2 / self.prod.strike, barrier=bound[1] ** 2 / bound[0],
                                            maturity=tau, rebate=0, inout=InOut.In,
                                            updown=UpDown.Up, callput=opposite_callput,
                                            discrete_obs_interval=None,
                                            engine=AnalyticBarrierEngine(self.process))
        down_barrier_opposite = BarrierOption(strike=bound[0] ** 2 / self.prod.strike, barrier=bound[0] ** 2 / bound[1],
                                              maturity=tau, rebate=0, inout=InOut.In,
                                              updown=UpDown.Down, callput=opposite_callput,
                                              discrete_obs_interval=None,
                                              engine=AnalyticBarrierEngine(self.process))
        value += (up_barrier.price() + down_barrier.price()

                  - up_barrier_opposite.price() * self.prod.strike / bound[1]
                  - down_barrier_opposite.price() * self.prod.strike / bound[0])

        up_barrier.strike = self.prod.strike * bound[1] ** 2 / bound[0] ** 2
        up_barrier.barrier = math.pow(bound[1], 3) / bound[0] ** 2
        down_barrier.strike = self.prod.strike * bound[0] ** 2 / bound[1] ** 2
        down_barrier.barrier = math.pow(bound[0], 3) / bound[1] ** 2
        up_barrier_opposite.strike = math.pow(bound[1], 4) / (bound[0] ** 2 * self.prod.strike)
        up_barrier_opposite.barrier = math.pow(bound[1], 4) / math.pow(bound[0], 3)
        down_barrier_opposite.strike = math.pow(bound[0], 4) / (bound[1] ** 2 * self.prod.strike)
        down_barrier_opposite.barrier = math.pow(bound[0], 4) / math.pow(bound[1], 3)

        value += (up_barrier.price() * bound[0] / bound[1] + down_barrier.price() * bound[1] / bound[0]
                  - up_barrier_opposite.price() * self.prod.strike * bound[0] / bound[1] ** 2
                  - down_barrier_opposite.price() * self.prod.strike * bound[1] / bound[0] ** 2)

        up_barrier.strike = self.prod.strike * (math.pow(bound[1], 4) / math.pow(bound[0], 4))
        up_barrier.barrier = math.pow(bound[1], 5) / math.pow(bound[0], 4)
        down_barrier.strike = self.prod.strike * (math.pow(bound[0], 4) / math.pow(bound[1], 4))
        down_barrier.barrier = math.pow(bound[0], 5) / math.pow(bound[1], 4)

        value += (up_barrier.price() * bound[0] ** 2 / bound[1] ** 2
                  + down_barrier.price() * bound[1] ** 2 / bound[0] ** 2)

        double_in = min(value, vanilla_option.price())
        if self.prod.inout == InOut.In:
            return double_in * self.prod.parti
        # else: self.inout == InOut.Out
        double_out = vanilla_option.price() - double_in
        return double_out * self.prod.parti
