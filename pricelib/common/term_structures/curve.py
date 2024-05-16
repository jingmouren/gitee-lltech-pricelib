#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from functools import lru_cache
from enum import Enum, unique
import numpy as np
from scipy.interpolate import interp1d
from ..utilities.patterns import Observable, SimpleQuote


class ConstantRate(SimpleQuote):
    """无风险利率/分红率常数类"""

    def disc_factor(self, t2, t1=0.):
        """返回指定时间t2对应的折现因子，或者t1到t2的折现因子(t1 < t2)
        Args:
            t1: 可以是float或numpy.ndarray, 默认为0。
            t2: 可以是float或numpy.ndarray, 但是t1和t2都是array时，维数必须一致
        Returns:
            float, 折现因子
        """
        return np.exp(-self(t2) * t2 + self(t1) * t1)


class RateTermStructure(Observable):
    """无风险利率/分红融券率曲线，线性插值"""

    def __init__(self, t0, fn, name=None):
        super().__init__()
        self.t0 = t0  # rate的初始时间
        self.__fn_rate = fn  # rate关于时间的函数
        self.__name = name

    @staticmethod
    def _interp_fn(tenors, rates):
        """返回曲线的线性插值函数
        Args:
            tenors: 期限数组
            rates: 无风险利率/分红融券率数组
        Returns:
            function，曲线的线性插值函数
        """
        assert len(tenors) == len(rates), "Error: 输入曲线的t与value数组长度不一致"
        linear_rate = interp1d(tenors, rates, kind='linear', bounds_error=False, fill_value=(rates[0], rates[-1]))
        return linear_rate

    def update_fn_rate(self, tenors, rates):
        """更新rate关于时间的函数，自动通知观察者"""
        self.__fn_rate = self._interp_fn(tenors, rates)
        self.notify_observers(name=self.__name)  # 自动向观察者发送通知

    @classmethod
    def from_array(cls, tenors, rates):
        """新建一个 RateTermStructure 对象 - 方法1: 用数组新建
        tenors是时间点数组, divs是分红率数组"""
        linear_rate = cls._interp_fn(tenors, rates)
        return cls(tenors[0], lambda t: linear_rate(t))

    @classmethod
    def from_fn(cls, t0, fn):
        """用函数新建一个 RateTermStructure 对象 - 方法2: 直接用函数新建
        t0是起始日，fn是rate关于时间的函数"""
        return cls(t0, fn)

    @lru_cache(maxsize=None)
    def __call__(self, t):
        """返回指定时间t对应的rate，t可以是float或numpy.arrays"""
        return self.__fn_rate(t)

    def disc_factor(self, t2, t1=0.):
        """返回指定时间t2对应的折现因子，或者t1到t2的折现因子
        t1可以是float或numpy.ndarray, 默认为0。
        t2可以是float或numpy.ndarray, 但是t1和t2都是array时，维数必须一致。"""
        return np.exp(-self(t2) * t2 + self(t1) * t1)

    @property
    def name(self):
        return self.__name


@unique
class InterpolationMode(Enum):
    LinearZero = 0  # 初始时刻t0是0，discs为折现到t0的折现因子，取对数除以时间后，对r线性插值，时间是距离起始时间t0的时长，两侧水平外推
    LinearLogDiscount = 1  # 折现因子为exp(-rt)，取对数，对-rt线性插值，时间是年化时间点，两侧线性外推


class DiscountCurve:
    """
    折现因子曲线。假设即期利率曲线分段线性，即每个标准期限即期利率线性相连。
    利率互换即期利率曲线为SC(t)，贴现函数曲线为DF(t)，满足DF(t)=exp^{-SC(t)*t}
    每期的贴现现金流应该等于初始的本金100，应有 100 = PV(k) = 利息部分 + 本金部分
        利息部分 = 100 * SC(k) * sum_{i=1}^k [T(i) - T(i-1)] * DF(i)
        本金部分 = 100 x DF(k)
        其中i为第i个付息期
    """

    def __init__(self, t0, fn):
        self.t0 = t0  # 初始时间
        self.__discount_factor = fn  # 折现因子插值函数

    @classmethod
    def from_array(cls, tenors, discs, interp_mode=InterpolationMode.LinearZero):
        """新建一个 DiscountCurve 对象 - 方法1: 用数组新建
        tenors是时间点数组, discs是折现因子数组, interp_mode是插值方式
            LinearZero 初始时刻t0是0，discs为折现到t0的折现因子，取对数除以时间后，对r线性插值，时间是距离起始时间t0的时长，两侧水平外推
            LinearLogDiscount 折现因子为exp(-rt)，取对数，对-rt线性插值，时间是年化时间点，两侧线性外推"""
        assert len(tenors) == len(discs), "Error: 输入曲线的t与value数组长度不一致"
        if interp_mode == InterpolationMode.LinearZero:
            y = -np.log(discs[1:]) / (tenors[1:] - tenors[0])
            linzero = interp1d(tenors[1:], y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
            return cls(tenors[0], lambda t: np.exp(-linzero(t) * t))
        if interp_mode == InterpolationMode.LinearLogDiscount:
            linlogdisc = interp1d(tenors, np.log(discs), kind='linear', bounds_error=False, fill_value='extrapolate')
            return cls(tenors[0], lambda t: np.exp(linlogdisc(t)))

        raise NotImplementedError(f"不支持的插值方式{interp_mode}")

    @classmethod
    def from_fn(cls, t0, fn):
        """用函数新建一个 DiscountCurve 对象 - 方法2: 直接用函数新建
        t0是起始日，fn是折现因子函数"""
        return cls(t0, fn)

    @lru_cache(maxsize=None)
    def __call__(self, t, dt=3e-3):
        """经过差分计算，泰勒展开e^x-1=x，得到的t处利率rt
        其中self(t)：t处的折现因子exp(-rt), dt=3e-3约等于1/365，即一个自然日
        """
        return (self(t - dt) / self(t + dt) - 1) / dt / 2

    def disc_factor(self, t2, t1=0.):
        """返回指定时间t2对应的折现因子，或者t1到t2的折现因子
        t1可以是float或numpy.ndarray, 默认为0。
        t2可以是float或numpy.ndarray, 但是t1和t2都是array时，维数必须一致。"""
        return self.__discount_factor(t2) / self.__discount_factor(t1)


if __name__ == "__main__":
    div = RateTermStructure.from_array([0, 1 / 12, 2 / 12, ], [0.02, 0.15, 0.18])
    r = RateTermStructure.from_array([0, 1 / 12, 2 / 12], [0.015, 0.02, 0.025])
    tau = 0.5
    print(r(tau) - div(tau))
