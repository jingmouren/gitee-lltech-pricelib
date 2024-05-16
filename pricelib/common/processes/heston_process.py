#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from numba import njit
from ..utilities.enums import ProcessType
from ..utilities.patterns import Observable, Observer
from .stoch_process import StochProcessBase


@njit(parallel=True, fastmath=True, cache=False)
def evolve_jit(x0, dt, dw, var_kappa, var_theta, var_vol, drift):
    """Heston SDE演化函数，jit加速
    Heston方差非负修正 - 采用部分截断模式
    Args:
        x0: np.ndarray, [s0, v0]是上一时刻的标的价格和方差, 即s0 = x0[0], v0 = x0[1]
        dt: float, 年化时间增量
        dw: np.ndarray, [dw_s, dw_v]是s和v的布朗运动增量，这里只是标准正态分布随机数，函数内部会 * sqrt(dt)
        var_kappa: float，Heston参数，方差回归速度
        var_theta: float，Heston参数，方差均值
        var_vol: float，Heston参数，方差的波动率
        drift: float，s的漂移率
    Returns:
        [s1, v1], t0+dt时刻的标的价格和方差np.ndarray
    """
    v1 = x0[1] + var_kappa * (var_theta - x0[1]) * dt + var_vol * np.sqrt(
        np.maximum(x0[1], 0) * dt) * dw[1]
    s1 = x0[0] * (1 + drift * dt + dw[0] * np.sqrt(np.maximum(x0[1], 0) * dt))
    return [s1, v1]


class HestonProcess(StochProcessBase, Observer, Observable):
    """Heston SDE:
        dS/S = (r-q)dt + sqrt(v) dW
        dv = kappa(theta - v)dt + var_vol * sqrt(v) dZ
        E[dW dZ] = rho dt
    既是观察者又是被观察者，观察者模式的中介者。观察S、r、q、heston参数的变化，被定价引擎engine观察。
    """
    process_type = ProcessType.Heston

    def __init__(self, spot: Observable, interest: Observable, div: Observable, *,
                 v0: float, var_theta: float, var_kappa: float, var_vol: float, var_rho: float):
        """初始化HestonProcess实例
        Args:
            spot: 市场行情实例，标的价格
            interest: 市场行情实例，无风险利率
            div: 市场行情实例，分红融券率
            v0: float, 方差初始值
            var_theta: float, 方差均值
            var_kappa: float, 方差均值回归速率
            var_vol: float, 方差的波动率
            var_rho: float, 方差与标的资产布朗运动的相关系数
        """
        self.spot = spot  # 标的价格
        self.interest = interest  # 无风险利率
        self.div = div  # 分红融券率
        spot.add_observer(self)  # 将本实例加入spot观察者列表
        interest.add_observer(self)  # 将本实例加入interest观察者列表
        div.add_observer(self)  # 将本实例加入div观察者列表
        self._v0 = v0  # 方差初始值
        self._var_kappa = var_kappa  # 回归速率
        self._var_theta = var_theta  # 方差均值
        self._var_vol = var_vol  # 方差的波动率
        self._var_rho = var_rho  # 方差与标的资产布朗运动的相关系数
        super().__init__()  # 调用被观察者Observable的构造函数

    def remove_self(self):
        """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
        self.spot.remove_observer(self)
        self.interest.remove_observer(self)
        self.div.remove_observer(self)
        del self

    def update(self, observable, *args, **kwargs):
        """被观察者的值发生变化时，自动调用update方法，通知观察者"""
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    def __call__(self):
        return ProcessType.Heston

    # @property
    def drift(self, t):
        """Heston过程价格s的漂移项
        Args:
            t: float，距离起始日的年化时间
        Returns:
            drift: float = r - q，r是无风险利率，q是分红融券率
        """
        return self.interest(t) - self.div(t)

    def diffusion(self):
        """heston的扩散项是sqrt(v), 此处方差v是evolve方法的参数，从外界传入，todo: process暂时未存储s和v路径。"""

    def evolve(self, t, x, dt, dw):
        """演化函数，根据Heston SDE进行演化
        Args:
            t: float，距离起始日的年化时间
            x: np.ndarray, [s0, v0]是上一时刻的标的价格和方差, 即s0 = x0[0], v0 = x0[1]
            dt: float, 年化时间增量
            dw: np.ndarray, [dw_s, dw_v]是s和v的布朗运动增量，这里只是标准正态分布随机数，函数内部会 * sqrt(dt)
        Returns:
            [s1, v1], t0+dt时刻的标的价格和方差np.ndarray
        """
        return evolve_jit(np.array(x), dt, np.array(dw), self.var_kappa, self.var_theta, self.var_vol, self.drift(t))

    def get_fn_pde_coef(self):
        raise NotImplementedError("TODO: Heston PDE数值解法 - ADI 交替隐式迭代法")

    @property
    def v0(self):
        return self._v0

    @v0.setter
    def v0(self, new_value):
        """覆写给对象的属性赋值的方法，当属性值变化时，自动向观察者发送通知"""
        self._v0 = float(new_value)  # 属性变化
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    @property
    def var_kappa(self):
        return self._var_kappa

    @var_kappa.setter
    def var_kappa(self, new_value):
        """覆写给对象的属性赋值的方法，当属性值变化时，自动向观察者发送通知"""
        self._var_kappa = float(new_value)  # 属性变化
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    @property
    def var_theta(self):
        return self._var_theta

    @var_theta.setter
    def var_theta(self, new_value):
        """覆写给对象的属性赋值的方法，当属性值变化时，自动向观察者发送通知"""
        self._var_theta = float(new_value)  # 属性变化
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    @property
    def var_vol(self):
        return self._var_vol

    @var_vol.setter
    def var_vol(self, new_value):
        """覆写给对象的属性赋值的方法，当属性值变化时，自动向观察者发送通知"""
        self._var_vol = float(new_value)  # 属性变化
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    @property
    def var_rho(self):
        return self._var_rho

    @var_rho.setter
    def var_rho(self, new_value):
        """覆写给对象的属性赋值的方法，当属性值变化时，自动向观察者发送通知"""
        self._var_rho = float(new_value)  # 属性变化
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变
