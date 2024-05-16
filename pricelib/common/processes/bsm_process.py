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


@njit(parallel=True, fastmath=True, cache=True)
def evolve_bs(t0, x0, dt, dw, drift, diffusion):
    """Black-Scholes-Merton SDE演化函数，jit加速
    Black-Scholes-Merton SDE: dS/S = drift * dt + diffusion * dW
    Args:
        t0: float, 距离起始日的年化时间
        x0: np.ndarray, t0时刻的标的价格向量
        dt: float, 年化时间增量
        dw: np.ndarray, 布朗运动增量，这里只是标准正态分布随机数，函数内部会 * sqrt(dt)
        drift: float, BSM的漂移率
        diffusion: np.ndarray, BSM的波动率
    Returns:
        np.ndarray, t0+dt时刻的标的价格向量
    """
    return x0 * (1 + drift * dt + diffusion * dw * np.sqrt(dt))


class GeneralizedBSMProcess(StochProcessBase, Observer, Observable):
    """BS风险中性 SDE: dS/S = (r-q)dt + vol dW
    既是观察者又是被观察者，观察者模式的中介者。观察S、r、q、sigma的变化，被定价引擎engine观察。
    """
    process_type = ProcessType.BSProcess1D

    def __init__(self, spot: Observable, interest: Observable, div: Observable, vol: Observable):
        """标的价格、无风险利率、分红融券率、波动率，都是被观察者。
        vol可以是常数波动率，也可以是局部波动率，可以由vol(t,S)取值"""
        self.spot = spot
        self.interest = interest
        self.div = div
        self.vol = vol
        spot.add_observer(self)  # 将本实例加入spot观察者列表
        interest.add_observer(self)  # 将本实例加入interest观察者列表
        div.add_observer(self)  # 将本实例加入div观察者列表
        vol.add_observer(self)  # 将本实例加入vol观察者列表
        super().__init__()  # 调用被观察者Observable的构造函数

    def remove_self(self):
        """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
        self.spot.remove_observer(self)
        self.interest.remove_observer(self)
        self.div.remove_observer(self)
        self.vol.remove_observer(self)
        del self

    def update(self, observable, *args, **kwargs):
        """被观察者的值发生变化时，自动调用update方法，通知观察者"""
        self.notify_observers()  # 自动向观察者发送通知, 参数已改变

    def __call__(self):
        return ProcessType.BSProcess1D

    def drift(self, t):
        """BSM过程的漂移项
        Args:
            t: float，距离起始日的年化时间
        Returns:
            drift: float = r - q，r是无风险利率，q是分红融券率
        """
        return self.interest(t) - self.div(t)

    def diffusion(self, t, spot):
        """BSM过程的扩散项
        Args:
            t: float，距离起始日的年化时间
            spot: np.ndarray，标的价格向量
        Returns:
            diffusion: np.ndarray = vol，vol是波动率
        """
        return self.vol(t, spot)

    def evolve(self, t, x, dt, dw):
        """演化函数，根据BSM SDE进行演化
        Args:
            t: float，距离起始日的年化时间
            x: np.ndarray, t0时刻的标的价格向量
            dt: float, 年化时间增量
            dw: np.ndarray, 布朗运动增量，这里只是标准正态分布随机数，函数内部会 * sqrt(dt)
        Returns:
            np.ndarray, t0+dt时刻的标的价格向量
        """
        return evolve_bs(t, x, dt, dw, self.drift(t), self.diffusion(t, x))

    def get_fn_pde_coef(self, maturity, spot):
        """获取返回pde系数的函数
        Args:
            maturity: float，年化到期时间
            spot: float，标的价格
        Returns:
            fn_pde_coef: function, 返回PDE系数组件abc的函数
        """

        # TODO: 目前局部波动率使用K坐标，以后可能重构为log moneyness坐标
        # def fn_pde_coef(u, z):  # z在校准中是执行价K，在定价中是标的价S_u, u是距离起始日的年化时间
        #     """返回PDE系数组件abc的函数，设pde有限差分的价格向量索引为i_vec，
        #     a是需要乘以i_vec^2的系数，b是需要乘以i_vec的系数，c是不需要乘以i_vec的系数"""
        #     r_u = self.interest(u)  # u时刻的无风险利率r
        #     q_u = self.div(u)  # u时刻的分红融券率q
        #     fwd_u = spot * math.exp((maturity - u) * (r_u - q_u))  # u时刻的远期价格
        #     lv = self.vol(u, np.log(z / fwd_u))  # u时刻，log(K/F)对数moneyness的local_vol
        #     a, b, c = lv ** 2, r_u - q_u, r_u
        #     return a, b, c
        def fn_pde_coef(t, spot):  # z在校准中是执行价K，在定价中是标的价S_u, u是距离起始日的年化时间
            """返回PDE系数组件abc的函数
            Args:
                t: float，年化到期时间
                spot: float，标的价格
            Returns: 设pde有限差分的价格向量索引为i_vec
                a: 需要乘以i_vec^2的系数
                b: 需要乘以i_vec的系数
                c: 不需要乘以i_vec的系数
            """
            r_t = self.interest(t)  # t时刻的无风险利率r
            q_t = self.div(t)  # t时刻的分红融券率q
            lv = self.vol(t, spot)  # t时刻，spot的local_vol
            a, b, c = lv ** 2, r_t - q_t, r_t
            return a, b, c

        return fn_pde_coef
