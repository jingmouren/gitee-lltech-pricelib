#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.stats import norm
from ..processes import StochProcessBase
from ..utilities.enums import QuadMethod, ProcessType, EngineType
from .engine_base import PricingEngineBase


@lru_cache(maxsize=1)
def get_quad_vector(n_points, quad_method):
    """生成积分权重向量
    Args:
        n_points: int，积分点个数
        quad_method: 数值积分方式，QuadMethod枚举类，Trapezoid梯形法则/Simpson辛普森法则
    Returns:
        quad_vector: np.ndarray，积分权重向量
        quad_index: int，数值积分方法对应的调整系数
    """
    quad_vector = np.ones(n_points)
    if quad_method == QuadMethod.Trapezoid:
        quad_vector[1:-1] *= 2
        quad_index = 2
    else:  # quad_method == QuadMethod.Simpson:
        quad_vector[1::2] *= 4
        quad_vector[0::2] *= 2
        quad_vector[0] = quad_vector[-1] = 1
        quad_index = 3
    return quad_vector, quad_index


@njit(cache=True, fastmath=True, parallel=True)
def step_backward_jit(x, y, v, t, r, q, vol, quad_vector, quad_index):
    """向前递推一步，更新期权价值向量
    Args:
        x: np.ndarray，需要计算的未知时点，对应的标的价格向量
        y: np.ndarray，已知时点，对应的标的价格向量
        v: np.ndarray，已知时点，对应的期权价值向量
        t: float，时间间隔
        r: float，无风险利率
        q: float，分红率
        vol: float，波动率
        quad_vector: np.ndarray，积分权重向量
        quad_index: int，数值积分方法对应的调整系数
    Returns:
        target_v: np.ndarray(HashableArray)，更新后的期权价值向量
    """
    upper_barrier = np.max(y)
    lower_barrier = np.min(y)

    tvar = 0.5 * vol ** 2 * t
    rho = 1 / (2 * np.sqrt(np.pi * tvar) * y)
    rho = rho * np.exp(-1 / (4 * tvar) * (np.log(y / x.reshape(-1, 1)) - (r - q) * t + tvar) ** 2)
    target_v = np.dot(quad_vector, (rho * v).T) * (upper_barrier - lower_barrier) / (y.shape[0] - 1) / quad_index
    target_v *= np.exp(-r * t)
    return target_v


class QuadEngine(PricingEngineBase, metaclass=ABCMeta):
    """数值积分法定价引擎，目前只支持常数r、q、vol"""
    engine_type = EngineType.QuadEngine

    def __init__(self, stoch_process: StochProcessBase = None, quad_method=QuadMethod.Simpson, n_points=1001, n_max=4,
                 *,
                 s=None, r=None, q=None, vol=None):
        """初始化数值积分法定价引擎
        Args:
            stoch_process: StochProcessBase，随机过程
            quad_method: 数值积分方法，QuadMethod枚举类，Trapezoid梯形法则/Simpson辛普森法则
            n_points: int，积分点个数 (需要注意，辛普森法则的n_points必须是奇数)
            n_max: int，价格网格上界，设为n_smax倍初始价格
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.n_points = n_points
        self.quad_method = quad_method
        self.n_max = n_max
        # 以下为计算过程的中间变量
        self.backward_steps = 1
        self.r = None
        self.q = None
        self.vol = None

    def set_stoch_process(self, stoch_process: StochProcessBase):
        assert stoch_process.process_type == ProcessType.BSProcess1D, 'Error: 数值积分方法只能使用1维BSM动态过程'
        self.process = stoch_process

    def _check_method_params(self):
        """n_points数检查：设置价格格点数下限，检查格点数是否是奇数"""
        self.n_points = max(self.n_points, 100 * self.n_max)
        if self.n_points % 2 == 0:
            self.n_points += 1

    def set_quad_params(self, r=None, q=None, vol=None):
        """设置常数r、q、vol"""
        self.r = r
        self.q = q
        self.vol = vol

    def Vma(self, s, km, epsilon, t):
        """ Vma = 欧式香草期权中的资产或无的价值
        例如看涨期权，到期时刻，St > strike, Vma = St; St < strike, Vma = 0
        Args:
            s: 期初价格
            km: 行权价
            epsilon: 1 or -1，看涨期权为1，看跌期权为-1
            t: 年化到期时间
        Returns: Vma = 欧式香草期权中的资产或无的部分
        """
        tau = 0.5 * self.vol ** 2 * t
        d1 = (np.log(s / km) + (self.r - self.q + 0.5 * self.vol ** 2) * t) / (self.vol * np.sqrt(t))
        return np.exp((-2 * self.q * tau) / (self.vol ** 2)) * s * norm.cdf(epsilon * d1)

    def Vmb(self, s, km, epsilon, t):
        """ Vmb = 欧式香草期权中的现金或无的价值
        例如看涨期权，到期时刻，St > strike, Vmb = 1 ; St < strike, Vmb = 0
        Args:
            s: 期初价格
            km: 行权价
            epsilon: 1 or -1，看涨期权为1，看跌期权为-1
            t: 年化到期时间
        Returns: Vmb = 欧式香草期权中的现金或无的部分
        """
        tau = 0.5 * self.vol ** 2 * t
        d1 = (np.log(s / km) + (self.r - self.q + 0.5 * self.vol ** 2) * t) / (self.vol * np.sqrt(t))
        d2 = d1 - self.vol * np.sqrt(t)
        return np.exp((-2 * self.r * tau) / (self.vol ** 2)) * norm.cdf(epsilon * d2)

    def backward_samples(self, x, y, v, t, total_samples, n_samples):
        """非均匀价格格点下，稀疏格点向密集格点递推时的采样后递推方法，Universal option valuation using quadrature methods
        这里的非均匀指的是不同时点之间的格点不一致，但是同一时点的格点是均匀的；
        当密集格点向稀疏格点递推时，无需使用本方法采样，可以直接递推
        Args:
            x: np.ndarray，需要计算的未知时点，对应的标的价格向量
            y: np.ndarray，已知时点，对应的标的价格向量
            v: np.ndarray，已知时点，对应的期权价值向量
            t: float，时间间隔
            total_samples: int，待计算的密集的价格格点数
            n_samples: int, 采样的稀疏价格格点数
        Returns:
            target_v: np.ndarray，更新后的期权价值向量
        """
        indices = np.random.choice(total_samples, n_samples)
        indices = np.sort(indices)

        y_samples = y[indices]
        v_samples = v[indices]
        quad_vector, quad_index = get_quad_vector(n_samples, self.quad_method)
        v_target = step_backward_jit(x, y_samples, v_samples, t, self.r, self.q, self.vol, quad_vector, quad_index)
        v_target = np.array(v_target)
        v_target = np.minimum(v_target, 1e260)  # 防止价格极大和极小时，期权价值趋近于无穷大
        v_target = np.maximum(v_target, -1e260)
        return v_target

    def step_backward(self, x, y, v, t):
        """向前递推一步，更新期权价值向量
        Args:
            x: np.ndarray，需要计算的未知时点，对应的标的价格向量
            y: np.ndarray，已知时点，对应的标的价格向量
            v: np.ndarray，已知时点，对应的期权价值向量
            t: float，时间间隔
        Returns:
            result: np.ndarray，更新后的期权价值向量
        """
        quad_vector, quad_index = get_quad_vector(y.size, self.quad_method)  # y.size == y.shape[0]
        result = step_backward_jit(x, y, v, t, self.r, self.q, self.vol, quad_vector, quad_index)
        result = np.array(result)
        return result
