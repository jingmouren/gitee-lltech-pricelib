#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta, abstractmethod


class StochProcessBase(metaclass=ABCMeta):
    """动态过程抽象类"""

    @abstractmethod
    def evolve(self, t, x, dt, dw):
        """随机微分方程离散化，递推一步dt的演化函数
        Args:
            t: float，当前时间
            x: np.ndarray，当前时间的价格向量
            dt: float，时间增量
            dw: np.ndarray，布朗运动增量，这里只是标准正态分布随机数，函数内部会 * sqrt(dt)
        Returns:
        """

    @abstractmethod
    def drift(self, *args, **kwargs):
        """随机微分方程漂移项dt系数"""

    @abstractmethod
    def diffusion(self, *args, **kwargs):
        """随机微分方程扩散项dW系数"""

    @abstractmethod
    def get_fn_pde_coef(self, *args, **kwargs):
        """返回PDE系数组件abc的函数，设pde有限差分的价格向量索引为i_vec，
            a是需要乘以i_vec^2的系数，b是需要乘以i_vec的系数，c是不需要乘以i_vec的系数"""
