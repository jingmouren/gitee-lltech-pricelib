#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta
import numpy as np
from scipy.stats import norm
from ..processes import StochProcessBase
from ..utilities.enums import ProcessType, EngineType
from .engine_base import PricingEngineBase


class BiTreeEngine(PricingEngineBase, metaclass=ABCMeta):
    """二叉树定价引擎"""
    engine_type = EngineType.TreeEngine

    def __init__(self, stoch_process: StochProcessBase = None, tree_branches=500, *,
                 s=None, r=None, q=None, vol=None):
        """初始化二叉树定价引擎
        Args:
            stoch_process: StochProcessBase，随机过程
            tree_branches: int，二叉树每年的分支数
       在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.tree_branches = tree_branches  # 二叉树每年的分支数
        self.dt = 1 / tree_branches  # 时间步长
        # 以下为计算过程的中间变量
        self.p = None  # 风险中性概率，在path_generator方法中计算生成
        self.s_paths = None  # CRR二叉树
        self.v = None  # 期权价值矩阵
        self.r = None  # 无风险利率
        # 以下变量仅当需要使用Vma、Vmb这两个解析解方法的时候才需要在子类中赋值
        self.q = None  # 贴水率
        self.vol = None  # 波动率

    def set_stoch_process(self, stoch_process: StochProcessBase):
        """设置随机过程"""
        assert stoch_process.process_type == ProcessType.BSProcess1D, 'Error: 二叉树方法只能使用1维BSM动态过程'
        self.process = stoch_process

    def path_generator(self, n_step, t=None, spot=None):
        """生成上三角结构的标的资产价格CRR二叉树
        Args:
            n_step: int，二叉树的分支数
            t: float，到期时间
            spot: float，期初价格
        Returns:
            self.s_paths: np.array，上三角结构的标的资产价格CRR二叉树
        """
        if spot is None:
            spot = self.process.spot()
        u = np.exp(self.process.diffusion(t, spot) * np.sqrt(self.dt))
        d = 1 / u
        self.p = (np.exp(self.process.drift(t) * self.dt) - d) / (u - d)  # 风险中性概率
        u_power = np.arange(n_step + 1)
        u_power_matrix = np.tile(u_power, (n_step + 1, 1))
        d_power_matrix = u_power_matrix.T
        u_matrix = u ** u_power_matrix
        d_matrix = d ** d_power_matrix
        self.s_paths = u_matrix * d_matrix
        for i in range(n_step + 1):
            self.s_paths[i] = np.roll(self.s_paths[i], i)
        self.s_paths *= spot
        self.s_paths = np.triu(self.s_paths, k=0)
        return self.s_paths

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
