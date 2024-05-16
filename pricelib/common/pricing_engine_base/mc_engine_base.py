#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta
from contextlib import suppress
import numpy as np
from scipy.stats import norm, qmc
from ..processes import StochProcessBase
from ..utilities.enums import RandsMethod, LdMethod, ProcessType, EngineType
from ..utilities.patterns import Observer
from ..utilities.utility import logging
from .engine_base import PricingEngineBase


class McEngine(PricingEngineBase, Observer, metaclass=ABCMeta):
    """蒙特卡洛模拟定价引擎基类
    观察者，观察随机过程process对象，当process对象的属性变化时，自动更新状态，会重新生成价格路径"""
    engine_type = EngineType.McEngine

    def __init__(self, stoch_process: StochProcessBase = None, n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                 antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0, *,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: 随机过程StochProcessBase对象
            n_path: int，MC模拟路径数
            rands_method: 生成随机数方法，RandsMethod枚举类，Pseudorandom伪随机数/LowDiscrepancy低差异序列
            antithetic_variate: bool，是否使用对立变量法
            ld_method: 若使用了低差异序列，指定低差异序列方法，LdMethod枚举类，Sobol序列/Halton序列
            seed: int，随机数种子
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        # MC模拟路径数
        if n_path % 2 != 0 and antithetic_variate:
            n_path += 1
        self.n_path = n_path
        self.rands_method = rands_method  # 生成随机数方法
        self._ld_method = ld_method  # 低差异序列方法
        self._antithetic_variate = antithetic_variate  # 是否使用对立变量
        self.seed = seed  # 随机数种子
        self.__reset_rands = True  # 重置随机数标志位
        self.__reset_paths = True  # 重置路径标志位
        self.rands = None  # 随机数矩阵
        self.s_paths = None  # 价格路径矩阵
        self.var_paths = None  # 方差路径矩阵

    def set_stoch_process(self, stoch_process):
        """设置随机过程，先将自己从原来的随机过程的观察者列表中移除，再将自己加入新的随机过程的观察者列表"""
        with suppress(AttributeError, ValueError):
            self.process.remove_observer(self)
        self.process = stoch_process
        stoch_process.add_observer(self)  # 将本实例加入stoch_process观察者列表
        self.rands = None  # 更换随机过程之后需要重新生成随机数
        self.__reset_rands = True  # 更换随机过程之后需要重新生成随机数
        self.__reset_paths = True  # 更换随机过程之后需要重新生成路径

    def remove_self(self):
        """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
        self.process.remove_observer(self)
        del self

    def update(self, observable, *args, **kwargs):
        """被观察者的值发生变化时，自动调用update方法，通知观察者"""
        self.__reset_paths = True  # process自动向观察者发送通知, 参数已改变, 重置路径标志位，重新生成路径

    def reset_paths_flag(self):
        """重置路径标志位，重新生成路径"""
        self.__reset_paths = True

    @property
    def antithetic_variate(self):
        return self._antithetic_variate

    @antithetic_variate.setter
    def antithetic_variate(self, new_value):
        self._antithetic_variate = new_value
        self.__reset_paths = True  # 重置路径标志位，重新生成路径
        self.__reset_rands = True  # 重置随机数标志位，重新生成随机数

    @property
    def ld_method(self):
        return self._ld_method

    @ld_method.setter
    def ld_method(self, new_value):
        self._ld_method = new_value
        self.__reset_paths = True  # 重置路径标志位，重新生成路径
        self.__reset_rands = True  # 重置随机数标志位，重新生成随机数

    def _randoms_generator(self, shape):  # 参数rands_method控制随机数获取方法，LD_method低差异序列生成方法；
        """生成随机数，返回shape形状的随机数矩阵
        如果需要的随机数矩阵的形状大于已有随机数矩阵的形状，则重新生成随机数矩阵，否则复用已有随机数矩阵
            rands_method: 控制随机数生成方法，包括伪随机数和低差异序列两种方法
            _ld_method: 控制低差异序列生成方法，包括Sobol和Halton两种低差异序列类型
        Args:
            shape: np.ndarray，(n_row, n_col)，随机数矩阵的形状
        Returns: self.rands，shape形状的随机数矩阵
        """
        if (self.rands is not None and (not self.__reset_rands)
                and self.rands.shape[0] >= shape[0] and self.rands.shape[1] >= shape[1]):
            logging.info("复用已有随机数")
            return self.rands[:shape[0], :shape[1]]
        else:
            np.random.seed(self.seed)
            if self.rands_method == RandsMethod.Pseudorandom:
                self.rands = np.random.standard_normal(shape)
            elif self.rands_method == RandsMethod.LowDiscrepancy:
                if self._ld_method == LdMethod.Sobol:
                    sampler = qmc.Sobol(d=shape[0], scramble=True, seed=self.seed)
                elif self._ld_method == LdMethod.Halton:
                    sampler = qmc.Halton(d=shape[0], scramble=True, seed=self.seed)
                else:
                    raise ValueError(
                        f'随机数生成方法输入错误，应为（Halton, Sobol）二者之一， 当前输入为{RandsMethod}')
                sampler.fast_forward(30)  # 跳过前三十项
                uniform_rands = sampler.random(shape[1]).transpose()
                np.random.shuffle(uniform_rands)
                self.rands = norm.ppf(uniform_rands)
            else:
                raise ValueError(
                    f'随机数生成方法输入错误，应为（RandsMethod.LowDiscrepancy, RandsMethod.Pseudorandom）二者之一， 当前输入为{RandsMethod}')
            self.__reset_rands = False  # 重置随机数标志位
            return self.rands

    def path_generator(self, n_step, spot=None, t_step_per_year=243):
        """根据标的种类和标的参数，生成价格路径，返回价格路径矩阵
        如果标的参数不变，且需要的价格路径矩阵的形状 ≤ 已有价格路径矩阵的形状，则复用已有价格路径矩阵，否则重新生成价格路径矩阵
        Args:
            n_step: int，价格路径的时间步数
            spot: float，标的期初价格
            t_step_per_year: int，每年的时间步数
        Returns: self.s_paths，价格路径矩阵
        """
        if (self.s_paths is not None and (not self.__reset_paths)
                and self.s_paths.shape[0] >= n_step + 1 and self.s_paths.shape[1] >= self.n_path
                and spot == self.process.spot()):
            logging.info("复用已有价格路径")
            return self.s_paths[:n_step + 1, :self.n_path]
        # else:
        return self._regenerate_paths(n_step, self.n_path, spot, t_step_per_year)

    def _regenerate_paths(self, n_step, n_path, spot, t_step_per_year):
        """生成价格路径的执行函数
        Args:
            n_step: int，价格路径的时间步数
            n_path: int，模拟路径数量
            spot: float，标的期初价格
            t_step_per_year: int，每年的时间步数
        Returns:
        """
        self.s_paths = np.empty(shape=(n_step + 1, n_path))
        dt = 1 / t_step_per_year

        if self.process() == ProcessType.BSProcess1D:
            if self.antithetic_variate:
                if n_path % 2 != 0:
                    n_path += 1
                half_n_path = n_path // 2
                self.rands = self._randoms_generator(shape=(n_step, half_n_path))
                rand_s = np.concatenate((self.rands, -self.rands), axis=1)
            else:
                self.rands = self._randoms_generator(shape=(n_step, n_path))
                rand_s = self.rands.copy()

            self.s_paths[0] = spot
            for step in range(1, n_step + 1):
                self.s_paths[step] = self.process.evolve(dt * step, self.s_paths[step - 1], dt, rand_s[step - 1])

        elif self.process() == ProcessType.Heston:
            self.var_paths = np.empty(shape=(n_step + 1, n_path))
            if self.antithetic_variate:
                self.rands = self._randoms_generator(shape=(n_step, n_path))
                rand_s = self.rands[:, :self.rands.shape[1] // 2].copy()
                rand_v = self.rands[:, self.rands.shape[1] // 2:].copy()
                rand_v = self.process.var_rho * rand_s + np.sqrt(1 - self.process.var_rho ** 2) * rand_v
                rand_s = np.concatenate((rand_s, -rand_s), axis=1)
                rand_v = np.concatenate((rand_v, -rand_v), axis=1)
            else:
                self.rands = self._randoms_generator(shape=(n_step, n_path * 2))
                rand_s = self.rands[:, :self.rands.shape[1] // 2].copy()
                rand_v = self.rands[:, self.rands.shape[1] // 2:].copy()
                rand_v = self.process.var_rho * rand_s + np.sqrt(1 - self.process.var_rho ** 2) * rand_v
            self.s_paths[0] = spot
            self.var_paths[0] = self.process.v0

            for step in range(1, n_step + 1):
                [self.s_paths[step], self.var_paths[step]] = self.process.evolve(dt * step, [self.s_paths[step - 1],
                                                                                             self.var_paths[step - 1]],
                                                                                 dt,
                                                                                 [rand_s[step - 1], rand_v[step - 1]])
            self.var_paths = np.maximum(self.var_paths, 0)  # heston方差非负修正：部分截断形式
        else:
            raise ValueError(f'随机过程类型输入错误，应为（ProcessType.BSProcess1D, ProcessType.Heston）二者之一，'
                             f'当前输入为{self.process()}')

        self.__reset_paths = False  # 重置路径标志位
        return self.s_paths
