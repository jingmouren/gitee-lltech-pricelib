#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from ..processes import StochProcessBase
from ..utilities.numerical import LinearFlat, CubicSplineFlat, TDMA_ldu_jit
from ..utilities.patterns import HashableArray
from ..utilities.enums import ProcessType, EngineType, StatusType
from .engine_base import PricingEngineBase


# pylint: disable=invalid-name
class FdmEngine(PricingEngineBase, metaclass=ABCMeta):
    """障碍期权PDE有限差分法定价引擎，适用于连续观察障碍期权价值"""
    engine_type = EngineType.PdeEngine

    def __init__(self, stoch_process: StochProcessBase = None, s_step=400, n_smax=4, fdm_theta=0.5, *,
                 s=None, r=None, q=None, vol=None):
        """初始化有限差分法定价引擎
        Args:
            stoch_process: StochProcessBase随机过程
            s_step: int，价格步数
            n_smax: int，价格网格上界，设为n_smax倍初始价格
            fdm_theta: float，时间方向有限差分theta，0：explicit, 1: implicit, 0.5: Crank-Nicolson
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.s_step = max(s_step, 50)  # 价格步数
        self.n_smax = n_smax  # 价格网格上界s0倍数, 设为n_smax倍初始价格
        self.fdm_theta = fdm_theta  # 时间方向有限差分theta，0：explicit, 1: implicit, 0.5: Crank-Nicolson
        # 以下为计算过程的中间变量
        self.fdm = None  # 有限差分网格FdmGrid对象
        self.smax = None  # 价格网格上界

    def set_stoch_process(self, stoch_process: StochProcessBase):
        assert stoch_process.process_type == ProcessType.BSProcess1D, 'Error: PDE有限差分法目前只能使用1维BSM动态过程'
        self.process = stoch_process

    def delta(self, prod, t=0, spot=None, step=None):
        """求t时刻，价格spot处的delta值
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
        Returns: delta = ∂V/∂S
        """
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        f = self.fdm.functionize(self.fdm.v_grid[1:-1, t], kind='cubic')
        delta = (f(min(spot + step, self.smax)) - f(min(spot - step, self.smax))) / (2 * step)
        return delta

    def gamma(self, prod, t=0, spot=None, step=None):
        """求t时刻，价格spot处的gamma值
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
        Returns: gamma = ∂2V/∂S2
        """
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        f = interp1d(self.fdm.s_vec, self.fdm.v_grid[1:-1, t], kind='cubic')
        gamma = (f(spot + step) - 2 * f(spot) + f(spot - step)) / (step ** 2)
        return gamma

    def theta(self, prod, t=0, spot=None, step=1):
        """计算一天的theta值
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: int，时间步长，默认为1天
        Returns: theta = ∂V/∂t
        """
        spot = self.process.spot() if spot is None else spot
        f = self.fdm.functionize(self.fdm.v_grid[1:-1, t], kind='cubic')
        g = self.fdm.functionize(self.fdm.v_grid[1:-1, t + step], kind='cubic')
        theta = (g(spot) - f(spot)) / (step * spot)  # ( step * self.dt)
        return theta

    def vega(self, prod, t=0, spot=None, step=0.01):
        """计算波动率上升1%的vega todo:目前只支持常数波动率
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
        Returns: vega = ∂V/∂σ
        """
        spot = self.process.spot() if spot is None else spot
        last_vol = self.process.vol.volval
        last_grid = self.fdm.v_grid.copy()
        fdm_grid0 = self.fdm
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.vol.volval = last_vol + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fdm
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        vega = (g(spot) - f(spot)) / spot
        self.process.vol.volval = last_vol
        self.fdm.v_grid = last_grid
        return vega

    def rho(self, prod, t=0, spot=None, step=0.01):
        """计算无风险利率上升1%的rho todo:目前只支持常数无风险利率
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
        Returns: rho = ∂V/∂r
        """
        spot = self.process.spot() if spot is None else spot
        last_r = self.process.interest.data
        last_grid = self.fdm.v_grid.copy()
        fdm_grid0 = self.fdm
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.interest.data = last_r + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fdm
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        rho = (g(spot) - f(spot)) / spot
        self.process.interest.data = last_r
        self.fdm.v_grid = last_grid
        return rho

    def pv_and_greeks(self, prod, t=0, spot=None):
        """当一次性计算pv和5个greeks时，可以调用此函数，减少计算量
        Args:
            prod: Product，产品对象
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
        Returns: pv, delta, gamma, vega, theta, rho
        """
        spot = self.process.spot() if spot is None else spot
        s_step = spot * 0.01
        f = self.fdm.functionize(self.fdm.v_grid[1:-1, t], kind='cubic')

        pv = np.float64(f(spot))
        delta = (f(min(spot + s_step, self.smax)) - f(min(spot - s_step, self.smax))) / (2 * s_step)
        gamma = (f(spot + s_step) - 2 * f(spot) + f(spot - s_step)) / (s_step ** 2)

        g = self.fdm.functionize(self.fdm.v_grid[1:-1, t + 1], kind='cubic')
        theta = (g(spot) - f(spot)) / (1 * spot)

        last_v_grid = self.fdm.v_grid

        last_vol = self.process.vol.volval
        self.process.vol.volval = last_vol + 0.01
        self.calc_present_value(prod=prod)
        g1 = self.fdm.functionize(self.fdm.v_grid[1:-1, t], kind='cubic')
        vega = (g1(spot) - f(spot)) / spot
        self.process.vol.volval = last_vol

        last_r = self.process.interest.data
        self.process.interest.data = last_r + 0.01
        self.calc_present_value(prod=prod)
        g2 = self.fdm.functionize(self.fdm.v_grid[1:-1, t], kind='cubic')
        rho = (g2(spot) - f(spot)) / spot
        self.process.interest.data = last_r

        self.fdm.v_grid = last_v_grid

        return pv, delta, gamma, vega, theta, rho

    def value_matrix(self):
        """返回S-t矩阵"""
        return self.fdm.v_grid, self.fdm.s_vec

    def delta_matrix(self, step=0):
        """计算S-t矩阵中，每个t时刻所有s格点的delta
            step: 标的价步长，默认为0，即选择PDE有限差分网格的价格步长"""
        if step == 0:
            # 默认PDE模拟时的步长 step = fdm_grid.ds
            delta_s_t = (self.fdm.v_grid[2:] - self.fdm.v_grid[:-2]) / (2 * self.fdm.ds)
            return delta_s_t, self.fdm.s_vec  # 中心差分，价格向量掐头0去尾smax，正好是fdm_grid.s_vec
        else:
            # 重新定义步长，在模拟矩阵的基础上插值
            n_step = np.round(self.smax / step).astype('int')
            spots = np.arange(0, n_step, 1) * step
            # 下方n_step和spots用于避免计算远离敲入水平的场景，降低运算量
            # n_step = 2 * np.round((self.spot - self.barrier_in + 1) / step)
            # spots = np.arange(0, n_step, 1) * step + self.barrier_in - 1
            delta_s_t = np.zeros((n_step - 1, self.fdm.v_grid.shape[1]))
            # 分别在fdm_grid.s_vec价格向量首位添加0和smax
            x_vec = np.insert(self.fdm.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            for j in range(self.fdm.v_grid.shape[1]):
                f = interp1d(x_vec, self.fdm.v_grid[:, j], kind='cubic')
                for i, spot in enumerate(spots[1:]):
                    delta_s_t[i, j] = (f(spot + step) - f(spot - step)) / (step * 2)
            return delta_s_t, spots

    def gamma_matrix(self, step=0):
        """计算S-t矩阵中，每个t时刻所有s格点的gamma
            step: 标的价步长，默认为0，即选择PDE有限差分网格的价格步长"""
        if step == 0:
            # 默认PDE模拟时的步长 step = fdm_grid.ds
            gamma_s_t = np.diff(self.fdm.v_grid, n=2, axis=0) / self.fdm.ds ** 2
            return gamma_s_t, self.fdm.s_vec
        else:
            # 重新定义步长，在模拟矩阵的基础上插值
            n_step = np.round(self.smax / step).astype('int')
            step = self.smax / n_step
            spots = np.arange(0, n_step, 1) * step

            gamma_s_t = np.zeros((n_step - 2, self.fdm.v_grid.shape[1]))
            # 分别在fdm_grid.s_vec价格向量首位添加0和smax
            x_vec = np.insert(self.fdm.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            for j in range(self.fdm.v_grid.shape[1]):
                f = interp1d(x_vec, self.fdm.v_grid[:, j], kind='cubic')
                for i, spot in enumerate(spots[1:-2]):
                    gamma_s_t[i, j] = (f(spot + step) - 2 * f(spot) + f(spot - step)) / (step ** 2)
            return gamma_s_t, spots

    def theta_matrix(self, step=0, status: StatusType = None):
        """计算S-t矩阵中，每个s价格所有t时刻的theta'
            step: 时间步长，默认为0，即选择PDE有限差分网格的时间步长"""
        if step == 0:
            # 默认PDE模拟时的步长
            spot = self.process.spot()
            theta_s_t = np.diff(self.fdm.v_grid, axis=1) / spot  # / self.dt
            x_vec = np.insert(self.fdm.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            return theta_s_t, x_vec
        else:  # todo: 时间步长与pde有限差分网格不同的theta
            raise NotImplementedError("todo: 时间步长与pde有限差分网格不同的theta.")


class FdmGrid:
    """有限差分网格基类，设置系数矩阵和边界条件，定义递推方法
    网格的边界是[0, smax]；演化方法是逐天evolve，便于处理敲入覆盖未敲入的值
    适用于离散观察障碍PDE有限差分法定价，这样求得的是离散观察障碍期权的价值"""

    def __init__(self, smax, maturity, t_step_per_year=243, s_step=400, fn_pde_coef=None, fdm_theta=1):
        """初始化有限差分网格
        注意stdev、fwd、fn_abc以及bound的量纲必须一致，要么都是绝对值，要么都是除以F标准化的相对值
        Args:
            smax: float，价格网格上界
            t_step_per_year: int，每年的时间步数
            s_step: int，价格步数
            fn_pde_coef: function，根据时间t和价格向量s_vec，返回PDE系数相关向量a,b,c。设pde有限差分的价格向量索引为i_vec，
                        a是需要乘以i_vec^2的系数，b是需要乘以i_vec的系数，c是不需要乘以i_vec的系数
                        对于BSM而言，a是 vol^2, b是r-q, c是r
            fdm_theta: float，有限差分时间方向的theta方法，theta=0.5是Crank-Nicolson方法，theta=0是显式Euler方法，theta=1是隐式Euler方法
        """
        self._theta = fdm_theta
        self.t_step_per_year = t_step_per_year  # 每年的时间步数
        self.s_min = 0  # 价格向量下界
        self.s_max = smax  # 价格向量上界
        self.s_vec = np.linspace(self.s_min, self.s_max, int(max(s_step, 50)) + 1)[1:-1]  # s_vec 标的价格向量
        self.ds = (self.s_vec[-1] - self.s_vec[0]) / (self.s_vec.size - 1)  # 价格步长
        num = round(maturity * self.t_step_per_year)  # 时间格点数
        self.tv = np.linspace(0, maturity, num + 1)  # 时间向量
        self.dt = (self.tv[-1] - self.tv[0]) / (self.tv.size - 1)  # 时间步长
        self.i_vec = np.round(self.s_vec / self.ds).astype(int)  # s_vec 对应的索引向量 (1,2,...,s_step) 不含边界0和s_step
        self.fn_pde_coef = fn_pde_coef  # PDE系数函数，如果是倒向PDE，a是BSM的(sigma*S)^2, b是BSM的(r-q)*S,c是r
        self.eye = sparse.eye(self.s_vec.size)  # 单位矩阵
        self.v_grid = np.zeros((self.s_vec.size + 2, self.tv.size))  # 初始化一个零矩阵，用于记录每个时间t对应的期权价值向量
        # 以下为计算过程的中间变量
        self.lower = None  # 下对角线
        self.diag = None  # 主对角线
        self.upper = None  # 上对角线
        self.A = None
        self.M1 = None
        self.M2 = None

    @lru_cache(maxsize=1)  # 缓存函数结果，避免重复计算。(常数波动率、无风险利率、分红率时，每个时间点的系数矩阵都是一样的)
    def _set_matrix(self, a, b, c, dt):
        """设置线性方程组的系数矩阵，t是时间，yv是上一时间点的期权价值向量
        M1 yv{j} = M2 yv{j-1} - g{j-1} - g{j}，其中g{j}和g{j-1}是边界条件。注意如果是从后向前递推，j的时间比j-1的时间靠前
        fn_pde_coef返回预计算系数，a是需要乘以i_vec^2的系数，b是需要乘以i_vec的系数，c是不需要乘以i_vec的系数
        Args:
            a: np.ndarray，需要乘以i_vec^2的系数
            b: np.ndarray，需要乘以i_vec的系数
            c: np.ndarray，不需要乘以i_vec的系数
            dt: float，格点的时间步长
        Returns: None
        """
        # [1:-1]去掉最上面和最下面的边界条件点，因为这两个点的边界条件是已知的，不需要计算
        diffusion_square = a * self.i_vec ** 2  # 对于BSM而言，a是 vol^2
        drift = b * self.i_vec  # 对于BSM而言，b是 r-q
        #  系数矩阵预计算, l、d、u分别是对角线下方lower、主对角线diagonal和对角线上方upper的系数数组
        self.lower = 0.5 * (diffusion_square - drift)
        self.diag = -diffusion_square - c
        self.upper = 0.5 * (diffusion_square + drift)
        # 创建三对角矩阵A，用来构造半隐式矩阵M1，M2。M1是隐式部分，M2是显式部分
        self.A = sparse.diags((self.lower[1:], self.diag, self.upper[:-1]), (-1, 0, 1), format='csc') * dt  # csc格式的稀疏矩阵
        # j 时点的期权价值向量（未知量）的系数矩阵M1
        self.M1 = self.eye - self._theta * self.A
        # j - 1 时点的期权价值向量（已知量）的系数矩阵M2（对于隐式imp，M2是单位阵）
        self.M2 = self.eye + (1 - self._theta) * self.A

    def _set_vector(self, j, yv, dt):
        """设置矩阵方程等号右侧的 V
        V = 矩阵方程右侧，系数M2乘以j - 1 时点的期权价值向量（已知量）
        Args:
            j: int，第j个时间点(j的顺序是从后向前)
            yv: np.ndarray，j - 1 时点的期权价值向量（已知量）
            dt: float，格点的时间步长
        Returns: V，矩阵方程右侧，系数M2乘以j - 1 时点的期权价值向量（已知量）
        """
        v_vec = self.M2.dot(yv)
        # V的最上面和最下面，分别减去矩阵第一行的lower*yv[0]和最后一行的upper*yv[-1]，这两个值是已知的边界条件，不需要计算
        v_vec[0] += (self._theta * self.v_grid[0, j]
                     + (1 - self._theta) * self.v_grid[0, j + 1]) * self.lower[0] * dt
        v_vec[-1] += (self._theta * self.v_grid[-1, j]
                      + (1 - self._theta) * self.v_grid[-1, j + 1]) * self.upper[-1] * dt
        return v_vec  # 生成三对角矩阵及对应的向量

    def evolve(self, j, yv, dt):
        """根据时间 t 和 j - 1 时间点的期权价值向量yv，返回当前时间点的期权价值向量
        Args:
            j: int，第j个时间点(j的顺序是从后向前)
            yv: np.ndarray，j - 1 时间点的期权价值向量yv
            dt: float，格点的时间步长
        Returns: 当前时间点的期权价值向量
        """
        a, b, c = self.fn_pde_coef(self.tv[j], self.s_vec)
        a = HashableArray(a)
        # yv = self._step_condition(j, yv)  # 步骤条件
        self._set_matrix(a, b, c, dt)  # 设置线性方程组的系数矩阵
        v_vec = self._set_vector(j, yv, dt)  # 设置线性方程组的向量
        # yv = TDMA(self.M1, v_vec)  # 三对角矩阵Thomas算法求解
        # 三对角矩阵Thomas算法求解,jit加速
        yv = TDMA_ldu_jit(self.M1.diagonal(-1), self.M1.diagonal(0), self.M1.diagonal(1), v_vec)
        return yv

    def functionize(self, yv, kind="linear"):
        """返回插值函数，插值的x是价格向量self.s_vec，y是期权价值向量yv
        Args:
            yv: np.ndarray，期权价值向量
            kind: str，插值方法，linear线性插值/cubic三次样条插值
        Returns:
        """
        if kind == "linear":
            return LinearFlat(self.s_vec, yv)
        else:
            return CubicSplineFlat(self.s_vec, yv)


class FdmGridwithBound(FdmGrid):
    """连续观察障碍PDE有限差分网格对象
    与离散观察障碍网格的区别在于，连续观察障碍网格的上下界是障碍价格，而不是[0, smax]；evolve方法可以自行指定起止时间，支持观察时间窗口
    这样求得的期权价值是连续观察障碍期权的价值"""

    def __init__(self, smax, t_step_per_year=243, s_step=400,
                 fn_pde_coef=None, bound=(-np.inf, np.inf), fdm_theta=1):
        """初始化有限差分网格
        注意stdev、fwd、fn_abc以及bound的量纲必须一致，要么都是绝对值，要么都是除以F标准化的相对值
        Args:
            smax: float，价格网格上界
            t_step_per_year: int，每年的时间步数
            s_step: int，价格步数
            fn_pde_coef: function，根据时间t和价格向量s_vec，返回PDE系数相关向量a,b,c。设pde有限差分的价格向量索引为i_vec，
                        a是需要乘以i_vec^2的系数，b是需要乘以i_vec的系数，c是不需要乘以i_vec的系数
                        对于BSM而言，a是 vol^2, b是r-q, c是r
            fdm_theta: float，有限差分时间方向的theta方法，theta=0.5是Crank-Nicolson方法，theta=0是显式Euler方法，theta=1是隐式Euler方法
        """
        self._theta = fdm_theta
        self.t_step_per_year = t_step_per_year  # 每年的时间步数
        self.bound = bound  # 长度为2的元组，高低障碍价格
        self.s_min = max(bound[0], 0)  # 价格向量下界
        self.s_max = min(bound[1], smax)  # 价格向量上界
        # [1:-1]去掉最上面和最下面的边界条件点，因为这两个点的边界条件是已知的，不需要计算
        self.s_vec = np.linspace(self.s_min, self.s_max, int(max(s_step, 50)) + 1)[1:-1]  # s_vec 标的价格向量
        self.ds = (self.s_vec[-1] - self.s_vec[0]) / (self.s_vec.size - 1)  # 价格步长
        # self.i_vec = np.linspace(1, s_step - 1, self.s_vec.size)
        self.i_vec = np.round(self.s_vec / self.ds).astype(int)  # s_vec 对应的索引向量 (1,2,...,s_step) 不含边界0和s_step
        self.fn_pde_coef = fn_pde_coef  # PDE系数函数，如果是倒向PDE，a是BSM的(sigma*S)^2, b是BSM的(r-q)*S,c是r
        self.eye = sparse.eye(self.s_vec.size)  # 单位矩阵
        # 中间变量
        self.fn_bound = None  # 边界条件函数

    def set_boundary_condition(self, fn_bound):
        """设置边界条件
        Args:
            fn_bound: List[lambda]，长度为2的列表，元素为上下两个边界条件关于时间t的函数，
                      fn_bound[0]是下边界，fn_bound[1]是上边界。
        """
        self.fn_bound = fn_bound

    def _set_vector(self, j, yv, dt):
        """设置矩阵方程等号右侧的 V
        V = 矩阵方程右侧，系数M2乘以j - 1 时点的期权价值向量（已知量）
        Args:
            j: int，第j个时间点(j的顺序是从后向前)
            yv: np.ndarray，j - 1 时点的期权价值向量（已知量）
            dt: float，格点的时间步长
        Returns: V，矩阵方程右侧，系数M2乘以j - 1 时点的期权价值向量（已知量）
        """
        V = self.M2.dot(yv)
        # V的最上面和最下面，分别减去矩阵第一行的lower*yv[0]和最后一行的upper*yv[-1]，这两个值是已知的边界条件，不需要计算
        V[0] += (self._theta * self.fn_bound[0](self.tv[j])
                 + (1 - self._theta) * self.fn_bound[0](self.tv[j - 1])) * self.lower[0] * dt
        V[-1] += (self._theta * self.fn_bound[1](self.tv[j])
                  + (1 - self._theta) * self.fn_bound[1](self.tv[j - 1])) * self.upper[-1] * dt
        return V  # 生成三对角矩阵及对应的向量

    def evolve_with_interval(self, start, end, yv, x=None):
        """从start时间点递推到end时间点，更新yv
        若dt>0, 从前向后递推，local_vol校准; dt<0, 从后向前递推，有限差分法定价
        Args:
            start: float，起始时间点，距离估值日的年化期限
            end: float，结束时间点，距离估值日的年化期限
            yv: np.ndarray，start时刻的期权价值向量
            x: 若x不是None，将每个时间t对应的指定的x对应的y值记录下来。
               一般在计算敲入时，传入x=bound即障碍价格，用已敲入的值作为未敲入的边界条件
        Returns:
            yv: np.ndarray，end时刻的期权价值向量
            bc_bound: 边界条件函数，input时间t，output边界条件函数值
        """
        num = max(round(abs(start - end) * self.t_step_per_year), 1)  # 时间格点, 最小取1，是为了应对start=end的情况
        self.tv = np.linspace(start, end, num + 1)  # 生成时间格点
        self.dt = (self.tv[-1] - self.tv[0]) / (self.tv.size - 1)  # dt既可以是正数，也可以是负数
        # self.v_grid = np.zeros((self.s_vec.size + 2, self.tv.size))  # 初始化一个零矩阵，用于记录每个时间t对应的期权价值向量

        if x is None:
            for j in range(1, len(self.tv)):  # 从tv[1:]开始递推，因为tv[0]是已知的
                yv = self.evolve(j, yv, -self.dt)
                # self.v_grid[1:-1, j] = yv  # 记录每个时间t对应的期权价值向量
            return yv
        else:  # 如果需要将每个时间t对应的指定的x对应的y值记录下来return，则需传入x
            # 一般用于计算敲入时，传入x=bound即障碍价格，用已敲入的期权价值作为未敲入的边界条件
            fv = np.empty((self.tv.size, x.size))  # 初始化一个空向量，用于记录每个时间t对应的指定的x对应的y值
            fv[0, :] = self.functionize(yv)(x)
            for j, f in zip(range(1, len(self.tv)), fv[1:]):
                yv = self.evolve(j, yv, -self.dt)
                # self.v_grid[1:-1, j] = yv  # 记录每个时间t对应的期权价值向量
                f[:] = self.functionize(yv)(x)
            return yv, [LinearFlat(self.tv, f) for f in fv.T]  # 返回边界条件函数bc_bound，input时间t，output边界条件函数值
