#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import warnings
from abc import ABCMeta
from functools import lru_cache
import numpy as np
import numba as nb
from scipy.stats import norm
from ..processes import StochProcessBase
from ..utilities.enums import QuadMethod, ProcessType, EngineType
from ..utilities.patterns import HashableArray
from .engine_base import PricingEngineBase


@lru_cache(maxsize=1)
@nb.njit(cache=True, fastmath=True)
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


@nb.njit(cache=True, fastmath=True, parallel=True)
def step_backward_jit(x, y, v, t, r, q, vol, quad_vector, quad_index):
    """向前递推一步，更新期权价值向量，使用jit加速
    Args:
        x: np.ndarray，需要计算的未知时点，对应的标的价格向量的绝对值，不是对数价格
        y: np.ndarray，已知时点，对应的标的价格向量的绝对值，不是对数价格
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


@nb.njit(cache=True, fastmath=True)
def fft_convolve_jit(v, pdf, t, r, ln_ds, quad_vector, quad_index):
    """快速傅里叶变换计算卷积，使用jit加速
    FFT加速多项式乘法的原理: 把空间域中复杂的卷积给转换到频率去进行乘法运算，然后再通过逆变换，就可以得到在空间域中的卷积运算的结果
    Args:
        v: np.ndarray，已知时点，对应的期权价值向量
        pdf: np.ndarray，未知时点到已知时点，对应的转移概率密度向量
        t: float，时间间隔
        r: float，无风险利率
        ln_ds: float, 积分变量，对数价格均匀格点的间隔
        quad_vector: np.ndarray，积分权重向量
        quad_index: int，数值积分方法对应的调整系数
    Returns:
        np.ndarray, 卷积结果
    """
    len_v = v.size
    len_pdf = pdf.size
    v = np.hstack((v * quad_vector, np.zeros(len_pdf - len_v)))  # 对v加倍次数界，乘以辛普森积分权重
    fft_v = np.fft.fft(v)  # 对一个方程的系数进行傅里叶变换
    fft_pdf = np.fft.fft(pdf)  # 对第二个方程的系数进行傅里叶变换
    fft_res = fft_v * fft_pdf  # 对两个FFT变换结果进行相乘 (numpy中已经实现了数组相乘)
    fft_res = np.fft.ifft(fft_res).real
    fft_res = fft_res[len_v - 1:] / quad_index * ln_ds * np.exp(- r * t)
    return fft_res


class QuadEngine(PricingEngineBase, metaclass=ABCMeta):
    """数值积分法定价引擎，目前只支持常数r、q、vol"""
    engine_type = EngineType.QuadEngine

    def __init__(self, stoch_process: StochProcessBase = None, quad_method=QuadMethod.Simpson, n_points=1301,
                 *, s=None, r=None, q=None, vol=None):
        """初始化数值积分法定价引擎
        Args:
            stoch_process: StochProcessBase，随机过程
            quad_method: 数值积分方法，QuadMethod枚举类，Trapezoid梯形法则/Simpson辛普森法则
            n_points: int，积分点个数 (需要注意，辛普森法则的n_points必须是奇数)
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.n_points = n_points
        self.quad_method = quad_method
        # 以下为计算过程的中间变量
        self.backward_steps = 1
        self.r = None
        self.q = None
        self.vol = None
        # 积分法相关变量
        self.ln_s_vec = None
        self.ln_ds = None
        self.ln_boundary = None
        self.prod = None

    def set_stoch_process(self, stoch_process: StochProcessBase):
        assert stoch_process.process_type == ProcessType.BSProcess1D, 'Error: 数值积分方法只能使用1维BSM动态过程'
        self.process = stoch_process

    def _check_method_params(self):
        """n_points数检查：设置价格格点数下限，检查格点数是否是奇数"""
        if self.n_points % 2 == 0:
            self.n_points += 1

    def set_quad_params(self, r=None, q=None, vol=None):
        """设置常数r、q、vol"""
        self.r = r
        self.q = q
        self.vol = vol

    def init_grid(self, spot, vol, tau, barrier_low=None, barrier_high=None):
        """初始化积分法价格格点
        为了使用快速傅里叶变换进行卷积运算，需要将价格取对数，从而将分布转化为对称的正态分布
        如果存在高低障碍价格，需要确定价格格点的边界
        Args:
            spot: float, 当前标的价格
            vol: float, 波动率
            tau: float, 年化到期时间
            barrier_low: float, 存续期内，所有低障碍价格的最小值
            barrier_high: float, 存续期内，所有高障碍价格的最大值
        Returns:
            self.ln_s_vec: np.ndarray, 对数价格的均匀格点向量
            self.ln_ds: np.ndarray, 对数价格的均匀格点的间距
            self.ln_boundary: np.ndarray, 只有两个元素，(对数低价格边界，对数高价格边界)
        """
        c = np.exp(10 * vol * np.sqrt(tau) + 0.5 * vol ** 2 * tau)
        boundary = np.array([spot / c, spot * c])
        if barrier_low is not None:
            boundary[0] = np.max(barrier_low, boundary[0])
        if barrier_high is not None:
            boundary[1] = np.min(barrier_high, boundary[1])
        boundary = np.log(boundary)
        self.ln_s_vec = HashableArray(np.linspace(boundary[0], boundary[1], self.n_points))
        self.ln_ds = self.ln_s_vec[1] - self.ln_s_vec[0]
        self.ln_boundary = HashableArray(boundary)

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

    def quad_step_backward(self, x, y, v, t, quad_method=None):
        """直接计算数值积分: 向前递推一步，用vy推出vx，更新期权价值向量
        Args:
            x: np.ndarray，需要计算的未知时点，对应的标的价格向量的绝对值，不是对数价格
            y: np.ndarray，已知时点，对应的标的价格向量的绝对值，不是对数价格。为x之后的时点
            v: np.ndarray，已知时点，对应的期权价值向量vy
            t: float，时间间隔
            quad_method: 数值积分方式，QuadMethod枚举类，Trapezoid梯形法则/Simpson辛普森法则
        Returns:
            result: np.ndarray，更新后的期权价值向量vx
        """
        quad_method = self.quad_method if quad_method is None else quad_method
        quad_vector, quad_index = get_quad_vector(y.size, quad_method)
        result = step_backward_jit(x, y, v, t, self.r, self.q, self.vol, quad_vector, quad_index)
        result = np.array(result)
        return result

    def fft_step_backward(self, x, y, v, t):
        """FFT积分: 向前递推一步，用vy推出vx，更新期权价值向量
            当x, y使用相同间隔，并且分布具有对称性时，可以通过快速傅里叶变换加速计算
        Args:
            x: np.ndarray，需要计算的未知时点，对应的标的价格向量的对数值的均匀格点
            y: np.ndarray，已知时点，对应的标的价格向量的对数值的均匀格点。为x之后的时点
            v: np.ndarray，已知时点，对应的期权价值向量vy
            t: float，时间间隔
        Returns:
            result: np.ndarray，更新后的期权价值向量vx
        """
        quad_vector, quad_index = get_quad_vector(y.size, self.quad_method)
        result = fft_convolve_jit(v, self.get_pdf(x, y, t), self.r, t, self.ln_ds, quad_vector, quad_index)
        result = np.array(result)
        return result

    def get_pdf(self, x_vec, y_vec, t):
        """计算对数价格的概率密度函数，即x到y的转移概率密度 (仅FFT使用)
        Args:
            x_vec: np.ndarray, 未知时点的对数标的价格向量
            y_vec: np.ndarray, 已知时点的对数标的价格向量，为x之后的时点
            t: float, 时间间隔
        Returns:
            np.ndarray, x到y的转移概率密度
        """
        quad_vec = np.concatenate(((x_vec - x_vec[-1] + y_vec[0])[:-1], y_vec))
        quad_vec -= x_vec[0]
        pdf = norm.pdf(quad_vec, (self.r - self.q - 0.5 * self.vol ** 2) * t, self.vol * np.sqrt(t))
        return np.flip(pdf)

    @staticmethod
    @lru_cache
    def get_lns_vec(full_lns_vec, current_bound):
        """获取目标时刻的 需要FFT计算的价格向量 + [可能]低障碍调整价格向量 + [可能]高障碍调整价格向量
        Args:
            full_lns_vec: np.ndarray，初始生成的对数均匀价格格点向量
            current_bound: np.ndarray，障碍价格，只有两个元素，递推时目标时刻的[低障碍价格绝对值，高障碍价格绝对值]
        Returns:
            fft_lns_vec: np.ndarray, 需要FFT计算的价格向量
            lower_adj_vec: np.ndarray, 三个元素，对数的[低障碍B-，B-和L-的中点，价格向量最低点L-]; 若无需调整，为None
            upper_adj_vec: np.ndarray, 三个元素，对数的[价格向量最高点L+，L+和B+的中点，高障碍B+]; 若无需调整，为None
        """
        with warnings.catch_warnings():  # 临时忽略ln 0 = -inf的警告
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ln_bound = np.log(current_bound)
        fft_lns_vec = full_lns_vec[(full_lns_vec >= ln_bound[0]) & (full_lns_vec <= ln_bound[1])]
        lower_lns_vec = full_lns_vec[full_lns_vec < ln_bound[0]]
        upper_lns_vec = full_lns_vec[full_lns_vec > ln_bound[1]]
        if ln_bound[0] < fft_lns_vec[0] and not np.isinf(ln_bound[0]):
            lower_adj_vec = HashableArray([ln_bound[0], (ln_bound[0] + fft_lns_vec[0]) / 2, fft_lns_vec[0]])
        else:
            lower_adj_vec = None
        if ln_bound[1] > fft_lns_vec[0] and not np.isinf(ln_bound[1]):
            upper_adj_vec = HashableArray([fft_lns_vec[-1], (ln_bound[1] + fft_lns_vec[-1]) / 2, ln_bound[1]])
        else:
            upper_adj_vec = None
        return fft_lns_vec, lower_lns_vec, upper_lns_vec, lower_adj_vec, upper_adj_vec

    @staticmethod
    @lru_cache
    def exp_transfer(arg):
        return np.exp(arg)

    def fft_with_barrier(self, x, y, v, t, upper_adj_x=None, lower_adj_x=None,
                         upper_adj_y=None, lower_adj_y=None, upper_adj_v=None, lower_adj_v=None):
        """当存在障碍时的FFT积分，用vy推出vx
        当存在障碍时，障碍价格很可能不会恰好落在对数价格的均匀格点上，因此需要对最上/最下格点与障碍价格之间的区间单独积分，加入到FFT积分的结果中，
            上下边缘的积分固定使用辛普森法则.
        Args:
            x: np.ndarray，需要计算的未知时点，对应的标的价格向量的对数值的均匀格点(范围在高低障碍价之内的格点)
            y: np.ndarray，已知时点，对应的标的价格向量的对数值的均匀格点，为x之后的时点
            v: np.ndarray，已知时点，对应的期权价值向量vy
            t: float，时间间隔
            upper_adj_x: np.ndarray，未知时点，上边界3个对数价格点(最大格点，两者中点，对数高障碍价)
            lower_adj_x: np.ndarray，未知时点，下边界3个对数价格点(对数低障碍价，两者中点，最小格点)
            upper_adj_y: np.ndarray，已知时点，上边界3个对数价格点(最大格点，两者中点，对数高障碍价)
            lower_adj_y: np.ndarray，已知时点，下边界3个对数价格点(对数低障碍价，两者中点，最小格点)
            upper_adj_v: np.ndarray，已知时点，上边界3个对数价格点对应的期权价值向量
            lower_adj_v: np.ndarray，已知时点，下边界3个对数价格点对应的期权价值向量
        Returns:
            result: np.ndarray，更新后的期权价值向量vx
        """
        v_vec = self.fft_step_backward(x, y, v, t)
        # 由于quad_step_backward方法的输入参数是价格，而不是对数价格，此处需要做转换
        exp_x = self.exp_transfer(x)
        if lower_adj_y is not None:
            lower_exp_y = self.exp_transfer(lower_adj_y)
            v_vec += self.quad_step_backward(exp_x, lower_exp_y, lower_adj_v, t, quad_method=QuadMethod.Simpson)
        if upper_adj_y is not None:
            upper_exp_y = self.exp_transfer(upper_adj_y)
            v_vec += self.quad_step_backward(exp_x, upper_exp_y, upper_adj_v, t, quad_method=QuadMethod.Simpson)

        if lower_adj_x is not None:
            lower_exp_x = self.exp_transfer(lower_adj_x)
            lower_v = self.quad_step_backward(lower_exp_x[:2], y, v, t, quad_method=QuadMethod.Simpson)
            if lower_adj_y is not None:
                lower_v += self.quad_step_backward(lower_exp_x[:2], lower_exp_y, lower_adj_v, t,
                                                   quad_method=QuadMethod.Simpson)
            if upper_adj_y is not None:
                lower_v += self.quad_step_backward(lower_exp_x[:2], upper_exp_y, upper_adj_v, t,
                                                   quad_method=QuadMethod.Simpson)
        else:
            lower_v = None

        if upper_adj_x is not None:
            upper_exp_x = self.exp_transfer(upper_adj_x)
            upper_v = self.quad_step_backward(upper_exp_x[-2:], y, v, t, quad_method=QuadMethod.Simpson)
            if lower_adj_y is not None:
                upper_v += self.quad_step_backward(upper_exp_x[-2:], lower_exp_y, lower_adj_v, t,
                                                   quad_method=QuadMethod.Simpson)
            if upper_adj_y is not None:
                upper_v += self.quad_step_backward(upper_exp_x[-2:], upper_exp_y, upper_adj_v, t,
                                                   quad_method=QuadMethod.Simpson)
        else:
            upper_v = None

        return v_vec, lower_v, upper_v
