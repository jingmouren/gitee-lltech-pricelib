#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from numba import njit, prange
from scipy.interpolate import interp1d, UnivariateSpline


# pylint: disable=invalid-name
def TDMA(X, d):
    """三对角矩阵求解，Thomas算法，the Tridiagonal Matrix Algorithm，是高斯消元法对三对角矩阵的特例。
    Args:
        X: np.adarray，三对角系数矩阵
        d: np.adarray，方程右端向量
    Returns:
        p: np.adarray，方程的解向量，Xp=d
    """
    n = len(d)
    a = np.array(X[np.arange(n - 1) + 1, np.arange(n - 1)])[0]
    b = np.array(X[np.arange(n), np.arange(n)])[0]
    c = np.array(X[np.arange(n - 1), np.arange(n - 1) + 1])[0]
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p.reshape(n, )


# pylint: disable=invalid-name
@njit(cache=True, fastmath=True)  # 不要使用parallel=True，因为计算有顺序，不能并行。否则计算结果不对
def TDMA_jit(X, d):
    """使用numba.jit加速的三对角矩阵求解Thomas算法，the Tridiagonal Matrix Algorithm
    Args:
        X: np.adarray，三对角系数矩阵
        d: np.adarray，方程右端向量
    Returns:
        p: np.adarray，方程的解向量，Xp=d
    """
    # 将矩阵的三条对角线取出来
    n = len(d)
    a = np.zeros(n - 1)
    b = np.zeros(n)
    c = np.zeros(n - 1)
    for i in prange(n):
        if i > 0:
            a[i - 1] = X[i, i - 1]
        b[i] = X[i, i]
        if i < n - 1:
            c[i] = X[i, i + 1]
    # Thomas算法
    for i in prange(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    p = b
    p[-1] = d[-1] / b[-1]
    indices = np.arange(n - 2, -1, -1)  # 创建逆序索引数组
    for j in prange(n - 1):
        i = indices[j]  # 使用逆序索引
        p[i] = (d[i] - c[i] * p[i + 1]) / b[i]
    return p


# pylint: disable=invalid-name
@njit(cache=True, fastmath=True)  # 不要使用parallel=True，因为计算有顺序，不能并行。否则计算结果不对
def TDMA_ldu_jit(a, b, c, d):
    """使用numba.jit大幅加速的三对角矩阵求解Thomas算法，the Tridiagonal Matrix Algorithm
    Args:
        a: np.adarray，三对角系数矩阵的下对角线元素向量
        b: np.adarray，三对角系数矩阵的主对角线元素向量
        c: np.adarray，三对角系数矩阵的上对角线元素向量
        d: np.adarray，方程右端向量
    Returns:
        p: np.adarray，方程的解向量，Xp=d
    """
    n = len(d)
    for i in prange(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    p = b
    p[-1] = d[-1] / b[-1]
    indices = np.arange(n - 2, -1, -1)  # 创建逆序索引数组
    for j in prange(n - 1):
        i = indices[j]  # 使用逆序索引
        p[i] = (d[i] - c[i] * p[i + 1]) / b[i]
    return p


class LinearFlat(interp1d):
    """自定义以最边缘点的值水平外推的线性插值，两侧外推时，使用边界值y[0], y[-1]"""

    def __init__(self, x, y):
        """初始化，执行单变量线性插值interp1d"""
        super().__init__(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))


class CubicSplineFlat(UnivariateSpline):
    """自定义以最边缘点的值水平外推的三次样条插值，
    s是平滑系数，是非负实数，用于控制拟合曲线的平滑程度，s越大平滑程度越大，s越小拟合曲线越接近原始数据点。当s=0时，曲线会穿过所有的原始数据点。
    ext控制x在外推时的行为，ext=3 or 'const', 以边界值水平外推。
    k控制平滑样条的程度，1~5的整数。默认k=3， 三次样条"""

    def __init__(self, x, y):
        """初始化，执行单变量插值UnivariateSpline"""
        super().__init__(x, y, s=0, ext=3)  #


class FlatCubicSpline(UnivariateSpline):
    """自定义边界上的一阶导和二阶导都是0的三次样条插值(这样边界值自然是水平外推的。)
    s是平滑系数，是非负实数，用于控制拟合曲线的平滑程度，s越大平滑程度越大，s越小拟合曲线越接近原始数据点。当s=0时，曲线会穿过所有的原始数据点。
    ext控制x在外推时的行为，ext=3 or 'const', 以边界值水平外推。
    k控制平滑样条的程度，1~5的整数。默认k=3， 三次样条"""

    def __init__(self, x, y):
        """初始化，执行单变量插值UnivariateSpline
        x必须是升序的ascending
        为了模拟边界上的一阶导和二阶导都是0，将x的两端分别外扩一个小量epsilon，对应的y值与边界y值相同"""
        epsilon = 1e-8
        x = np.hstack((x[0] * (1 - epsilon), x, x[-1] * (1 + epsilon)))
        y = np.hstack((y[0], y, y[-1]))
        super().__init__(x, y, s=0, ext=3)
