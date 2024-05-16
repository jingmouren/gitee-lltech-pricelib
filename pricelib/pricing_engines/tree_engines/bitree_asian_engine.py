#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import CallPut, AverageMethod, AsianAveSubstitution
from pricelib.common.processes import StochProcessBase
from pricelib.common.pricing_engine_base import BiTreeEngine
from pricelib.common.time import global_evaluation_date


class BiTreeAsianEngine(BiTreeEngine):
    """算术平均结算价(收盘价)亚式二叉树定价引擎
    不支持几何平均，不支持增强亚式，不支持平均执行价"""

    def __init__(self, stoch_process: StochProcessBase = None, tree_branches=100, n_samples=200, *,
                 s=None, r=None, q=None, vol=None):
        """初始化算术平均亚式二叉树定价引擎
        Args:
            stoch_process: StochProcessBase，随机过程
            tree_branches: int，二叉树每年的分支数
            n_samples: int，在每个树节点历史路径平均股价的最大值和最小值之间，插入数值点的数量
       在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process, tree_branches, s=s, r=r, q=q, vol=vol)
        self.n_samples = n_samples
        # 以下为计算过程的中间变量
        self.prod = None  # Product产品对象
        self.n_step = None  # 二叉树的总分支数
        self.limited_price = None  # 兼容增强亚式的限制价格变量

    def cal_path(self, s_path, obs_start_step, obs_end_step, n_samples):
        """计算每个树节点对应的历史路径平均值的极大值s_max_path与极小值s_min_path，在其中插入固定数量n_samples的数值点，
                        最后返回一个三维数组s_path，其第一维数是n_samples，第二和第三维数等同于树s的第一维和第二维(tree_branches)。
        原理:
        假设第j层树节点分支，(根节点j=0)，每层第i个节点(该层最低股价处i=0)
        不妨设期初(j=0)就开始观察平均收盘价，则到第j层的时候，历史价格路径有(j+1)个节点
        那么历史路径平均值的极大值，前(i+1)层时的价格都一直处于当时股价的最顶端，之后开始逐点下降，直到到达第j层时，下降到第i个节点
                       也就是该价格路径由(i+1)个[1, u, u^2, ..., u^i]和(j-i)个 u^i * d * [1, d, d^2, ...,d^(j-i-1)]构成
        同理历史路径平均值的极小值，前(j-i)层时的价格都一直处于当时股价的最底端，之后开始逐点上升，直到到达第j层时，上升到第i个节点
                       也就是该价格路径由(j-i+1)个[1, d, d^2, ..., d^(j-i)]和i个 d^(j-i) * u * [1, u, u^2, ...,u^(i-1)]构成
        本方法就是按照这种规律进行累加，最后除以每层对应的价格总数(j+1)，即可得到历史路径平均值的极大极小值
        Args:
            s_path: np.ndarray，上三角结构的标的资产价格CRR二叉树，二维数组，标的价格x时间分支
            obs_start_step: float，观察平均的开始时间点，对应的二叉树分支索引
            obs_end_step: float，观察平均的终止时间点，对应的二叉树分支索引
            n_samples: int，在每个树节点历史路径平均股价的最大值和最小值之间，插入数值点的数量
        Returns:
            s_path: np.ndarray，三维数组，其第一维数是n_samples，第二和第三维数等同于树s的第一维和第二维
                    相当于n_samples个二叉树，分别对应了历史路径平均值的极大值s_max_path与极小值s_min_path中间的一个数值点的二叉树
        """
        s_max_path = np.array(s_path)
        s_min_path = np.array(s_path)
        for j in range(self.n_step + 1):
            if j < obs_start_step:
                continue
            if j == obs_start_step:  # 观察起点，开始记录历史价格路径之和的最小值和历史价格路径之和的的最大值
                if self.prod.callput == CallPut.Call:
                    if self.prod.ave_method == AverageMethod.Arithmetic:
                        s_max_path[:j + 1, j] = np.maximum(s_path[:j + 1, j], self.limited_price)
                        s_min_path[:j + 1, j] = np.maximum(s_path[:j + 1, j], self.limited_price)
                    elif self.prod.ave_method == AverageMethod.Geometric:
                        s_max_path[:j + 1, j] = np.maximum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                        s_min_path[:j + 1, j] = np.maximum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                elif self.prod.callput == CallPut.Put:
                    if self.prod.ave_method == AverageMethod.Arithmetic:
                        s_max_path[:j + 1, j] = np.minimum(s_path[:j + 1, j], self.limited_price)
                        s_min_path[:j + 1, j] = np.minimum(s_path[:j + 1, j], self.limited_price)
                    elif self.prod.ave_method == AverageMethod.Geometric:
                        s_max_path[:j + 1, j] = np.minimum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                        s_min_path[:j + 1, j] = np.minimum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                continue
            if j > obs_end_step:  # 观察终点，结束观察平均价格，后续节点的历史价格路径之和的最小值和历史价格路径之和的的最大值都等于上一天顺延
                s_max_path[0, j] = s_max_path[0, j - 1]
                s_max_path[1:j + 1, j] = s_max_path[:j, j - 1]
                s_min_path[j, j] = s_min_path[j - 1, j - 1]
                s_min_path[:j, j] = s_min_path[:j, j - 1]
            else:  # 观察起始点之间，记录历史价格路径之和的最小值和历史价格路径之和的的最大值
                s_max_path[0, j] = s_max_path[0, j - 1]
                s_max_path[1:j + 1, j] = s_max_path[:j, j - 1]
                s_min_path[j, j] = s_min_path[j - 1, j - 1]
                s_min_path[:j, j] = s_min_path[:j, j - 1]
                if self.prod.callput == CallPut.Call:
                    if self.prod.ave_method == AverageMethod.Arithmetic:
                        s_max_path[:j + 1, j] += np.maximum(s_path[:j + 1, j], self.limited_price)
                        s_min_path[:j + 1, j] += np.maximum(s_path[:j + 1, j], self.limited_price)
                    elif self.prod.ave_method == AverageMethod.Geometric:
                        s_max_path[:j + 1, j] += np.maximum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                        s_min_path[:j + 1, j] += np.maximum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                elif self.prod.callput == CallPut.Put:
                    if self.prod.ave_method == AverageMethod.Arithmetic:
                        s_max_path[:j + 1, j] += np.minimum(s_path[:j + 1, j], self.limited_price)
                        s_min_path[:j + 1, j] += np.minimum(s_path[:j + 1, j], self.limited_price)
                    elif self.prod.ave_method == AverageMethod.Geometric:
                        s_max_path[:j + 1, j] += np.minimum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))
                        s_min_path[:j + 1, j] += np.minimum(np.log(s_path[:j + 1, j]), np.log(self.limited_price))

        for i in range(obs_start_step, obs_end_step + 1):
            # 对加和除以天数，即为历史价格路径平均值的最大值/历史价格路径平均值的最小值
            s_min_path[:i + 1, i] = (s_min_path[:i + 1, i] / (i - obs_start_step + 1))
            s_max_path[:i + 1, i] = (s_max_path[:i + 1, i] / (i - obs_start_step + 1))

        if self.prod.ave_method == AverageMethod.Geometric:
            s_min_path = np.exp(s_min_path)
            s_max_path = np.exp(s_max_path)

        # 在历史路径平均股价的最大值和最小值之间，插入固定数量n_samples的数值点，s_path是一个三维数组
        s_path = np.linspace(s_min_path, s_max_path, n_samples)
        return s_path

    @staticmethod
    def _trans_v(ave_s, s_list, v_list):
        """线性插值，求出插入的标的价格数值点对应的期权价值
        Args:
            ave_s: np.ndarray，插入的数值点对应的标的价格
            s_list: np.ndarray，已知点对应的的标的价格
            v_list: np.ndarray，已知点对应的的期权价值
        Returns:
            ave_v: np.ndarray，插入的数值点对应的期权价值
        """
        ave_v = ave_s.copy()
        for i in range(len(s_list[0, :])):
            # np.interp(x, xp, fp)  x: 插值点 xp: 已知点的x坐标  fp: 已知点的y坐标
            ave_v[:, i] = np.interp(ave_s[:, i], s_list[:, i], v_list[:, i])
        return ave_v

    def cal_price(self, s_path, interp_s_path, obs_start_step, obs_end_step, n_samples):
        """递推，计算每个节点的每个可能的历史平均股价数值点对应的期权价格。
            考虑一个具体的平均股价数值点，该点在股价下一步上升或下降后分别对应于下一层一个节点的一个历史平均股价数值，
            但该价格不一定是在已计算出的数值点中，所以一般需要使用插值法求出数值点对应的期权价格。
            然后对股价上升或下降后对应数值所对应期权价格进行加权平均并贴现，即为当前层考虑节点的一个平均股价数值点对应的期权价格。
            不断重复上述过程，直到计算出根节点处期权的价格。
        Args:
            s_path: np.ndarray，上三角结构的标的资产价格CRR二叉树
            interp_s_path: np.ndarray，三维数组，其第一维数是n_samples，第二和第三维数等同于树s的第一维和第二维
                    相当于n_samples个二叉树，分别对应了历史路径平均值的极大值s_max_path与极小值s_min_path中间的一个数值点的二叉树
            obs_start_step: float，观察平均的开始时间点，对应的二叉树分支索引
            obs_end_step: float，观察平均的终止时间点，对应的二叉树分支索引
            n_samples: int，在每个树节点历史路径平均股价的最大值和最小值之间，插入数值点的数量
        Returns:
            float，期权现值
        """
        v_grid = np.zeros(shape=(n_samples, self.n_step + 1, self.n_step + 1))
        for j in range(self.n_step, -1, -1):
            # 终止条件
            if j == self.n_step:
                v_grid[:, :, j] = np.maximum(self.prod.callput.value * (interp_s_path[:, :, -1] - self.prod.strike), 0)
                continue
            if j < obs_start_step or j > obs_end_step:
                # 非观察期内，直接用下一步的上升和下降的v做加权平均并贴现
                v_grid[:, 0:j + 1, j] = (v_grid[:, 0:j + 1, j + 1] * self.p + v_grid[:, 1:j + 2, j + 1] * (1 - self.p)
                                         ) * math.exp(-self.r * self.dt)
            else:  # 观察期内
                # 考虑某一个平均股价数值点，用插值法找到该点在股价下一步上升或下降后的期权价格，然后加权平均并贴现
                up_ave_s = ((j - obs_start_step + 1) * interp_s_path[:, 0:j + 1, j] + s_path[0:j + 1, j + 1]) / (
                        j - obs_start_step + 2)
                down_ave_s = ((j - obs_start_step + 1) * interp_s_path[:, 0:j + 1, j] + s_path[1:j + 2, j + 1]) / (
                        j - obs_start_step + 2)
                v_grid[:, 0:j + 1, j] = self._trans_v(up_ave_s, interp_s_path[:, 0:j + 1, j + 1],
                                                      v_grid[:, 0:j + 1, j + 1]) * self.p
                v_grid[:, 0:j + 1, j] += self._trans_v(down_ave_s, interp_s_path[:, 1:j + 2, j + 1],
                                                       v_grid[:, 1:j + 2, j + 1]) * (1 - self.p)
                v_grid[:, 0:j + 1, j] *= math.exp(-self.r * self.dt)
        return np.mean(v_grid[:, 0, 0])

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert (prod.ave_method == AverageMethod.Arithmetic and prod.substitute == AsianAveSubstitution.Underlying
                and not prod.enhanced), "仅支持算术平均代替资产结算价的亚式期权，不支持几何平均、替代执行价、增强亚式"
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        _maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date,
                                                                            prod.end_date) / prod.t_step_per_year
        obs_start = prod.trade_calendar.business_days_between(calculate_date, prod.obs_start) / prod.t_step_per_year
        obs_end = prod.trade_calendar.business_days_between(calculate_date, prod.obs_end) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        if prod.enhanced:
            self.limited_price = prod.limited_price
        else:
            if prod.callput == CallPut.Call:
                self.limited_price = 0.0001
            else:  # prod.callput == CallPut.Put:
                self.limited_price = 100 * prod.strike  # np.inf

        self.n_step = int(_maturity_business_days * self.tree_branches)
        self.r = self.process.interest(_maturity)
        self.q = self.process.div(_maturity)
        self.vol = self.process.vol(_maturity_business_days, spot)
        # 生成标的价格树
        s_path = self.path_generator(self.n_step, _maturity_business_days, spot).copy()
        obs_start_step = int(obs_start * self.tree_branches)
        obs_end_step = int(obs_end * self.tree_branches)
        # 根据标的价格二叉树，计算每个树节点对应的历史路径平均值的极大值s_max_path与极小值s_min_path，在其中插入固定数量n_samples的数值点
        interp_s_path = self.cal_path(s_path, obs_start_step, obs_end_step, n_samples=self.n_samples)
        # 递推，计算每个节点的每个可能的历史平均股价数值点对应的期权价格
        price = self.cal_price(s_path, interp_s_path, obs_start_step, obs_end_step, n_samples=self.n_samples)
        return price
