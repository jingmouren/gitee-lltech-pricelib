#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCPhoenixEngine(McEngine):
    """凤凰类 AutoCallable Monte Carlo模拟定价引擎，不支持变敲入、变敲出、变票息
            较雪球式AutoCallable，增加派息价格，少了敲出和红利票息
            每个派息（敲出）观察日，如果价格高于派息价格，派固定利息，如果发生敲出，合约提前结束；
            发生敲入后，派息方式不变，到期如果未敲出，结构为看跌空头
    """

    # pylint: disable=too-many-locals
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        self._obs_dates = np.array([num for num in obs_dates if num >= 0])
        pay_dates = prod.pay_dates.count_calendar_days(calculate_date)
        self._pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= 0])
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        self._barrier_out = prod.barrier_out[-len(self._obs_dates):].copy()
        self._barrier_in = prod.barrier_in[-len(self._obs_dates):].copy()
        self._barrier_yield = prod.barrier_yield[-len(self._obs_dates):].copy()
        self._coupon = prod.coupon[-len(self._obs_dates):].copy()
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        self.s_paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                           t_step_per_year=prod.t_step_per_year).copy()
        # 统计各个情景的payoff
        self._cal_knock_out_date()
        self._cal_yield_date()
        self._cal_knock_in_scenario()
        result = self._cal_payoff()
        return result

    def _cal_knock_out_date(self):
        """统计每条路径敲出时间"""
        # 计算每条价格路径最小敲出时间
        barrier_out = np.tile(self._barrier_out, (self.n_path, 1)).T
        barrier_out_matrix = np.tile(np.arange(self._obs_dates.shape[0]), (self.n_path, 1)).T
        barrier_out_matrix = np.where(self.s_paths[self._obs_dates, :] >= barrier_out,
                                      barrier_out_matrix.astype(int), np.inf)
        # 返回每一列的最小值，即为每条路径的敲出时间
        self.knock_out_time_idx = np.min(barrier_out_matrix, axis=0)
        # 统计哪些路径属于发生了敲出的情景(布尔索引)
        self.knock_out_scenario = (self.knock_out_time_idx != np.inf)
        # 每条路径持有时长
        hold_time_idx = self.knock_out_time_idx.copy()
        # 对于未敲出情形，持有到期，因此将索引设置为最后一个观察日的索引，如24个观察日，则指定为23
        hold_time_idx[~self.knock_out_scenario] = self._obs_dates.size - 1
        # 转成整数
        self.hold_time_idx = hold_time_idx.astype(int)

    def _cal_yield_date(self):
        """统计每条路径发生派息的时间"""
        # 统计哪些派息日，标的价格在派息线上方(即该派息日发生派息)
        barrier_yield = np.tile(self._barrier_yield, (self.n_path, 1)).T
        coupon_time_idx = (self.s_paths[self._obs_dates, :] > barrier_yield)
        # 将发生敲出之后的派息bool由True改为False
        self.coupon_bool = np.where(np.arange(coupon_time_idx.shape[0])[:, np.newaxis] > self.knock_out_time_idx,
                                    False, coupon_time_idx)

    def _cal_knock_in_scenario(self):
        """统计哪些路径属于敲入未敲出的情景"""
        # 排除发生了敲出的路径，统计哪些路径属于敲入未敲出
        knock_in_level = np.array(self._barrier_in).repeat(
            np.diff(np.append(np.zeros((1,)), self._obs_dates)).astype(int))
        knock_in_level = np.append(self._barrier_in[0], knock_in_level)
        knock_in_level = np.tile(knock_in_level, (self.n_path, 1)).T
        knock_in_time_idx = (self.s_paths <= knock_in_level)
        # 统计哪些路径属于敲入未敲出
        knock_in_bool = np.any(knock_in_time_idx, axis=0)
        self.knock_in_scenario = np.where(self.knock_out_scenario, False, knock_in_bool)  # 将发生敲出之后的敲入由True改为False

    def _cal_payoff(self):
        """统计每条路径的收益加总"""
        # 不同派息日的票息分别计算贴现因子，得到派息的现值
        discount_factor = np.empty(self._pay_dates.size)
        for i, pay_d in enumerate(self._pay_dates):
            discount_factor[i] = self.process.interest.disc_factor(pay_d)
        discounted_coupon = self._coupon * self.prod.s0 * discount_factor

        # payoff汇总
        payoff = 0
        # 1.未敲入的部分（敲出/未敲入未敲出），到期还本
        payoff += np.sum(discount_factor[self.hold_time_idx[~self.knock_in_scenario]]
                         ) * self.prod.margin_lvl * self.prod.s0
        # 2.派息payoff
        payoff += np.sum(np.sum(self.coupon_bool, axis=1) * discounted_coupon)
        # 3.敲入，承担跌幅损失
        s_vec = self.s_paths[-1, self.knock_in_scenario].copy()
        s_vec[np.where(s_vec < self.prod.strike_lower)] = self.prod.strike_lower
        payoff += ((self.prod.margin_lvl * self.prod.s0 - self.prod.strike_upper
                    ) * np.sum(self.knock_in_scenario) + np.sum(s_vec)) * discount_factor[-1]
        payoff /= self.n_path
        return payoff
