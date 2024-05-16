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
        calculate_date = global_evaluation_date() if t is None else t
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        obs_dates = np.array([num for num in obs_dates if num >= 0])
        pay_dates = prod.pay_dates.count_calendar_days(calculate_date)
        pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= 0])
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        _barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        _barrier_in = prod.barrier_in[-len(obs_dates):].copy()
        _barrier_yield = prod.barrier_yield[-len(obs_dates):].copy()
        _coupon = prod.coupon[-len(obs_dates):].copy()
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        # 计算每条价格路径最小敲出时间
        barrier_out = np.tile(_barrier_out, (self.n_path, 1)).T
        # 每条路径敲出时间索引(第几个月敲出)，这里的矩阵是True、False，np.argmax返回的是每一列的第一个最大值True的索引
        knock_out_time_idx = np.argmax(paths[obs_dates, :] > barrier_out, axis=0)
        # 统计哪些路径属于发生了敲出的情景(布尔索引)
        knock_out_scenario = np.where(knock_out_time_idx > 0, True, False)
        # 未敲出情形，将索引设置为最后一个观察日的索引再+1，例如两年24个月的最后一个观察日索引是23，则未敲出索引设为24
        knock_out_time_idx[~knock_out_scenario] = obs_dates.size
        # 每条路径持有时长，对于未敲出情形，持有到期，因此将索引设置为最后一个观察日的索引，即23
        hold_time_idx = knock_out_time_idx.copy()
        hold_time_idx[~knock_out_scenario] = obs_dates.size - 1

        # 统计哪些派息日，标的价格在派息线下方(即该派息日不派息)
        barrier_yield = np.tile(_barrier_yield, (self.n_path, 1)).T
        no_coupon_time_idx = paths[obs_dates, :] < barrier_yield
        # 将发生敲出之后的未派息bool由True改为False
        no_coupon_bool = np.where(np.arange(no_coupon_time_idx.shape[0])[:, np.newaxis] > knock_out_time_idx, False,
                                  no_coupon_time_idx)

        # 排除发生了敲出的路径，统计哪些路径属于敲入未敲出
        barrier_in = np.tile(_barrier_in, (self.n_path, 1)).T
        knock_in_time_idx = paths[obs_dates, :] < barrier_in
        # 将发生敲出之后的敲入由True改为False
        knock_in_bool = np.where(np.arange(knock_in_time_idx.shape[0])[:, np.newaxis] > knock_out_time_idx, False,
                                 knock_in_time_idx)
        # 统计哪些路径属于敲入未敲出
        knock_in_scenario = np.argmax(knock_in_bool, axis=0) > 0  # 敲入未敲出情形

        # 不同派息日的票息分别计算贴现因子，得到派息的现值
        discount_factor = np.empty(pay_dates.size)
        for i, pay_d in enumerate(pay_dates):
            discount_factor[i] = self.process.interest.disc_factor(pay_d)
        discounted_coupon = _coupon * prod.s0 * discount_factor
        # 将每个月的派息现值累加到一起
        cumulated_coupon = np.cumsum(discounted_coupon)

        # payoff汇总
        payoff = 0
        # 1.未敲入的部分（敲出/未敲入未敲出），到期还本
        payoff += np.sum(discount_factor[hold_time_idx[~knock_in_scenario]]) * prod.s0
        # 2.派息payoff
        # 2.1 首先忽略派息线以下，假设在敲出之前的每个派息日都能拿到派息，将派息现值累加
        payoff += np.sum(cumulated_coupon[hold_time_idx])
        # 2.2 然后减去处于派息线下方的派息
        payoff -= np.sum(np.sum(no_coupon_bool, axis=1) * discounted_coupon)
        # 3.敲入，承担跌幅损失
        s_vec = paths[-1, knock_in_scenario].copy()
        s_vec[np.where(s_vec < prod.strike_lower)] = prod.strike_lower
        payoff += ((prod.s0 - prod.strike_upper) * np.sum(knock_in_scenario) + np.sum(s_vec)) * discount_factor[-1]

        payoff /= self.n_path
        return payoff
