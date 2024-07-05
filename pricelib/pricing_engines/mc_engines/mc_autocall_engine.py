#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import CallPut
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCAutoCallEngine(McEngine):
    """AutoCall Note(小雪球) MonteCarlo 模拟定价引擎"""

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
        # 支付日 - 计算起始日到支付日的天数，并转化为年化时间，用于计算应支付的敲出收益
        calculate_start_diff = (calculate_date - prod.start_date).days
        pay_dates = prod.pay_dates.count_calendar_days(prod.start_date)
        pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= calculate_start_diff])
        assert len(obs_dates) == len(pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        # 支付日 - 计算估值日到支付日的天数，用于折现
        pay_dates_tau = prod.pay_dates.count_calendar_days(calculate_date)
        pay_dates_tau = np.array([num / prod.annual_days.value for num in pay_dates_tau if num >= 0])
        # 经过估值日截断列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        _barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        _coupon_out = prod.coupon_out[-len(obs_dates):].copy()
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        s_paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                      t_step_per_year=prod.t_step_per_year).copy()

        barrier = np.tile(_barrier_out, (self.n_path, 1)).T
        # 记录每条路径的具体敲出日，如果无敲出则保留inf

        knock_out_scenario = np.tile(obs_dates, (self.n_path, 1)).T

        if prod.callput == CallPut.Call:
            knock_out_scenario = np.where(s_paths[obs_dates] >= barrier, knock_out_scenario, np.inf)
        elif prod.callput == CallPut.Put:
            knock_out_scenario = np.where(s_paths[obs_dates] <= barrier, knock_out_scenario, np.inf)
        knock_out_date = np.min(knock_out_scenario, axis=0)
        is_knock_out = knock_out_date != np.inf
        # 红利票息
        hold_payoff = ((self.n_path - np.sum(is_knock_out)) * prod.s0 * (prod.coupon_div * pay_dates[-1]
                       + prod.margin_lvl) * self.process.interest.disc_factor(pay_dates_tau[-1]))
        # 敲出部分
        # 敲出对应计息时长
        obs_to_pay = dict(zip(obs_dates, pay_dates))
        if any(is_knock_out):
            pay_dates_vec = np.vectorize(obs_to_pay.get)(knock_out_date[is_knock_out])
            # 敲出对应折现时长
            obs_to_pay_tau = dict(zip(obs_dates, pay_dates_tau))
            pay_dates_tau_vec = np.vectorize(obs_to_pay_tau.get)(knock_out_date[is_knock_out])
            # 敲出对应票息
            obs_to_coupon = dict(zip(obs_dates, _coupon_out))
            coupon_out = np.vectorize(obs_to_coupon.get)(knock_out_date[is_knock_out])
            # 敲出部分的收益
            knock_out_payoff = np.sum(prod.s0 * (coupon_out * pay_dates_vec + prod.margin_lvl)
                                      * self.process.interest.disc_factor(pay_dates_tau_vec))
        else:
            knock_out_payoff = 0
        value = (hold_payoff + knock_out_payoff) / self.n_path
        return value
