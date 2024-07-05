#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType
from .mc_autocallable_engine import MCAutoCallableEngine
from pricelib.common.time import global_evaluation_date


class MCParisSnowballEngine(MCAutoCallableEngine):
    """巴黎雪球MC定价引擎"""

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
        self._maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        # 敲出观察日
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        self.obs_dates = np.array([num for num in obs_dates if num >= 0])
        # 敲入观察日
        knock_in_obs_dates = prod.knock_in_obs_dates.count_business_days(calculate_date)
        self.knock_in_obs_dates = np.round(np.array([num for num in knock_in_obs_dates if num >= 0])).astype(int)
        # 支付日 - 计算起始日到支付日的天数，用于计算应支付的敲出收益
        calculate_start_diff = (calculate_date - prod.start_date).days
        pay_dates = prod.pay_dates.count_calendar_days(prod.start_date)
        self.pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= calculate_start_diff])
        assert len(self.obs_dates) == len(self.pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        # 支付日 - 计算估值日到支付日的天数，用于折现
        pay_dates_tau = prod.pay_dates.count_calendar_days(calculate_date)
        self.pay_dates_tau = np.array([num / prod.annual_days.value for num in pay_dates_tau if num >= 0])
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        self._barrier_out = prod.barrier_out[-len(self.obs_dates):].copy()
        self._coupon_out = prod.coupon_out[-len(self.obs_dates):].copy()
        self._barrier_in = prod.barrier_in[-len(self.knock_in_obs_dates):].copy()
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        self.s_paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                           t_step_per_year=prod.t_step_per_year).copy()
        # 统计各个情景的payoff
        self._cal_knock_out_date()
        self._cal_knock_out_payoff()
        self._cal_knock_in_scenario(calculate_date)
        self._cal_hold_to_maturity_payoff()
        self._cal_knock_in_payoff()
        result = (self.knock_out_profit + self.hold_to_maturity_profit + self.knock_in_loss) / self.n_path
        return result

    def _cal_knock_in_scenario(self, calculate_date):
        """计算每条路径敲入时间"""
        prod = self.prod
        if prod.status == StatusType.DownTouch:
            pass
        else:
            # 判断某一条路径是否有敲入
            knock_in_level = np.tile(self._barrier_in, (self.n_path, 1)).T  # 变敲入（与敲出观察日同长度的列表）
            knock_in_bool = np.where(self.s_paths[self.knock_in_obs_dates - 1] <= knock_in_level, 1, 0)
            knock_in_count = np.cumsum(knock_in_bool, axis=0)
            knock_in_scenario = np.max(knock_in_count, axis=0) >= prod.knock_in_times
            self.knock_in_scenario = knock_in_scenario & self.not_knock_out
