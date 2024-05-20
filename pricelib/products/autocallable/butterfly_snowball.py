#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType
from pricelib.common.time import CN_CALENDAR, AnnualDays
from .autocallable_base import AutocallableBase


class ButterflySnowball(AutocallableBase):
    """蝶变雪球结构产品类"""

    def __init__(self, s0, barrier_out, barrier_in, coupon_out1, coupon_out2, coupon_out3, coupon_div=None, lock_term=3,
                 maturity=None, start_date=None, end_date=None, obs_dates=None, pay_dates=None, parti_in=1,
                 margin_lvl=1, coupon_stair_ends=(12, 24, 36), status=StatusType.NoTouch, engine=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args: 以下为区别于经典雪球的参数
            coupon_out1: float，第一阶段敲出票息，百分比，年化
            coupon_out2: float，第二阶段敲出票息，百分比，年化
            coupon_out3: float，第三阶段敲出票息，百分比，年化
            coupon_stair_ends: tuple(int), 分段票息节点, 通常为(12, 24, 36), 代表第几个月末
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, status=status,
                         annual_days=annual_days, parti_in=parti_in, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year, engine=engine, s=s, r=r, q=q, vol=vol)
        len_obs_dates = len(self.obs_dates.date_schedule)

        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.barrier_in = np.ones(len_obs_dates) * barrier_in
        coupon_out = np.ones(len_obs_dates + lock_term - 1)
        coupon_out[:coupon_stair_ends[0]] = coupon_out1
        coupon_out[coupon_stair_ends[0]:coupon_stair_ends[1]] = coupon_out2
        coupon_out[coupon_stair_ends[1]:coupon_stair_ends[2]] = coupon_out3
        self.coupon_out = coupon_out[lock_term - 1:]
        self.coupon_div = coupon_div if coupon_div is not None else coupon_out[-1]
        self.parti_out = 0
        self.strike_upper = s0
        self.strike_lower = 0

    def __repr__(self):
        """返回期权的描述"""
        return "蝶式雪球"
