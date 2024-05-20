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


class SnowballPlus(AutocallableBase):
    """看涨雪球(雪球增强，雪球plus)产品类"""

    def __init__(self, s0, barrier_out, barrier_in, coupon_out, coupon_div=None, lock_term=3, margin_lvl=1, parti_in=1,
                 parti_out=1, strike_call=None, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, obs_dates=None, pay_dates=None, engine=None, annual_days=AnnualDays.N365,
                 status=StatusType.NoTouch, t_step_per_year=243, s=None, r=None, q=None, vol=None):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args: 以下为区别于经典雪球的参数
            parti_out: float，看涨结构参与率
            strike_call: float，敲出价之上的看涨结构行权价
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, status=status,
                         annual_days=annual_days, parti_in=parti_in, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year, engine=engine, s=s, r=r, q=q, vol=vol)
        len_obs_dates = len(self.obs_dates.date_schedule)
        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.barrier_in = np.ones(len_obs_dates) * barrier_in
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div if coupon_div is not None else self.coupon_out[-1]
        self.parti_out = parti_out
        self.strike_upper = s0
        self.strike_lower = 0
        self.strike_call = strike_call if strike_call is not None else barrier_out

    def __repr__(self):
        """返回期权的描述"""
        return "看涨雪球(雪球增强，雪球plus)"
