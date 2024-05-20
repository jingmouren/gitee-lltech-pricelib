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


class FlooredSnowball(AutocallableBase):
    """保底雪球(限损雪球，不追保雪球)产品类"""

    def __init__(self, s0, barrier_out, barrier_in, coupon_out, strike_lower=None, margin_lvl=1,
                 coupon_div=None, lock_term=3, annual_days=AnnualDays.N365, parti_in=1,
                 maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR, obs_dates=None,
                 pay_dates=None, engine=None, status=StatusType.NoTouch, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args: 以下为区别于经典雪球的参数
            strike_lower: float，敲入发生后(熊市价差)的低行权价，即保底边界，越高提供的保护越多，值为0时不保底
            margin_lvl: float，保证金比例，默认为1，即无杠杆。
                                不追保雪球一般将保证金比例设为最大可能亏损，例如75敲入，保证金比例25%，即为4倍杠杆
            parti_in: float，敲入后参与率，限损
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
        self.parti_out = 0
        self.strike_upper = s0
        self.strike_lower = strike_lower if strike_lower is not None else barrier_in
        self.t_step_per_year = t_step_per_year

    def __repr__(self):
        """返回期权的描述"""
        return "保底雪球(限损雪球，不追保雪球)"
