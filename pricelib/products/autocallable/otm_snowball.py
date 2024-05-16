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


class OTMSnowball(AutocallableBase):
    """OTM雪球结构产品类"""

    def __init__(self, s0, barrier_out, barrier_in, coupon_out, strike_upper, coupon_div=None, lock_term=3,
                 maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR, obs_dates=None,
                 pay_dates=None, annual_days=AnnualDays.N365, parti_in=1, margin_lvl=1, engine=None,
                 status=StatusType.NoTouch, t_step_per_year=243):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args:
            strike_upper: float，敲入发生后(熊市价差)的高行权价，即OTM行权价(<s0)，敲入亏损较小
                                以低行权价为0为例，期初价100，高行权价为90，敲入价80，到期标的价格为78，则亏损比例为 (78 - 90) / 100
            parti_in: float，敲入后参与率，限损
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, status=status,
                         annual_days=annual_days, parti_in=parti_in, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year)
        len_obs_dates = len(self.obs_dates.date_schedule)

        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.barrier_in = np.ones(len_obs_dates) * barrier_in
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div if coupon_div is not None else self.coupon_out[-1]
        self.parti_out = 0
        self.strike_upper = strike_upper
        self.strike_lower = 0
        if engine is not None:
            self.set_pricing_engine(engine)

    def __repr__(self):
        """返回期权的描述"""
        return "OTM雪球"
