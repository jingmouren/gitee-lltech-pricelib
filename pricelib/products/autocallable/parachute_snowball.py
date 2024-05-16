#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType
from pricelib.common.time import CN_CALENDAR
from .autocallable_base import AutocallableBase


class ParachuteSnowball(AutocallableBase):
    """降落伞雪球产品类"""

    def __init__(self, s0, barrier_out, barrier_out_final, barrier_in, coupon_out, coupon_div=None,
                 lock_term=3, maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR, obs_dates=None,
                 pay_dates=None, engine=None, status=StatusType.NoTouch, t_step_per_year=243):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args:
            barrier_out: float，敲出障碍价，绝对值/百分比
            barrier_out_final: float，最后一个敲出观察日的敲出障碍价，绝对值/百分比
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, status=status,
                         t_step_per_year=t_step_per_year)
        len_obs_dates = len(self.obs_dates.date_schedule)

        barrier_out = np.ones(len_obs_dates) * barrier_out
        barrier_out[-1] = barrier_out_final
        self.barrier_out = barrier_out
        self.barrier_in = np.ones(len_obs_dates) * barrier_in
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div if coupon_div is not None else self.coupon_out[-1]
        self.parti_out = 0
        self.parti_in = 1
        self.margin_lvl = 1
        self.strike_upper = s0
        self.strike_lower = 0

        if engine is not None:
            self.set_pricing_engine(engine)

    def __repr__(self):
        """返回期权的描述"""
        return "降落伞雪球"
