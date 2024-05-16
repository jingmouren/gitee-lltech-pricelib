#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import numpy as np
from pricelib.common.utilities.enums import StatusType
from pricelib.common.time import Schedule, CN_CALENDAR, AnnualDays
from pricelib.common.utilities.utility import time_this
from .autocallable_base import AutocallableBase


class ParisSnowball(AutocallableBase):
    """巴黎雪球产品类
    与经典雪球相比，敲出观察频率变为每周观察，敲入观察频率变为每月观察，并且累计敲入两次才算敲入"""

    def __init__(self, s0, barrier_out, barrier_in, coupon_out, lock_term=3, obs_dates=None, pay_dates=None,
                 coupon_div=None, knock_in_obs_dates=None, knock_in_times=2, knockout_freq="w", parti_in=1,
                 margin_lvl=1, status=StatusType.NoTouch, engine=None, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243):
        """继承自动赎回基类AutocallableBase的参数，详见AutocallableBase的__init__方法
        Args:
            knock_in_obs_dates: 敲入观察日期列表，List[datetime.date]，可缺省，缺省时会自动生成每月观察的敲入日期序列(已根据节假日调整)
            knock_in_times: 巴黎特性 - 累计敲入n次才算敲入
            knockout_freq: 敲出观察频率，"w"表示每周观察，"m"表示每月观察
        Returns: None
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, status=status, engine=engine,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, parti_in=parti_in,
                         margin_lvl=margin_lvl)
        # 巴黎雪球参数
        if knock_in_obs_dates is None:  # 敲入观察日列表，每月观察
            self.knock_in_obs_dates = Schedule(trade_calendar=trade_calendar, start=self.start_date, end=self.end_date,
                                               freq='m', lock_term=1)
            self.end_date = self.knock_in_obs_dates.end  # 修正结束时间
        else:
            self.knock_in_obs_dates = Schedule(trade_calendar=trade_calendar, date_schedule=knock_in_obs_dates)
        assert knockout_freq in ["w",
                                 "m"], f'Error: 敲出观察频率仅支持每周或每月观察，knockout_freq当前为{knockout_freq}，请填"w"或"m"!'
        self.knockout_freq = knockout_freq
        if obs_dates is None:  # 敲出观察日列表，每周观察/每周观察
            self.obs_dates = Schedule(trade_calendar=trade_calendar, start=self.start_date, end=self.end_date,
                                      freq=self.knockout_freq, lock_term=lock_term)
        else:
            self.obs_dates = Schedule(trade_calendar=trade_calendar, date_schedule=obs_dates)
        if pay_dates is None:  # 付息日
            self.pay_dates = self.obs_dates
        else:
            self.pay_dates = Schedule(trade_calendar=trade_calendar, date_schedule=pay_dates)

        self.knock_in_times = knock_in_times  # 多少次敲入视为敲入，默认为2
        # 平敲参数
        len_obs_dates = len(self.obs_dates.date_schedule)
        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.barrier_in = np.ones(len(self.knock_in_obs_dates.date_schedule)) * barrier_in
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div if coupon_div is not None else self.coupon_out[-1]
        self.parti_out = 0
        self.strike_upper = s0
        self.strike_lower = 0

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price

    def __repr__(self):
        """返回期权的描述"""
        return "巴黎雪球"
