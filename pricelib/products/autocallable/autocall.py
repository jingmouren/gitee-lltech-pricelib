#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
import numpy as np
from pricelib.common.utilities.enums import CallPut
from pricelib.common.time import (AnnualDays, CN_CALENDAR)
from pricelib.common.utilities.utility import time_this, logging
from pricelib.pricing_engines.integral_engines import QuadAutoCallEngine
from .autocallable_base import AutocallableBase


class AutoCall(AutocallableBase):
    """Autocall Note(二元小雪球)，不带敲入的自动赎回结构，每月观察敲出，到期未敲出获得红利票息(保本产品)"""

    def __init__(self, s0, *, barrier_out, coupon_out, coupon_div=0, callput=CallPut.Call, lock_term=1,
                 maturity=None, start_date=None, end_date=None, obs_dates=None, pay_dates=None, margin_lvl=1,
                 engine=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            s0: float，标的初始价格
            barrier_out: float，敲出障碍价，绝对值/百分比
            coupon_out: float，敲出票息，百分比，年化
            coupon_div: float，红利票息，百分比，年化
            callput: 看涨看跌，CallPut枚举类，默认为Call
            lock_term: int，锁定期，单位为月，锁定期内不触发敲出
            margin_lvl:  float，保证金比例，默认为1，即无杠杆
            engine: 定价引擎，PricingEngine类
                    蒙特卡洛: MCAutoCallEngine
                    PDE: FdmSnowBallEngine
                    积分法: QuadAutoCallEngine
        时间参数: 要么输入年化期限，要么输入起始日和到期日；敲出观察日和票息支付日可缺省
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            obs_dates: List[datetime.date]，敲出观察日，可缺省，缺省时会自动生成每月观察的敲出日期序列(已根据节假日调整)
            pay_dates: List[datetime.date]，票息支付日，可缺省，长度需要与敲出观察日数一致，如不指定则默认为敲出观察日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用 积分法 定价引擎 QuadAutoCallEngine
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year, annual_days=annual_days)
        len_obs_dates = len(self.obs_dates.date_schedule)
        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div
        self.barrier_in = np.zeros(len_obs_dates)  # 小雪球无敲入
        self.callput = callput
        self.margin_lvl = margin_lvl  # 预付金比例
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = QuadAutoCallEngine(s=s, r=r, q=q, vol=vol)
            self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """设置定价引擎"""
        self.engine = engine
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def __repr__(self):
        """返回期权的描述"""
        return f"Auto{self.callput.name} Note(二元小雪球)"

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
