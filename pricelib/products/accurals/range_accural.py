#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.time import (global_evaluation_date, CN_CALENDAR, AnnualDays)
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.mc_engines import MCRangeAccuralEngine


class RangeAccural(OptionBase, Observer):
    """区间累计期权"""

    def __init__(self, s0, upper_strike=None, lower_strike=None, payment=0.1, engine=None, status=0,
                 maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR,
                 annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            s0: float，标的初始价格
            upper_strike: float，高行权价
            lower_strike: float，低行权价
            payment: float，区间内行权收益率，百分比
            engine: 定价引擎，PricingEngine类，仅支持Monte Carlo模拟 MCRangeAccuralEngine
            status: int，之前的累计次数，期初时是0
        时间参数: 要么输入年化期限，要么输入起始日和到期日
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用蒙特卡洛模拟定价引擎 (BSM模型)
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__()
        self.s0 = s0  # 标的初始价格
        self.trade_calendar = trade_calendar  # 交易日历
        self.annual_days = annual_days
        self.t_step_per_year = t_step_per_year
        assert maturity is not None or (start_date is not None and end_date is not None), "Error: 到期时间或起止时间必须输入一个"
        self.start_date = start_date if start_date is not None else global_evaluation_date()  # 起始时间
        self.end_date = end_date if end_date is not None else (
                self.start_date + datetime.timedelta(days=round(maturity * annual_days.value)))  # 结束时间
        if maturity is None:
            self.maturity = (end_date - start_date).days / annual_days.value
        else:
            self.maturity = maturity
        self.upper_strike = upper_strike
        self.lower_strike = lower_strike
        self.payment = payment  # 区间内行权收益率，百分比
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            self.engine = MCRangeAccuralEngine(s=s, r=r, q=q, vol=vol)

    def set_pricing_engine(self, engine):
        with suppress(AttributeError):
            self.engine.process.spot.remove_observer(self)
        self.engine = engine
        self.engine.process.spot.add_observer(self)
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def remove_self(self):
        """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
        self.engine.process.spot.remove_observer(self)
        del self

    def update(self, observable, *args, **kwargs):
        """收到标的价格变化的通知时，自动更新累计期权的累计次数"""
        if observable == self.engine.process.spot and self.lower_strike <= observable.value <= self.upper_strike:
            self.status += 1
            logging.info(f"{self}当前累计次数为{self.status}")

    def __repr__(self):
        """返回期权的描述"""
        return "区间累计(Range Accural)"

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        self.validate_parameters(t=t)
        price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price
