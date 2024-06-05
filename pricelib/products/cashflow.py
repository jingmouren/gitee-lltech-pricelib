#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from dateutil import rrule
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.utility import time_this
from pricelib.common.product_base import OptionBase
from pricelib.pricing_engines.analytic_engines.cashflow_engine import CashFlowEngine

__all__ = ['CashFlow']


class CashFlow(OptionBase):
    """固定现金流现值，相当于零息债券现值。"""

    def __init__(self, payment_date=None, cashflow=None, *, engine=None, trade_calendar=CN_CALENDAR,
                 annual_days=AnnualDays.N365, r=None):
        super().__init__()
        self.trade_calendar = trade_calendar
        self.annual_days = annual_days
        self.payment_date = payment_date
        self.cashflow = cashflow
        if engine is not None:
            self.set_pricing_engine(engine)
        elif r is not None:
            default_engine = CashFlowEngine(s=cashflow, r=r)
            self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """设置定价引擎"""
        self.engine = engine

    def __repr__(self):
        """返回期权的描述"""
        return "固定现金流现值"

    @time_this
    def price(self, t: datetime.date = None):
        """计算现金流现值
        Args:
            t: datetime.date，计算期权价格的日期
        Returns: 现金流现值
        """
        calculate_date = global_evaluation_date() if t is None else t

        n_year = rrule.rrule(rrule.YEARLY, dtstart=calculate_date, until=self.payment_date).count()
        n_years_later = calculate_date.replace(year=calculate_date.year + n_year)
        n_natural_days_per_year = (n_years_later - calculate_date).days / n_year

        maturity = (self.payment_date - calculate_date).days / n_natural_days_per_year
        self.engine.get_product_params(maturity, self.cashflow)
        price = self.engine.calc_present_value(self)
        return price

    def delta(self, *args, **kwargs):
        delta = 0
        return delta

    def gamma(self, *args, **kwargs):
        gamma = 0
        return gamma

    def vega(self, *args, **kwargs):
        vega = 0
        return vega

    def theta(self, t: datetime.date = None, *args, **kwargs):
        calculate_date = global_evaluation_date() if t is None else t
        theta = self.price() * -self.engine.process.interest(calculate_date)
        return theta

    def rho(self, *args, **kwargs):
        rho = 0
        return rho
