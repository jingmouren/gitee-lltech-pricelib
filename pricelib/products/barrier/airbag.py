#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.utilities.enums import CallPut, InOut, UpDown, ExerciseType, EngineType, PaymentType, StatusType
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.fdm_engines import FdmAirbagEngine


class Airbag(OptionBase, Observer):
    """安全气囊
    安全气囊是美式观察向下敲入看涨期权，到期支付
        敲入前是call，执行价等于合约设立时的标的期初价格
        敲入后payoff与持有标的资产相同(在上涨和下跌参与率为1的情况下)"""
    updown = UpDown.Down
    inout = InOut.In
    callput = CallPut.Call
    payment_type = PaymentType.Expire
    exercise_type = ExerciseType.American

    def __init__(self, strike, barrier, *, knockin_parti=1, call_parti=0.8, reset_call_parti=1, engine=None,
                 status=StatusType.NoTouch, maturity=None, start_date=None, end_date=None, discrete_obs_interval=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            strike:  float, 执行价
            barrier:  float, 敲入障碍价格
            knockin_parti: float, 敲入的下跌参与率
            call_parti: float, 未敲入的看涨参与率
            reset_call_parti: float, 敲入的重置后看涨参与率
            status: 敲入状态，StatusType枚举类，默认为NoTouch未敲入未敲出，DownTouch为已敲入
            engine: 定价引擎，PricingEngine类
                    解析解: AnalyticAirbagEngine 由障碍与二元解析解(资产或无)组合而成，所以敲入的下跌参与率与重置后看涨参与率必须相等。
                                                支持连续观察/离散观察(默认为每日观察)
                    PDE: FdmAirbagEngine 支持连续观察/离散观察(默认为每日观察)
                    蒙特卡洛: MCAirbagEngine 只支持离散观察(默认为每日观察)
        时间参数: 要么输入年化期限，要么输入起始日和到期日
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
            discrete_obs_interval: 观察时间间隔. 若为连续观察，None；若为均匀离散观察，为年化的观察时间间隔
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用 PDE 定价引擎 FdmAirbagEngine
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__()
        self.trade_calendar = trade_calendar
        self.annual_days = annual_days
        self.t_step_per_year = t_step_per_year
        assert maturity is not None or (start_date is not None and end_date is not None), "Error: 到期时间或起止时间必须输入一个"
        self.start_date = start_date if start_date is not None else global_evaluation_date()  # 起始时间
        self.end_date = end_date if end_date is not None else (
                self.start_date + datetime.timedelta(days=round(maturity * annual_days.value)))  # 结束时间
        self.maturity = maturity if maturity is not None else (self.end_date - self.start_date).days / annual_days.value
        self.strike = strike
        self.barrier = barrier
        self.knockin_parti = knockin_parti
        self.call_parti = call_parti
        self.reset_call_parti = reset_call_parti
        self.discrete_obs_interval = discrete_obs_interval  # 连续观察=None；均匀离散观察=观察时间间隔
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = FdmAirbagEngine(s=s, r=r, q=q, vol=vol)
            self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """设置定价引擎，同时将自己注册为观察者。若已有定价引擎，先将自己从原定价引擎的观察者列表中移除"""
        with suppress(AttributeError, ValueError):
            self.engine.process.spot.remove_observer(self)
        self.engine = engine
        self.engine.process.spot.add_observer(self)
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def remove_self(self):
        """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
        self.engine.process.spot.remove_observer(self)
        del self

    def update(self, observable, *args, **kwargs):
        """收到标的价格变化的通知时，自动更新是否已经触碰障碍，默认每日观察"""
        if observable == self.engine.process.spot:
            if observable.value <= self.barrier:
                self.status = StatusType.DownTouch

    def __repr__(self):
        """返回期权的描述"""
        return "安全气囊"

    @time_this
    def price(self, t=None, spot=None):
        self.validate_parameters(t=t)
        if self.engine.engine_type == EngineType.PdeEngine:  # 接口特殊是因为PDE引擎兼容了单双边障碍
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot, bound=(self.barrier, None),
                                                   rebate=(0, 0))
        else:
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price
