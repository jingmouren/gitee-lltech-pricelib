#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.utilities.enums import StatusType, CallPut
from pricelib.common.time import (global_evaluation_date, Schedule, CN_CALENDAR, AnnualDays)
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.mc_engines import MCAccumulatorEngine


class Accumulator(OptionBase, Observer):
    """累购累沽期权
    以累购为例：行权价通常低于二级市场标的现价，敲出障碍价通常高于二级市场标的现价。
             在每个交易日，如果标的价格在执行价之下，以行权价购买N份标的，N是杠杆比例，例如N=2；
                         如果标的价格在执行价之上，以行权价购买1份标的；
             在敲出观察日，如果标的价格在敲出障碍价之上，发生敲出，合约提前终止，结算收益。
             最终的收益即是累计的所有标的份数 x (结算时的标的价格 - 行权价)。
    累沽与累沽相反，行权价通常高于二级市场标的现价，敲出障碍价通常低于二级市场标的现价。
    """

    def __init__(self, s0, barrier_out=None, strike=None, leverage_ratio=None, obs_dates=None, margin_lvl=0.2,
                 engine=None, status=StatusType.NoTouch, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            s0: float，标的初始价格
            barrier_out: float，敲出障碍价
            strike: float，行权价
            leverage_ratio: float, 杠杆比例，累购期权下，当标的价格在执行价之下时，购买leverage_ratio份标的；
                                                    当标的价格在执行价之上、敲出价之下时，购买1份标的
            margin_lvl: float，保证金比例，对于定价没有影响
            status: 敲出状态，默认为StatusType.NoTouch表示未敲出
            engine: 定价引擎，PricingEngine类，仅支持Monte Carlo模拟 MCAccumulatorEngine
        时间参数: 要么输入年化期限，要么输入起始日和到期日；敲出观察日可缺省
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            obs_dates: List[datetime.date]，敲出观察日，可缺省，缺省时会自动生成每月观察的敲出日期序列(已根据节假日调整)
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
        self.barrier_out = barrier_out  # 敲出线
        self.strike = strike  # 行权价
        if barrier_out > strike:  # 累购
            self.callput = CallPut.Call
        else:  # 累沽
            self.callput = CallPut.Put
        if obs_dates is None:  # 观察日列表
            self.obs_dates = Schedule(trade_calendar=trade_calendar, start=self.start_date, end=self.end_date,
                                      freq='m', lock_term=0)  # 累购累沽无锁定期
            self.end_date = self.obs_dates.end  # 修正结束时间
        else:
            self.obs_dates = Schedule(trade_calendar=trade_calendar, date_schedule=obs_dates)
        self.leverage_ratio = leverage_ratio
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            self.engine = MCAccumulatorEngine(s=s, r=r, q=q, vol=vol)


    def set_pricing_engine(self, engine):
        """切换定价引擎"""
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
        """收到标的价格变化的通知时，自动更新是否已经敲出"""
        if observable == self.engine.process.spot:
            if self.callput == CallPut.Call and observable.value >= self.barrier_out:
                self.status = StatusType.UpTouch
            elif self.callput == CallPut.Put and observable.value <= self.barrier_out:
                self.status = StatusType.DownTouch

    def __repr__(self):
        """返回期权的描述"""
        if self.barrier_out >= self.strike:
            return "累购(Accumulator)"
        return "累沽(Decumulator)"

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
