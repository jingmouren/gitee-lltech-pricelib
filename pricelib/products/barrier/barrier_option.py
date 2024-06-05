#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.utilities.enums import CallPut, InOut, UpDown, BarrierType, PaymentType, EngineType, StatusType
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.analytic_engines import AnalyticBarrierEngine


class BarrierOption(OptionBase, Observer):
    """单边障碍期权
    敲入期权在障碍被触及时才会生效，敲出期权在障碍被触及时立即失效"""

    def __init__(self, strike, barrier, rebate, callput: CallPut, inout: InOut, updown: UpDown, *, parti=1,
                 payment_type=None, engine=None, status=StatusType.NoTouch, maturity=None,
                 start_date=None, end_date=None, discrete_obs_interval=None, trade_calendar=CN_CALENDAR,
                 annual_days=AnnualDays.N365, t_step_per_year=243, s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            strike: float, 执行价
            barrier: float, 障碍价
            rebate: float, 敲出现金返还金额，或未敲入现金返还金额，绝对数值，非百分比，非年化
            callput: 看涨看跌类型，CallPut枚举类，Call/Put
            inout: 敲入敲出类型，InOut枚举类，In敲入/Out敲出
            updown: 向上向下类型，UpDown枚举类，Up/Down
            payment_type: 现金返还支付类型，PaymentType枚举类，敲入默认为到期支付Expire；敲出默认为立即支付Hit
            parti: float, 香草期权的参与率，默认为1
            status: 估值日前的敲入敲出状态，StatusType枚举类，默认为NoTouch未敲入/未敲出，UpTouch为向上敲入/敲出，DownTouch为向下敲入/敲出
            engine: 定价引擎，PricingEngine类，以下几种引擎均支持向上/向下、敲入/敲出、看涨/看跌8种单边障碍期权，支持敲出/未敲入现金返还
                            解析解和PDE引擎支持离散观察/连续观察，其余引擎只支持离散观察(默认为每日观察)
                    解析解: AnalyticBarrierEngine 敲入现金返还为到期支付；敲出现金返还为立即支付
                    蒙特卡洛: MCBarrierEngine  敲入现金返还为到期支付；敲出现金返还为到期支付 todo: 敲出现金返还立即支付
                    PDE: FdmBarrierEngine  敲入现金返还为到期支付；敲出现金返还 支持 立即支付/到期支付
                    积分法: QuadBarrierEngine  敲入现金返还为到期支付；敲出现金返还 支持 立即支付/到期支付
                    二叉树: BiTreeBarrierEngine 敲入现金返还为到期支付；敲出现金返还 支持 立即支付/到期支付
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
            则默认使用解析解定价引擎 AnalyticBarrierEngine - Merton(1973), Reiner & Rubinstein(1991a)
                                支持 Broadie, Glasserman和Kou(1995)均匀离散观察调整
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
        self.rebate = rebate  # 绝对值，敲出补偿，或未敲入补偿
        self.parti = parti
        self._callput = callput
        self._inout = inout
        self._updown = updown
        self.barrier_type = BarrierType.get_category(self._updown, self._inout, self._callput)
        if payment_type is None:
            if self._inout == InOut.In:
                self.payment_type = PaymentType.Expire
            else:
                self.payment_type = PaymentType.Hit
        else:
            self.payment_type = payment_type
            if self._inout == InOut.In:
                assert payment_type == PaymentType.Expire, "ValueError: 敲入期权的现金返还方式PaymentType一定是到期支付Expire"
        self.discrete_obs_interval = discrete_obs_interval  # 连续观察=None；均匀离散观察=观察时间间隔
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = AnalyticBarrierEngine(s=s, r=r, q=q, vol=vol)
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
        """收到标的价格变化的通知时，自动更新是否已经触碰障碍"""
        if observable == self.engine.process.spot:
            if self.discrete_obs_interval is None:  # TODO: 目前update仅支持连续观察，离散观察需要判断是否是观察日
                if observable.value >= self.barrier and self.updown == UpDown.Up:
                    self.status = StatusType.UpTouch
                elif observable.value <= self.barrier and self.updown == UpDown.Down:
                    self.status = StatusType.DownTouch

    def __repr__(self):
        """返回期权的描述"""
        if self.discrete_obs_interval is None:
            obs_type = "连续观察"
        else:
            obs_type = "离散观察"
        return f"{obs_type}{self.barrier_type.value.text}期权"

    @property
    def callput(self):
        return self._callput

    @callput.setter
    def callput(self, value):
        self._callput = value
        self.status = StatusType.NoTouch  # 产品参数改变，重置是否已经敲出或敲入
        self.barrier_type = BarrierType.get_category(self._updown, self._inout, self._callput)

    @property
    def updown(self):
        return self._updown

    @updown.setter
    def updown(self, value):
        self._updown = value
        self.status = StatusType.NoTouch  # 产品参数改变，重置是否已经敲出或敲入
        self.barrier_type = BarrierType.get_category(self._updown, self._inout, self._callput)

    @property
    def inout(self):
        return self._inout

    @inout.setter
    def inout(self, value):
        self._inout = value
        self.status = StatusType.NoTouch  # 产品参数改变，重置是否已经敲出或敲入
        self.barrier_type = BarrierType.get_category(self._updown, self._inout, self._callput)

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        self.validate_parameters(t=t)
        if self.engine.engine_type == EngineType.PdeEngine:  # 接口特殊是因为PDE引擎兼容了单双边障碍
            if self.updown == UpDown.Up:
                bound = (None, self.barrier)
            else:
                bound = (self.barrier, None)
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot, bound=bound,
                                                   rebate=(self.rebate, self.rebate))
        else:
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price
