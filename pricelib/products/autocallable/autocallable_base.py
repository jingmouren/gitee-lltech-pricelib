#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import bisect
from contextlib import suppress
import datetime
import numpy as np
from pricelib.common.utilities.enums import StatusType, ExerciseType
from pricelib.common.time import (global_evaluation_date, Schedule, CN_CALENDAR, AnnualDays)
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.fdm_engines import FdmSnowBallEngine, FdmPhoenixEngine


class AutocallableBase(OptionBase, Observer):
    """自动赎回结构定价基类：二元小雪球Autocall Note(无敲入)/雪球式Snowball Autocallable/触发器Trigger Autocallable
                        二元小雪球：无敲入（敲入价为-1），有红利票息（未敲出有保底收益），为保本产品
                        雪球式：有敲入，敲入后下跌部分保护，敲出年化票息
                        触发器：有敲入，敲入后下跌部分保护，敲出绝对票息
       支持变敲出、变票息、变敲入、雪球plus、保底雪球、OTM、触发式等多种结构"""

    def __init__(self, s0, barrier_out=None, barrier_in=None, coupon_out=None, coupon_div=None, parti_out=0,
                 parti_in=1, margin_lvl=1, strike_upper=None, strike_lower=0, strike_call=None, obs_dates=None,
                 pay_dates=None, lock_term=1, trigger=False, in_obs_type=ExerciseType.American,
                 status=StatusType.NoTouch, engine=None, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            s0: float，标的初始价格
            barrier_out: List[float]，敲出障碍价，绝对值/百分比，长度需要与敲出观察日数一致
                         若敲出价不是常数，为阶梯式雪球（降敲）
            barrier_in: List[float]，敲入障碍价，绝对值/百分比，仅雪球有，值大于0时非保本，值为0时保本
                        若敲入价不是常数，为变敲入型雪球，每个敲出观察日对应一个敲入价，在这个敲出观察日和上个敲出观察日之间，敲入价为该敲入价
            coupon_out: List[float]，敲出票息，百分比，年化
                        数组list，长度需要与敲出观察日数一致，为阶梯式浮动票息（降息或双降）雪球，主流降票息结构为早利(2年)和蝶变(3年)
            coupon_div: float，百分比，年化，雪球为红利票息；Autocall Note为未敲出票息
                        是持有到期未敲入未敲出的票息，与敲出票息分开，为数值型，不能是数组，默认为None，即与最后一个敲出票息相同
            parti_out: float，看涨结构参与率
            parti_in: float，敲入后参与率，限损
            margin_lvl: float，保证金比例，默认为1，即无杠杆
            strike_upper: float，敲入发生后(熊市价差)的高行权价，即OTM行权价(<s0)，普通雪球敲入发生后的高行权价为期初价格s0
            strike_lower: float，敲入发生后(熊市价差)的低行权价，即保底边界，普通雪球敲入发生后的低行权价为0
                                对于雪球和触发器，为保底价，越高提供的保护越多，值为0时不保底
            strike_call: float，敲出价之上的看涨结构行权价
            lock_term: int，锁定期，单位为月，锁定期内不触发敲出
            trigger: bool，是否为触发器，即敲出票息是否年化，True为绝对值，False为百分比
            in_obs_type: 敲入观察类型，ExerciseType枚举类，默认为American每日观察；European为仅到期日观察是否敲入
            status: 敲入敲出状态，StatusType枚举类，默认为NoTouch, 即未敲入未敲出；UpTouch为已敲出，DownTouch为已敲入
            engine: 定价引擎，PricingEngine类
                    蒙特卡洛: MCAutoCallableEngine
                    PDE: FdmSnowBallEngine
                    积分法: QuadSnowballEngine
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
            则默认使用PDE定价引擎 FdmSnowBallEngine
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
        self.barrier_in = barrier_in  # 敲入线
        self.coupon_out = coupon_out  # 敲出票息
        self.coupon_div = coupon_out if coupon_div is None else coupon_div  # 红利票息
        self.parti_out = parti_out  # 上涨参与率
        self.parti_in = parti_in  # 下跌参与率
        self.margin_lvl = margin_lvl  # 预付金比例
        self.strike_upper = s0 if strike_upper is None else strike_upper  # 敲入后高执行价
        self.strike_lower = strike_lower  # 低行权价
        self.strike_call = s0 if strike_call is None else strike_call  # 敲出看涨执行价
        if obs_dates is None:  # 观察日列表
            self.obs_dates = Schedule(trade_calendar=trade_calendar, start=self.start_date, end=self.end_date,
                                      freq='m', lock_term=lock_term)
            self.end_date = self.obs_dates.end  # 修正结束时间
        else:
            self.obs_dates = Schedule(trade_calendar=trade_calendar, date_schedule=obs_dates)
        if pay_dates is None:  # 付息日
            self.pay_dates = self.obs_dates
        else:
            self.pay_dates = Schedule(trade_calendar=trade_calendar, date_schedule=pay_dates)
        self.lock_term = lock_term  # 锁定期
        self.trigger = trigger  # 是否是触发式
        self.status = status
        self.in_obs_type = in_obs_type  # 敲入观察为欧式，到期观察敲入；美式，每日观察敲入
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = FdmSnowBallEngine(s=s, r=r, q=q, vol=vol)
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
        """收到标的价格变化的通知时，自动更新自动赎回结构是否已经敲入或敲出"""
        if observable == self.engine.process.spot:
            calculate_date = global_evaluation_date()
            if calculate_date in self.obs_dates:  # 如果估值日是敲出观察日
                obs_idx = self.obs_dates.date_schedule.index(calculate_date)
                if observable.value >= self.barrier_out[obs_idx]:  # 如果标的价格大于等于敲出价，敲出
                    self.status = StatusType.UpTouch
                    return
            if self.status in [StatusType.DownTouch, StatusType.UpTouch]:  # 如果已敲入/已敲出，维持原来的状态
                return
            if self.status == StatusType.NoTouch:  # 如果未敲入未敲出
                obs_idx = bisect.bisect_left(self.obs_dates.date_schedule, global_evaluation_date())
                if observable.value <= self.barrier_in[obs_idx]:  # 且标的价格小于等于敲入价，敲入
                    self.status = StatusType.DownTouch
                    return

    def __repr__(self):
        """返回期权的描述"""
        return "Autocallable自动赎回结构"

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


class PhoenixBase(OptionBase, Observer):
    """FCN/DCN/凤凰结构定价基类
        凤凰类 AutoCallable
            较雪球式AutoCallable，增加派息价格，少了敲出和红利票息
            每个派息（敲出）观察日，如果价格高于派息价格，派固定利息，如果发生敲出，合约提前结束；
            发生敲入后，派息方式不变，到期如果未敲出，结构为看跌空头或熊市价差空头
        FCN(Fixed Coupon Note, 固定派息票据)/DCN(Digital Coupon Note, 二元派息票据)/Phoenix凤凰票据
            FCN：无派息价格（派息价格为0），每月固定派息；到期日观察是否敲入
            DCN：有派息价格，每个观察日如果价格高于派息价格，派固定利息；到期日观察是否敲入
            Phoenix：有派息价格，每个观察日如果价格高于派息价格，派固定利息；每日观察是否敲入
    """

    def __init__(self, s0,
                 barrier_out=None, barrier_yield=None, coupon=None, barrier_in=None, parti_in=1, margin_lvl=1,
                 strike_upper=None, strike_lower=0, obs_dates=None, pay_dates=None, lock_term=1,
                 in_obs_type=ExerciseType.American, status=StatusType.NoTouch, engine=None, maturity=None,
                 start_date=None, end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365,
                 t_step_per_year=243, s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            s0: float，标的初始价格
            barrier_out: List[float]，敲出障碍价，绝对值/百分比，长度需要与敲出观察日数一致
                         若不是常数，为阶梯式凤凰（降敲）
            barrier_yield: List[float]，派息边界价格，绝对值/百分比，长度需要与派息观察日数一致
                           凤凰和DCN，barrier_yield一般等于敲入价格；FCN每月固定派息，barrier_yield等于0
            coupon: List[float]，派息票息，百分比，非年化
                    数组list，长度需要与派息观察日数一致，为阶梯式浮动票息（降息或双降）凤凰
            barrier_in: List[float]，敲入障碍价，绝对值/百分比
                    若不是常数，为变敲入型雪球，每个敲出观察日对应一个敲入价，在这个敲出观察日和上个敲出观察日之间，敲入价为该敲入价
                    对于FCN，进到期日观察是否敲入，到期日前的敲入价都是0
            parti_in: float，敲入后参与率，限损
            margin_lvl: float，保证金比例，默认为1，即无杠杆
            strike_upper: float，敲入发生后(熊市价差)的高行权价，即OTM行权价(<s0)，普通凤凰敲入发生后的高行权价为期初价格s0
            strike_lower: float，敲入发生后(熊市价差)的低行权价，即保底边界，越高提供的保护越多，值为0时不保底
            lock_term: int，锁定期，单位为月，锁定期内不触发敲出
            status: 敲入敲出状态，StatusType枚举类，默认为NoTouch，已敲入未敲出为DownTouch，已敲出为UpTouch
            in_obs_type: 敲入观察类型，ExerciseType枚举类，默认为American每日观察；European为仅到期日观察是否敲入
            engine: 定价引擎，PricingEngine类
                    蒙特卡洛: MCPhoenixEngine
                    PDE: FdmPhoenixEngine
                    积分法: QuadFCNEngine
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
            则默认使用PDE定价引擎 FdmSnowBallEngine
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__()
        self.s0 = s0  # 标的初始价格
        self.trade_calendar = trade_calendar  # 交易日历
        self.annual_days = annual_days
        assert maturity is not None or (start_date is not None and end_date is not None), "Error: 到期时间或起止时间必须输入一个"
        self.start_date = start_date if start_date is not None else global_evaluation_date()  # 起始时间
        self.end_date = end_date if end_date is not None else (
                self.start_date + datetime.timedelta(days=round(maturity * annual_days.value)))  # 结束时间
        if maturity is None:
            self.maturity = (end_date - start_date).days / annual_days.value
        else:
            self.maturity = maturity
        len_months = round(self.maturity * 12)
        # obs_dates无锁定期，因为迭代过程中paydates和barrier in的改变都发生在敲出日obs_dates，锁定期靠敲出价+inf实现
        if obs_dates is None and pay_dates is None:  # 观察日列表
            self.obs_dates = self.pay_dates = Schedule(trade_calendar=trade_calendar, start=self.start_date,
                                                       end=self.end_date, freq='m', lock_term=1)
            self.end_date = self.obs_dates.end  # 修正结束时间
        elif obs_dates is not None and pay_dates is not None:
            assert len(pay_dates) == len_months, "Error: 输入派息日数组长度不等于存续期月份数"
            assert len(obs_dates) == len_months - lock_term + 1, "Error: 输入敲出观察日数组长度不等于存续期月份数减去锁定期"
            # todo: 区分敲出观察日和付息观察日，以及敲出支付日和付息日
            self.obs_dates = Schedule(trade_calendar=trade_calendar,
                                      date_schedule=pay_dates[:lock_term - 1] + obs_dates)
            self.pay_dates = Schedule(trade_calendar=trade_calendar, date_schedule=pay_dates)
        else:
            raise ValueError("观察日和付息日必须同时输入或同时不输入")
        self.t_step_per_year = t_step_per_year

        self.lock_term = lock_term  # 锁定期
        # 敲出线
        if barrier_out is not None:
            assert isinstance(barrier_out, (list, np.ndarray)) and len(barrier_out) == len_months, \
                f"Error: 输入的敲出线必须是与存续月份数{len_months}等长的list列表或np.adarray数组"
            self.barrier_out = np.array(barrier_out).astype(float)
            self.barrier_out[:(lock_term - 1)] = np.inf  # 锁定期敲出价格 为+inf
        else:
            self.barrier_out = barrier_out
        self.barrier_in = barrier_in  # 敲入线
        self.barrier_yield = barrier_yield  # 派息边界价格
        self.coupon = coupon  # 票息
        if coupon is not None:
            assert isinstance(coupon, (list, np.ndarray)) and len(coupon) == len_months, \
                f"Error: 输入的票息必须是与存续月份数{len_months}等长的list列表或np.adarray数组"
        self.parti_in = parti_in  # 下跌参与率
        self.margin_lvl = margin_lvl  # 预付金比例
        self.strike_upper = s0 if strike_upper is None else strike_upper  # 敲入后高执行价
        self.strike_lower = strike_lower  # 低行权价
        self.status = status
        self.in_obs_type = in_obs_type  # 敲入观察为欧式，到期观察敲入；美式，每日观察敲入
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = FdmPhoenixEngine(s=s, r=r, q=q, vol=vol)
            self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
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
        """收到标的价格变化的通知时，自动更新自动赎回结构是否已经敲入或敲出"""
        if observable == self.engine.process.spot:
            # TODO: 目前默认敲入敲出是常数，尚未支持敲入敲出线可变
            if isinstance(self.barrier_in, (int, float)) and observable.value <= self.barrier_in:
                self.status = StatusType.DownTouch
            elif isinstance(self.barrier_out, (int, float)) and observable.value >= self.barrier_out:
                self.status = StatusType.UpTouch

    def __repr__(self):
        """返回期权的描述"""
        return "FCN/DCN/Phoenix"

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
