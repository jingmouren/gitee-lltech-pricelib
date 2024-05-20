#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.utilities.enums import TouchType, EngineType, ExerciseType, PaymentType, StatusType
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.analytic_engines import AnalyticDoubleDigitalEngine
from pricelib.pricing_engines.fdm_engines import FdmDigitalEngine


class DoubleDigitalOption(OptionBase, Observer):
    """双边二元期权
        欧式: 二元凹式/二元凸式
        美式: 美式双接触/美式双不接触"""

    def __init__(self, touch_type: TouchType, exercise_type: ExerciseType, payment_type: PaymentType,
                 *, bound=(80, 120), rebate=(0, 0), engine=None, status=StatusType.NoTouch,
                 maturity=None, start_date=None, end_date=None, discrete_obs_interval=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            touch_type: 接触方式 - 接触Touch/不接触NotTouch
            exercise_type: 行权方式，ExerciseType枚举类，European/American
            payment_type: 支付方式，PaymentType枚举类，立即支付Hit/到期支付Expire
            bound: (float, float), (低障碍价, 高障碍价)
            rebate: (float, float), 触碰payoff的绝对数值，非年化
                    双接触: (下边界payoff, 上边界payoff); 双不接触: (到期不接触的payoff, 无作用)
            status: 定价时，触碰障碍的状态，StatusType枚举类，默认为NoTouch
            engine: 定价引擎，PricingEngine类
                    解析解: AnalyticDoubleDigitalEngine - Hui(1996) 级数近似解，双接触期权的下边界payoff和上边界payoff必须相等
                                                        支持 连续/离散观察、到期支付的(美式)双接触/双不接触
                    PDE: FdmDigitalEngine 支持 (欧式)二元凹式/二元凸式；支持 连续/离散观察、立即/到期支付的(美式)双接触/美式双不接触
                    蒙特卡洛: MCDoubleDigitalEngine 支持 (欧式)二元凹式/二元凸式；只支持离散观察、立即/到期支付的(美式)双接触/美式双不接触
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
            则使用默认定价引擎：
                到期支付的美式双不接触默认使用 Hui(1996) 级数近似解 AnalyticDoubleDigitalEngine
                其余情况默认使用 PDE 定价引擎 FdmDigitalEngine
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
        self.bound = bound  # (低障碍价, 高障碍价)
        self.rebate = rebate  # 双接触: (下边界payoff, 上边界payoff); 双不接触: (不接触payoff, 无作用)
        self.touch_type = touch_type
        self.exercise_type = exercise_type
        self.payment_type = payment_type
        self.discrete_obs_interval = discrete_obs_interval  # 连续观察=None；均匀离散观察=观察时间间隔
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            if (exercise_type == ExerciseType.American and payment_type == PaymentType.Expire
                    and touch_type == TouchType.NoTouch):
                default_engine = AnalyticDoubleDigitalEngine(s=s, r=r, q=q, vol=vol)
                self.set_pricing_engine(default_engine)
            else:
                default_engine = FdmDigitalEngine(s=s, r=r, q=q, vol=vol)
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
                if observable.value >= self.bound[1]:
                    self.status = StatusType.UpTouch
                elif observable.value <= self.bound[0]:
                    self.status = StatusType.DownTouch

    def __repr__(self):
        """返回期权的描述"""
        if self.exercise_type == ExerciseType.European:
            if self.touch_type == TouchType.Touch:
                return f"{self.payment_type.value}二元凹式(欧式双接触期权)"
            else:  # NoTouch
                return f"{self.payment_type.value}二元凸式(欧式双不接触期权)"
        elif self.exercise_type == ExerciseType.American:
            if self.discrete_obs_interval is None:
                obs_type = "连续观察"
            else:
                obs_type = "离散观察"
            return f"{self.payment_type.value}{obs_type}美式双{self.touch_type.value}二元期权"
        else:
            raise ValueError("无法识别的期权类型")

    @time_this
    def price(self, t=None, spot=None):
        if self.engine.engine_type == EngineType.PdeEngine:  # 接口特殊是因为PDE引擎兼容了单双边障碍
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot, bound=self.bound, rebate=self.rebate)
        else:
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price
