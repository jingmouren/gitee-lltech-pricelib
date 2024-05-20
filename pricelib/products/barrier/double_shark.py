# !/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from contextlib import suppress
from pricelib.common.utilities.enums import InOut, ExerciseType, EngineType, PaymentType, StatusType
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.patterns import Observer
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.fdm_engines import FdmDoubleSharkEngine


class DoubleShark(OptionBase, Observer):
    """双鲨结构
    双鲨结构属于敲出期权，由两个高低障碍价相同的双边看涨、双边看跌敲出障碍期权组合而成，且put的执行价小于call的执行价"""
    inout = InOut.Out  # 双鲨结构属于敲出期权

    def __init__(self, exercise_type=ExerciseType.American, payment_type=PaymentType.Expire, strike=(90, 110),
                 bound=(80, 120), rebate=(0, 0), parti=(1, 1), engine=None, status=StatusType.NoTouch,
                 maturity=None, start_date=None, end_date=None, window=(None, None), discrete_obs_interval=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            exercise_type: 障碍观察类型，ExerciseType枚举类，美式观察American/欧式到期观察European
            payment_type: 支付类型，PaymentType枚举类，默认为到期支付Expire
            strike: (lower_strike, upper_strike), 低行权价/高行权价，一般低行权价是put的执行价，高行权价是call的执行价
            bound: (lower_barrier, upper_barrier), 低障碍价/高障碍价，障碍价格绝对数值。如果是默认值None，即为无穷大或无穷小inf。
            rebate: (lower_rebate, upper_rebate), 未敲出补偿/敲入补偿的绝对数值，非年化
                                                  对敲出期权，元组第一位是低障碍价下方的补偿，第二位是高障碍价上方的补偿，
                                                  对敲入期权，元组第一位是敲入补偿，第二位没有用处。
            parti: (put_parti, call_parti), 香草期权的参与率，默认为1
            status: 敲入敲出状态，StatusType枚举类，默认为NoTouch
            engine: 定价引擎，PricingEngine类
                    解析解: AnalyticDoubleSharkEngine 分为两种，只支持美式观察(整个有效期观察)；支持连续观察/离散观察(默认为每日观察)；
                                                    Ikeda and Kunitomo(1992) 仅当行权价在障碍范围内时，公式才是成立的
                                                    Haug(1998) 仅适用于标的持有成本为0的期权(期货期权)
                                    两者的现金返还部分都使用美式双接触闭式解Hui(1996)，因此高低敲出现金返还也必须相等，现金返还到期支付
                    PDE: FdmDoubleSharkEngine  只支持美式观察(整个有效期观察)；支持连续观察/离散观察(默认为每日观察)；
                                           支持现金返还；敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付
                    蒙特卡洛: MCDoubleSharkEngine  支持欧式观察(仅到期观察)/美式观察(整个有效期观察)；只支持离散观察(默认为每日观察)；
                                            支持现金返还；敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付
        时间参数: 要么输入年化期限，要么输入起始日和到期日
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            window: (start_date, end_date)，障碍观察时间窗口，距离估值日的年化期限。 仅PDE定价引擎支持该参数。
                    如果是默认值None，说明观察区间是整个存续期: 估值日-终止日
            discrete_obs_interval: 观察时间间隔. 若为连续观察，None；若为均匀离散观察，为年化的观察时间间隔
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用PDE定价引擎 FdmDoubleSharkEngine
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
        self.strike = strike  # (低行权价, 高行权价)
        self.bound = bound  # (低障碍价, 高障碍价)
        self.rebate = rebate  # 双接触: (下边界payoff, 上边界payoff); 双不接触: (不接触payoff, 无作用)
        self.window = window  # (观察窗口起始时间, 观察窗口结束时间)
        self.parti = parti  # (低参与率，高参与率)
        self.exercise_type = exercise_type
        self.payment_type = payment_type
        self.discrete_obs_interval = discrete_obs_interval  # 连续观察=None；均匀离散观察=观察时间间隔
        self.status = status
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = FdmDoubleSharkEngine(s=s, r=r, q=q, vol=vol)
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
        if self.discrete_obs_interval is None:
            obs_type = "连续观察"
        else:
            obs_type = "离散观察"
        return f"{obs_type}{self.payment_type.value}双鲨期权"


    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        if self.engine.engine_type == EngineType.PdeEngine:  # 接口特殊是因为PDE引擎兼容了单双边障碍
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot, bound=self.bound,
                                                   rebate=self.rebate, window=self.window)
        else:
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return price
