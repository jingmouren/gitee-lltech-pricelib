#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib.common.utilities.enums import CallPut, ExerciseType
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase
from pricelib.pricing_engines.analytic_engines import AnalyticVanillaEuEngine, AnalyticVanillaAmEngine


class VanillaOption(OptionBase):
    """香草期权，包括欧式期权和美式期权
    均支持解析解、MC、PDE、Quad、二叉树"""

    def __init__(self, strike, callput=CallPut.Call, exercise_type=ExerciseType.European, *, engine=None,
                 maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365,
                 t_step_per_year=243, s=None, r=None, q=None, vol=None):
        """构造函数
        产品参数:
            strike: float，行权价
            callput: 看涨看跌类型，CallPut枚举类，Call/Put
            exercise_type: 行权方式，ExerciseType枚举类，European/American
            engine: 定价引擎，PricingEngine类
                    解析解: 欧式AnalyticVanillaEuEngine/美式AnalyticVanillaAmEngine
                    蒙特卡洛: MCVanillaEngine 欧式/美式
                    PDE: FdmVanillaEngine 欧式/美式
                    积分法: QuadVanillaEngine 欧式/美式
                    二叉树: BiTreeVanillaEngine 欧式/美式
        时间参数: 要么输入年化期限，要么输入起始日和到期日
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用解析解定价引擎 - (欧式期权BSM公式, 美式期权BAW近似解)
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__()
        self.trade_calendar = trade_calendar
        self.annual_days = annual_days
        assert maturity is not None or (start_date is not None and end_date is not None), "Error: 到期时间或起止时间必须输入一个"
        self.start_date = start_date if start_date is not None else global_evaluation_date()  # 起始时间
        self.end_date = end_date if end_date is not None else (
                self.start_date + datetime.timedelta(days=round(maturity * annual_days.value)))  # 结束时间
        self.maturity = maturity if maturity is not None else (self.end_date - self.start_date).days / annual_days.value
        self.strike = strike
        self.callput = callput
        self.exercise_type = exercise_type
        self.t_step_per_year = t_step_per_year
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            if exercise_type == ExerciseType.European:
                default_engine = AnalyticVanillaEuEngine(s=s, r=r, q=q, vol=vol)
                self.set_pricing_engine(default_engine)
            elif exercise_type == ExerciseType.American:
                default_engine = AnalyticVanillaAmEngine(s=s, r=r, q=q, vol=vol, an_method="BAW")
                self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """切换定价引擎"""
        self.engine = engine
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def __repr__(self):
        """返回期权的描述"""
        return f"{self.exercise_type.value}{self.callput.name}香草期权"

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
