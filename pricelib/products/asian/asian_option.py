#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib.common.product_base.option_base import OptionBase
from ...common.utilities.enums import CallPut, AverageMethod, AsianAveSubstitution
from ...common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from ...common.utilities import time_this, logging
from pricelib.pricing_engines.analytic_engines import AnalyticAsianEngine
from pricelib.pricing_engines.mc_engines import MCAsianEngine


class AsianOption(OptionBase):
    """
    亚式期权：几何平均/算术平均；替代标的资产价格/替代执行价；增强亚式
        解析解只支持平均结算价，无增强亚式
        蒙特卡洛支持几何平均/算术平均，支持增强亚式，支持替代标的资产价格/替代执行价
        二叉树支持几何平均/算术平均，支持增强亚式，支持替代标的资产价格，不支持替代执行价
        todo:若定价时已经处于均价观察期内，需要获得当前已经实现的均价，并对行权价进行调整
    """

    def __init__(self, callput: CallPut, ave_method: AverageMethod, strike=None,
                 substitute=AsianAveSubstitution.Underlying, parti=1, enhanced=False, limited_price=None,
                 engine=None, maturity=None, start_date=None, end_date=None, obs_start=None, obs_end=None,
                 s_average=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """亚式期权初始化
        产品参数:
            callput: 看涨看跌类型，CallPut枚举类，Call/Put
            ave_method: 平均价计算方法，AverageMethod枚举类，几何平均Geometric/算术平均Arithmetic
            strike: float, 执行价
            substitute: 以均价代替标的资产价格/替代行权价，AsianAveSubstitution枚举类，Underlying/Strike
            parti: float, 参与率，默认为1
            enhanced: bool，是否为增强亚式期权，默认为False
            limited_price: float，增强型亚式限定标的资产价格的上/下限
            s_average: float，已实现的资产价格平均值(已经处于资产价格平均期内时)，默认为None
            engine: 定价引擎，PricingEngine类
                    解析解: AnalyticAsianEngine
                           几何平均替代结算价亚式: Kemma Vorst (1990) 连续观察，平均价观察期为整个期权有效期内
                           算术平均替代结算价亚式: Turnbull Wakeman (1991) 近似估计，平均观察期的起点任意，终点为期权到期日
                    二叉树: BiTreeAsianEngine 只支持算术平均替代结算价亚式
                    蒙特卡洛: MCAsianEngine
                           几何平均/算术平均替代结算价亚式
                           几何平均/算术平均替代执行价亚式
                           增强亚式
        时间参数: 要么输入年化期限，要么输入起始日和到期日
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            obs_start: datetime.date，观察期起始日
            obs_end: datetime.date，观察期终止日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则平均替代收盘价亚式默认使用解析解定价引擎；平均替代执行价亚式和增强亚式默认使用蒙特卡洛定价引擎
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
        self.obs_start = obs_start if obs_start is not None else self.start_date
        self.obs_end = obs_end if obs_end is not None else self.end_date
        self.parti = parti
        self.enhanced = enhanced
        self.limited_price = limited_price
        if enhanced:
            assert limited_price is not None, "Error: 增强型亚式必须输入限定标的价格上/下限参数"
        assert strike is not None or substitute == AsianAveSubstitution.Strike, "Error: 替代平均结算价，必需输入执行价参数"
        self.strike = strike
        self.callput = callput
        self.ave_method = ave_method
        self.substitute = substitute
        self.s_average = s_average
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            if substitute == AsianAveSubstitution.Underlying:
                default_engine = AnalyticAsianEngine(s=s, r=r, q=q, vol=vol)
                self.set_pricing_engine(default_engine)
            else:  # 其他情形：替代执行价亚式/增强亚式
                default_engine = MCAsianEngine(s=s, r=r, q=q, vol=vol)
                self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """设置定价引擎"""
        self.engine = engine
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def __repr__(self):
        """返回期权的描述"""
        enhance_str = "增强型" if self.enhanced else ""
        return f"{self.ave_method.value}-{self.substitute.value}-{self.callput.name}{enhance_str}亚式期权"

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        self.validate_parameters(t=t)
        price = self.engine.calc_present_value(prod=self, t=t, spot=spot) * self.parti
        return price
