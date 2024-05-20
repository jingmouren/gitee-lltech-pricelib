#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.utilities.enums import EngineType, ExerciseType, CallPut
from pricelib.common.product_base.option_base import OptionBase
from pricelib.common.time import CN_CALENDAR, AnnualDays, global_evaluation_date
from ...pricing_engines.analytic_engines.analytic_vanilla_european_engine import AnalyticVanillaEuEngine
from .vanilla_option import VanillaOption
from .underlying import UnderlyingAsset


class VanillaPortfolio(OptionBase):
    """香草期权组合
    可以组合出价差、跨式、宽跨式、风险反转、领口、鹰式、蝶式等香草期权组合"""

    def __init__(self, maturity=None, start_date=None, end_date=None, trade_calendar=CN_CALENDAR,
                 annual_days=AnnualDays.N365, t_step_per_year=243, s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        初始化属性：
            underlying_list: 标的资产列表，[(underlying, position), ...]，其中underlying为UnderlyingAsset对象，position为持仓数量
            vanilla_list: 香草期权列表，[(vanilla, position), ...] ，其中vanilla为VanillaOption对象，position为持仓数量
        若提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用解析解定价引擎 (BSM公式)
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__()
        logging.info("创建香草期权组合")
        self.trade_calendar = trade_calendar
        self.annual_days = annual_days
        self.t_step_per_year = t_step_per_year
        assert maturity is not None or (start_date is not None and end_date is not None), "Error: 到期时间或起止时间必须输入一个"
        self.start_date = start_date if start_date is not None else global_evaluation_date()  # 起始时间
        self.end_date = end_date if end_date is not None else (
                self.start_date + datetime.timedelta(days=round(maturity * annual_days.value)))  # 结束时间
        self.maturity = maturity if maturity is not None else (self.end_date - self.start_date).days / annual_days.value

        self.underlying_list = []  # 标的资产列表
        self.vanilla_list = []  # 期权列表
        if s is not None and r is not None and q is not None and vol is not None:
            self.engine = AnalyticVanillaEuEngine(s=s, r=r, q=q, vol=vol)
            self.default_engine = True
        else:
            self.engine = None
            self.default_engine = False

    def add_underlying(self, s0, position=1.0):
        """添加标的资产
        Args:
            s0: float，标的资产期初价格
            position: float，标的资产头寸
        Returns:
        """
        current_underlying = UnderlyingAsset(s0)
        if self.default_engine:
            current_underlying.set_pricing_engine(self.engine)
        self.underlying_list.append((current_underlying, position))
        self.show_current_portfolio()

    def add_vanilla(self, strike, callput: CallPut, position, maturity=None, start_date=None, end_date=None):
        """添加香草期权, 仅支持欧式期权
        Args:
            strike: float, 行权价
            callput: 看涨看跌类型, CallPut枚举类
            position: float, 香草期权头寸
            maturity: float, 年化自然日期限
            start_date: datetime.date, 期权起始日期
            end_date: datetime.date, 期权到期日期
        Returns:
        """
        if maturity is not None and start_date is not None and end_date is not None:
            assert maturity == (end_date - start_date).days / self.annual_days.value, "Error: 到期时间与起止时间不匹配！"
        _maturity = maturity if maturity is not None else self.maturity
        _start_date = start_date if start_date is not None else self.start_date
        _end_date = end_date if end_date is not None else self.end_date
        current_vanilla = VanillaOption(strike=strike, maturity=_maturity, start_date=_start_date, end_date=_end_date,
                                        callput=callput, exercise_type=ExerciseType.European)
        if self.default_engine:
            current_vanilla.set_pricing_engine(self.engine)
        self.vanilla_list.append((current_vanilla, position))
        self.show_current_portfolio()

    def __str__(self):
        return "自定义香草组合"

    def __repr__(self):
        """返回期权的描述"""
        description = f"{str(self)} - 当前资产组合为：\n"
        for vanilla, position in self.vanilla_list:
            description += f"行权价为{vanilla.strike}的{vanilla.callput.name}, 期限为{vanilla.maturity}，头寸为：{position}\n"
        return description

    def show_current_portfolio(self):
        """展示当前组合的标的资产和香草期权信息"""
        logging.info(repr(self))

    def del_underlying(self, index):
        """删除第index个标的资产"""
        logging.info(f"删除第{index}个标的资产")
        del self.underlying_list[index - 1]
        self.show_current_portfolio()

    def del_vanilla(self, index):
        """删除第index个香草期权"""
        logging.info(f"删除第{index}个香草期权")
        del self.vanilla_list[index - 1]
        self.show_current_portfolio()

    def set_pricing_engine(self, engine):
        """设置定价引擎"""
        self.default_engine = False
        self.engine = engine
        process = self.engine.process
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")
        for underlying, position in self.underlying_list:
            underlying.set_pricing_engine(engine)
        if engine.engine_type == EngineType.AnEngine:
            engine = AnalyticVanillaEuEngine(process)
            for vanilla, position in self.vanilla_list:
                vanilla.set_pricing_engine(engine)

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        result = 0
        for asset, position in self.underlying_list:
            result += position * asset.price(t=t, spot=spot)
        if self.engine.engine_type in [EngineType.AnEngine, EngineType.PdeEngine]:
            for vanilla, position in self.vanilla_list:
                result += position * vanilla.price(t=t, spot=spot)
        else:
            result += self.engine.calc_present_value(prod=self, t=t, spot=spot)
        return result


class Spread(VanillaPortfolio):
    """价差期权"""

    def __init__(self, lower_strike, upper_strike, callput: CallPut, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.lower_strike = lower_strike
        self.upper_strike = upper_strike
        self.callput = callput
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)
        self.add_vanilla(strike=self.upper_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=-self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"牛市价差期权"
        return r"熊市价差期权"


class Butterfly(VanillaPortfolio):
    """蝶式期权"""

    def __init__(self, lower_strike, middle_strike, upper_strike, callput: CallPut, maturity=None, start_date=None,
                 end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.lower_strike = lower_strike
        self.middle_strike = middle_strike
        self.upper_strike = upper_strike
        self.callput = callput
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)
        self.add_vanilla(strike=self.middle_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=-2 * self.callput.value)
        self.add_vanilla(strike=self.upper_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"^蝶式期权"
        return r"∨蝶式期权"


class Strangle(VanillaPortfolio):
    """宽跨式"""

    def __init__(self, lower_strike, upper_strike, callput: CallPut, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.lower_strike = lower_strike
        self.upper_strike = upper_strike
        self.callput = callput
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)
        self.add_vanilla(strike=self.upper_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"v宽跨式"
        return r"^宽跨式"


class Straddle(VanillaPortfolio):
    """跨式"""

    def __init__(self, strike, callput: CallPut, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.strike = strike
        self.callput = callput
        self.add_vanilla(strike=self.strike, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Put,
                         position=self.callput.value)
        self.add_vanilla(strike=self.strike, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"v跨式"
        return r"^跨式"


class Condor(VanillaPortfolio):
    """鹰式"""

    def __init__(self, strike1, strike2, strike3, strike4, callput: CallPut, maturity=None, start_date=None,
                 end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.strike1 = strike1
        self.strike2 = strike2
        self.strike3 = strike3
        self.strike4 = strike4
        self.callput = callput
        self.add_vanilla(strike=self.strike1, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=self.callput.value)
        self.add_vanilla(strike=self.strike2, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=-self.callput.value)
        self.add_vanilla(strike=self.strike3, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=-self.callput.value)
        self.add_vanilla(strike=self.strike4, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"^鹰式"
        return r"v鹰式"


class DiscountCall(VanillaPortfolio):
    """折价看涨期权"""

    def __init__(self, strike, participate, callput=CallPut.Call, lower_strike=None, maturity=None, start_date=None,
                 end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.strike = strike
        if lower_strike is not None:
            self.lower_strike = lower_strike
        else:
            if callput == CallPut.Call:
                self.lower_strike = strike * participate
            elif callput == CallPut.Put:
                raise NotImplementedError("折价看跌期权不支持")
        self.participate = participate
        self.callput = callput
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)
        self.add_vanilla(strike=self.strike, start_date=self.start_date, end_date=self.end_date, callput=CallPut.Call,
                         position=-self.participate * self.callput.value)

    def __str__(self):
        return r"折价看涨"


class Collar(VanillaPortfolio):
    """领式"""

    def __init__(self, lower_strike, higher_strike, callput: CallPut, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.lower_strike = lower_strike
        self.higher_strike = higher_strike
        self.callput = callput
        self.add_underlying((self.lower_strike + self.higher_strike) / 2, 1)
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)
        self.add_vanilla(strike=self.higher_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=-self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"买入领式"
        return r"卖出领式"


class RiskReversal(VanillaPortfolio):
    """风险反转"""

    def __init__(self, lower_strike, higher_strike, callput: CallPut, maturity=None, start_date=None, end_date=None,
                 trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        super().__init__(maturity=maturity, start_date=start_date, end_date=end_date, trade_calendar=trade_calendar,
                         annual_days=annual_days, t_step_per_year=t_step_per_year, s=s, r=r, q=q, vol=vol)
        self.lower_strike = lower_strike
        self.higher_strike = higher_strike
        self.callput = callput
        self.add_vanilla(strike=self.lower_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=-self.callput.value)
        self.add_vanilla(strike=self.higher_strike, start_date=self.start_date, end_date=self.end_date,
                         callput=CallPut.Call, position=self.callput.value)

    def __str__(self):
        if self.callput == CallPut.Call:
            return r"/风险反转"
        return r"\风险反转"
