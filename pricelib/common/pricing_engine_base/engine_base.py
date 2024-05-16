#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta, abstractmethod
import datetime
from ..utilities.enums import EngineType
from ..utilities.patterns import SimpleQuote
from ..utilities.utility import logging
from ..term_structures import ConstantRate
from ..volmodels import BlackConstVol
from ..processes import StochProcessBase, GeneralizedBSMProcess


class PricingEngineBase(metaclass=ABCMeta):
    """定价引擎基类"""
    engine_type: EngineType = None  # 定价引擎类型，枚举类

    def __init__(self, stoch_process=None, s=None, r=None, q=None, vol=None):
        """构造函数，若stoch_process不为None，优先设置随机过程为stoch_process，忽略s、r、q、vol参数；
                   若stoch_process=None，默认构造BSM随机过程，需要输入标的价格、无风险利率、分红/融券率、波动率；
                   若未设置随机过程，也未提供构造随机过程的参数，也可以后续通过set_stoch_process方法设置随机过程
        Args:
            stoch_process: 随机过程StochProcessBase对象
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        if stoch_process is not None:
            self.set_stoch_process(stoch_process)
        elif s is not None and r is not None and q is not None and vol is not None:
            spot_price = SimpleQuote(value=s, name="标的价格")
            riskfree = ConstantRate(value=r, name="无风险利率")
            dividend = ConstantRate(value=q, name="分红/融券率")
            volatility = BlackConstVol(vol, name="波动率")
            self.process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
        else:
            self.process = None
            logging.warning(
                "构造定价引擎时，未设置随机过程，也未提供构造随机过程的参数，无法初始化随机过程。请调用set_stoch_process方法设置随机过程!")

    def set_stoch_process(self, stoch_process: StochProcessBase):
        """设置随机过程，其中包含了价格动态过程及波动率信息
        Args:
            stoch_process: 随机过程StochProcessBase对象
        """
        self.process = stoch_process

    @abstractmethod
    def calc_present_value(self, prod, t: datetime.date = None, spot: float = None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
