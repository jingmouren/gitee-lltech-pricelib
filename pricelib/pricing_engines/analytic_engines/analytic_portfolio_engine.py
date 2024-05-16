#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticPortfolioEngine(AnalyticEngine):
    """香草组合解析解定价引擎
    该类没有实现具体的计算方法，只是为了提供engine_type = EngineType.AnEngine的引擎类型，
    这样在计算时，会调用各个香草期权的解析解定价引擎
    """

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
