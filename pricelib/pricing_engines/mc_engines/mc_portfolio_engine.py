#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCPortfolioEngine(McEngine):
    """香草组合 Monte Carlo 模拟定价引擎"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        _maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径
        price = 0
        n_step = round(_maturity * prod.t_step_per_year)
        paths = self.path_generator(n_step=n_step, spot=spot, t_step_per_year=prod.t_step_per_year).copy()
        for vanilla, position in prod.vanilla_list:
            price += np.mean(np.maximum(vanilla.callput.value * (paths[-1, :] - vanilla.strike),
                                        0)) * position * self.process.interest.disc_factor(_maturity)
        return price
