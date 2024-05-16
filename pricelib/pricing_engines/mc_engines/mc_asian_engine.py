#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import AverageMethod, AsianAveSubstitution
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCAsianEngine(McEngine):
    """亚式期权 Monte Carlo 模拟定价引擎"""

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
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        obs_start = prod.trade_calendar.business_days_between(calculate_date, prod.obs_start)
        obs_end = prod.trade_calendar.business_days_between(calculate_date, prod.obs_end)

        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()
        r = self.process.interest(_maturity)
        obs_steps = np.arange(obs_start, obs_end, 1, dtype=int)
        if prod.substitute == AsianAveSubstitution.Underlying:
            if prod.enhanced:
                paths[obs_steps, :] = paths[obs_steps, :] + prod.callput.value * np.maximum(
                    prod.callput.value * (prod.limited_price - paths[obs_steps, :]), 0)
            if prod.ave_method == AverageMethod.Arithmetic:
                ave_s = np.mean(paths[obs_steps, :], axis=0)
            elif prod.ave_method == AverageMethod.Geometric:
                ave_s = np.exp(np.mean(np.log(paths[obs_steps, :]), axis=0))
            else:
                raise ValueError("无效的平均价计算方法，只支持几何平均/算术平均")
            price = np.mean(np.maximum(prod.callput.value * (ave_s - prod.strike), 0)) * np.exp(-r * _maturity)
            return price
        if prod.substitute == AsianAveSubstitution.Strike:
            if prod.ave_method == AverageMethod.Arithmetic:
                ave_s = np.mean(paths[obs_steps, :], axis=0)
            elif prod.ave_method == AverageMethod.Geometric:
                ave_s = np.exp(np.mean(np.log(paths[obs_steps, :]), axis=0))
            else:
                raise ValueError("无效的平均价计算方法，只支持几何平均/算术平均")
            price = np.mean(np.maximum(prod.callput.value * (paths[-1] - ave_s), 0)) * np.exp(-r * _maturity)
            return price
        raise ValueError("无效的substitute类型，只支持替代标的资产到期价格/替代行权价")
