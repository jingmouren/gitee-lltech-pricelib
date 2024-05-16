#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.time import global_evaluation_date
from pricelib.common.utilities.enums import InOut, UpDown
from pricelib.common.pricing_engine_base import McEngine


class MCBarrierEngine(McEngine):
    """障碍期权 Monte Carlo 模拟定价引擎"""

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
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径
        r = self.process.interest(_maturity)
        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        if prod.discrete_obs_interval is None:  # 每日观察
            if prod.updown == UpDown.Up:
                knock_inout = np.any(paths >= prod.barrier, axis=0)
            elif prod.updown == UpDown.Down:
                knock_inout = np.any(paths <= prod.barrier, axis=0)
            else:
                raise ValueError("不支持的UpDown类型")
        else:  # 均匀离散观察
            dt_step = prod.discrete_obs_interval * prod.t_step_per_year
            obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
            obs_points = np.concatenate((np.array([0]), obs_points))
            obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界
            if prod.updown == UpDown.Up:
                knock_inout = np.any(paths[obs_points] >= prod.barrier, axis=0)
            elif prod.updown == UpDown.Down:
                knock_inout = np.any(paths[obs_points] <= prod.barrier, axis=0)
            else:
                raise ValueError("不支持的UpDown类型")

        if prod.inout == InOut.In:
            payoff = np.ones(paths[-1].size) * prod.rebate
            payoff[knock_inout] = np.maximum(prod.callput.value * (paths[-1, knock_inout] - prod.strike),
                                             0) * prod.parti
        elif prod.inout == InOut.Out:
            payoff = np.maximum(prod.callput.value * (paths[-1] - prod.strike), 0) * prod.parti
            payoff[knock_inout] = prod.rebate
        else:
            raise ValueError("不支持的InOut类型")
        price = np.mean(payoff) * np.exp(- r * _maturity)
        return price
