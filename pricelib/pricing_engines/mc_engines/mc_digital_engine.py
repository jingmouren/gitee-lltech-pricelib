#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.time import global_evaluation_date
from pricelib.common.utilities.enums import CallPut, ExerciseType, PaymentType
from pricelib.common.pricing_engine_base import McEngine


class MCDigitalEngine(McEngine):
    """二元(数字)期权-现金或无-Monte Carlo模拟定价引擎"""

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
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date,
                                                                            prod.end_date)
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        r = self.process.interest(_maturity)
        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        # 美式：
        if prod.exercise_type == ExerciseType.American:
            if prod.discrete_obs_interval is None:  # 每日观察
                obs_points = np.arange(0, _maturity_business_days + 1, 1)
            else:  # 均匀离散观察
                dt_step = prod.discrete_obs_interval * prod.t_step_per_year
                obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
                obs_points = np.concatenate((np.array([0]), obs_points))
                obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界

            if prod.callput == CallPut.Call:
                strike_bool = np.where(paths[obs_points] >= prod.strike, 1, 0)
            else:  # self.callput == CallPut.Put:
                strike_bool = np.where(paths[obs_points] <= prod.strike, 1, 0)
            value = prod.rebate * np.max(strike_bool, axis=0)  # 能否拿到票息
            if prod.payment_type == PaymentType.Hit:
                strike_time = (np.argmax(strike_bool, axis=0) + 1) * (1 / prod.t_step_per_year)  # 停时
                value = np.mean(value * np.exp(-r * strike_time))
            else:  # self.payment_type == PaymentType.Expire:
                value = np.mean(value * math.exp(-r * _maturity))

        # 欧式：
        elif prod.exercise_type == ExerciseType.European:
            if prod.callput == CallPut.Call:
                payoff = (paths[-1] >= prod.strike) * prod.rebate
            else:  # prod.callput == CallPut.Put:
                payoff = (paths[-1] <= prod.strike) * prod.rebate
            value = np.mean(payoff) * math.exp(-r * _maturity)
        else:
            raise ValueError("观察方式只支持美式观察American或欧式观察European")

        return value
