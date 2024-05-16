#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.time import global_evaluation_date
from pricelib.common.utilities.enums import TouchType, ExerciseType, PaymentType
from pricelib.common.pricing_engine_base import McEngine


class MCDoubleDigitalEngine(McEngine):
    """双边二元期权(欧式-二元凸式/二元凹式；美式-双接触/双不接触) Monte Carlo 模拟定价引擎"""

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

        # 美式：美式双接触/美式双不接触
        if prod.exercise_type == ExerciseType.American:
            if prod.discrete_obs_interval is None:  # 每日观察
                obs_points = np.arange(0, _maturity_business_days + 1, 1)[1:]
            else:  # 均匀离散观察
                dt_step = prod.discrete_obs_interval * prod.t_step_per_year
                obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
                obs_points = np.concatenate((np.array([0]), obs_points))
                obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界

            if prod.touch_type == TouchType.NoTouch:  # 美式双不接触一定是到期支付
                strike_bool = np.where((paths[obs_points] <= prod.bound[0]) | (paths[obs_points] >= prod.bound[1]),
                                       0, 1)
                value = prod.rebate[0] * math.exp(-r * _maturity) * np.mean(np.min(strike_bool, axis=0))
            elif prod.touch_type == TouchType.Touch:
                step_index = np.tile(obs_points, (self.n_path, 1)).T
                hit_lower = np.min(
                    np.where(paths[obs_points] <= prod.bound[0], step_index.astype(int), _maturity_business_days + 1),
                    axis=0)
                hit_upper = np.min(
                    np.where(paths[obs_points] >= prod.bound[1], step_index.astype(int), _maturity_business_days + 1),
                    axis=0)
                if prod.payment_type == PaymentType.Hit:
                    value = np.mean(
                        np.where(hit_lower < hit_upper, prod.rebate[0] * np.exp(-r * hit_lower / prod.t_step_per_year),
                                 np.where(hit_lower > hit_upper,
                                          prod.rebate[1] * np.exp(-r * hit_upper / prod.t_step_per_year), 0)))
                elif prod.payment_type == PaymentType.Expire:
                    value = np.mean(np.where(hit_lower < hit_upper, prod.rebate[0],
                                             np.where(hit_lower > hit_upper, prod.rebate[1],
                                                      0))) * math.exp(-r * _maturity)
                else:
                    raise ValueError("PaymentType must be Hit or Expire")
            else:
                raise ValueError("TouchType must be Touch or NoTouch")
        # 欧式：二元凹式/二元凸式
        elif prod.exercise_type == ExerciseType.European:
            if prod.touch_type == TouchType.Touch:
                payoff = np.where(paths[-1] <= prod.bound[0], prod.rebate[0],
                                  np.where(paths[-1] >= prod.bound[1], prod.rebate[1], 0))
            elif prod.touch_type == TouchType.NoTouch:
                payoff = ((paths[-1] > prod.bound[0]) & (paths[-1] < prod.bound[1])) * prod.rebate[0]
            else:
                raise ValueError("TouchType must be Touch or NoTouch")
            value = np.mean(payoff) * math.exp(-r * _maturity)
        else:
            raise ValueError("ExerciseType must be American or European")
        return value
