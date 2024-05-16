#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import ExerciseType, PaymentType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import McEngine


class MCDoubleSharkEngine(McEngine):
    """双鲨期权 Monte Carlo 模拟定价引擎"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert (prod.strike[0] <= prod.strike[1]) and (
                prod.bound[0] < prod.bound[1]), "Error: 双鲨结构输入的行权价和障碍价必须是递增的，不能是None"
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径
        calculate_date = global_evaluation_date() if t is None else t
        _maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)

        r = self.process.interest(_maturity)
        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        call_payoff = (paths[-1] - prod.strike[1]) * prod.parti[1]
        put_payoff = (prod.strike[0] - paths[-1]) * prod.parti[0]
        # 美式：
        if prod.exercise_type == ExerciseType.American:
            if prod.discrete_obs_interval is None:  # 每日观察
                obs_points = np.arange(0, _maturity_business_days + 1, 1)[1:]
            else:  # 均匀离散观察
                dt_step = prod.discrete_obs_interval * prod.t_step_per_year
                obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
                obs_points = np.concatenate((np.array([0]), obs_points))
                obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界

            # 双鲨结构是双边敲出
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
                                      prod.rebate[1] * np.exp(-r * hit_upper / prod.t_step_per_year),
                                      np.where(call_payoff > 0, call_payoff * math.exp(-r * _maturity),
                                               np.where(put_payoff > 0, put_payoff * math.exp(-r * _maturity), 0)))))
            elif prod.payment_type == PaymentType.Expire:
                value = np.mean(np.where(hit_lower < hit_upper, prod.rebate[0] * np.exp(-r * _maturity),
                                         np.where(hit_lower > hit_upper, prod.rebate[1] * np.exp(-r * _maturity),
                                                  np.where(call_payoff > 0, call_payoff * math.exp(-r * _maturity),
                                                           np.where(put_payoff > 0,
                                                                    put_payoff * math.exp(-r * _maturity),
                                                                    0)))))
            else:
                raise ValueError("PaymentType must be Hit or Expire")

        # 欧式：
        elif prod.exercise_type == ExerciseType.European:
            # 双鲨结构是双边敲出
            value = np.where(paths[-1] <= prod.bound[0], prod.rebate[0],
                             np.where(paths[-1] >= prod.bound[1], prod.rebate[1],
                                      np.where(call_payoff > 0, call_payoff * math.exp(-r * _maturity),
                                               np.where(put_payoff > 0, put_payoff * math.exp(-r * _maturity), 0))))
            value = np.mean(value) * math.exp(-r * _maturity)
        else:
            raise ValueError("ExerciseType must be American or European")
        return value
