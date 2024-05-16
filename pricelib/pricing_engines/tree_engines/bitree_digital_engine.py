#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import CallPut, ExerciseType, PaymentType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import BiTreeEngine


class BiTreeDigitalEngine(BiTreeEngine):
    """二元(数字)期权PDE二叉树定价引擎"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        self.r = self.process.interest(tau)
        n_step = round(tau * self.tree_branches)
        s_paths = self.path_generator(n_step, tau, spot).copy()
        v_grid = np.zeros(shape=s_paths.shape)

        # 欧式：
        if prod.exercise_type == ExerciseType.European:
            for j in range(n_step, -1, -1):
                if j == n_step:
                    if prod.callput == CallPut.Call:
                        v_grid[:, -1] = (s_paths[:, -1] >= prod.strike) * prod.rebate
                    elif prod.callput == CallPut.Put:
                        v_grid[:, -1] = (s_paths[:, -1] <= prod.strike) * prod.rebate
                else:
                    v_grid[:-1, j] = v_grid[:-1, j + 1] * self.p + v_grid[1:, j + 1] * (1 - self.p)
                    v_grid[:-1, j] *= np.exp(-self.r * self.dt)

        # 美式：
        if prod.exercise_type == ExerciseType.American:
            # 设置观察时点obs_points
            if prod.discrete_obs_interval is None:
                obs_points = np.arange(0, n_step + 1, 1)[1:]
            else:
                obs_points = np.flip(np.round(np.arange(tau, 0, -prod.discrete_obs_interval) * self.tree_branches))
                obs_points = np.concatenate((np.array([0]), obs_points))
            obs_points = np.round(obs_points).astype(int)

            for j in range(n_step, -1, -1):
                if j == n_step:
                    if prod.callput == CallPut.Call:
                        v_grid[:, -1] = (s_paths[:, -1] >= prod.strike) * prod.rebate
                    elif prod.callput == CallPut.Put:
                        v_grid[:, -1] = (s_paths[:, -1] <= prod.strike) * prod.rebate
                    else:
                        raise ValueError("Unknown CallPut type")
                else:
                    v_grid[:j + 1, j] = v_grid[:j + 1, j + 1] * self.p + v_grid[1:j + 2, j + 1] * (1 - self.p)
                    v_grid[:-1, j] *= np.exp(-self.r * self.dt)
                    if j in obs_points:
                        if prod.callput == CallPut.Call:
                            upper_s = np.where(self.s_paths[:j, j] >= prod.strike)
                            if prod.payment_type == PaymentType.Hit:
                                v_grid[upper_s, j] = prod.rebate
                            elif prod.payment_type == PaymentType.Expire:
                                time_remain = self.dt * (n_step - j)
                                v_grid[upper_s, j] = prod.rebate * math.exp(-self.r * time_remain)

                        elif prod.callput == CallPut.Put:
                            lower_s = np.where(self.s_paths[:j, j] <= prod.strike)
                            if prod.payment_type == PaymentType.Hit:
                                v_grid[lower_s, j] = prod.rebate
                            elif prod.payment_type == PaymentType.Expire:
                                time_remain = self.dt * (n_step - j)
                                v_grid[lower_s, j] = prod.rebate * math.exp(-self.r * time_remain)
        return v_grid[0, 0]
