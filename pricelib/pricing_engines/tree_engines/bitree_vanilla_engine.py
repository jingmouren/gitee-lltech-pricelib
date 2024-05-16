#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import ExerciseType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import BiTreeEngine


class BiTreeVanillaEngine(BiTreeEngine):
    """香草期权二叉树定价引擎
    支持欧式期权和美式期权"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        tau = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()

        self.r = self.process.interest(tau)
        n_step = round(tau * self.tree_branches)
        s_paths = self.path_generator(n_step, tau, spot).copy()
        v_grid = np.zeros(shape=s_paths.shape)

        for j in range(n_step, -1, -1):
            if j == n_step:
                v_grid[:, j] = np.maximum(prod.callput.value * (s_paths[:, j] - prod.strike), 0)
            else:
                v_grid[:j + 1, j] = (v_grid[:j + 1, j + 1] * self.p + v_grid[1:j + 2, j + 1] * (
                        1 - self.p)) * np.exp(-self.r * self.dt)
                if prod.exercise_type == ExerciseType.American:
                    v_grid[:j + 1, j] = np.maximum(v_grid[:j + 1, j],
                                                   prod.callput.value * (self.s_paths[:j + 1, j] - prod.strike))
        return v_grid[0, 0]
