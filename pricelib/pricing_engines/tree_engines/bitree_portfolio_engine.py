#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import BiTreeEngine
from pricelib.common.time import global_evaluation_date


class BiTreePortfolioEngine(BiTreeEngine):
    """香草组合二叉树定价引擎"""

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
        n_step = int(tau * self.tree_branches)

        s_paths = self.path_generator(n_step, tau, spot).copy()
        v_grid = np.zeros(shape=s_paths.shape)
        for vanilla, position in prod.vanilla_list:
            v_grid[:, n_step] += np.maximum(vanilla.callput.value * (s_paths[:, -1] - vanilla.strike), 0) * position
        for j in range(n_step - 1, -1, -1):
            v_grid[:j + 1, j] = (v_grid[:j + 1, j + 1] * self.p + v_grid[1:j + 2, j + 1] * (
                    1 - self.p)) * np.exp(-self.process.interest(tau) * self.dt)
        return v_grid[0, 0]
