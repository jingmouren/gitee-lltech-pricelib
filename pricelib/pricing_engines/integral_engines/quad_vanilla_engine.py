#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import ExerciseType
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.time import global_evaluation_date


class QuadVanillaEngine(QuadEngine):
    """香草期权数值积分法定价引擎
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

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        self._check_method_params()

        if prod.exercise_type == ExerciseType.European:
            self.backward_steps = 1
        elif prod.exercise_type == ExerciseType.American:
            dt = 1 / prod.t_step_per_year
            self.backward_steps = round(tau / dt)
        else:
            raise ValueError(f"不支持的行权方式{prod.exercise_type.value}")

        v_grid = np.empty(shape=(self.n_points, self.backward_steps + 1))
        s_min = 0.01
        s_max = self.n_max * spot
        s_vec = np.linspace(s_min, s_max, self.n_points)
        s_vec = np.tile(s_vec, reps=(self.backward_steps + 1, 1)).T
        # 设定期末价值
        v_grid[:, -1] = np.maximum(prod.callput.value * (s_vec[:, -1] - prod.strike), 0)
        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)
        # 逐步回溯计算
        if prod.exercise_type == ExerciseType.European:
            return self.step_backward(np.array(spot), s_vec[:, -1], v_grid[:, -1], tau)[0]
        if prod.exercise_type == ExerciseType.American:
            for step in range(self.backward_steps - 1, 0, -1):
                # 积分计算区域
                y = s_vec[:, step + 1]
                x = s_vec[:, step]
                v_grid[:, step] = self.step_backward(x, y, v_grid[:, step + 1], dt)
                v_grid[:, step] = np.minimum(v_grid[:, step], 1e250)  # 防止价格极大和极小时，期权价值趋近于无穷大
                v_grid[:, step] = np.maximum(v_grid[:, step], prod.callput.value * (s_vec[:, step] - prod.strike))
            erase = 1
            y = s_vec[erase:-erase, 1]
            x = np.array(spot)
            value = self.step_backward(x, y, v_grid[erase:-erase, 1], dt)[0]
            return value

        raise NotImplementedError(f"不支持的行权方式{prod.exercise_type.value}")
