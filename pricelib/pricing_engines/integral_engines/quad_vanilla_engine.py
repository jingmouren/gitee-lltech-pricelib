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
        dt = tau
        if prod.exercise_type == ExerciseType.European:
            self.backward_steps = 1
        elif prod.exercise_type == ExerciseType.American:
            dt = 1 / prod.t_step_per_year
            self.backward_steps = round(tau / dt)
        else:
            raise ValueError(f"不支持的行权方式{prod.exercise_type.value}")

        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)
        # 设定期末价值
        self.init_grid(spot, vol, tau)
        v_vec = np.maximum(prod.callput.value * (np.exp(self.ln_s_vec) - prod.strike), 0)

        # 逐步回溯计算
        if prod.exercise_type == ExerciseType.European:
            return self.fft_step_backward(np.log(np.array([spot])), self.ln_s_vec, v_vec, tau)[0]
        if prod.exercise_type == ExerciseType.American:
            for step in range(self.backward_steps - 1, 0, -1):
                # 积分计算区域
                v_vec = self.fft_step_backward(self.ln_s_vec, self.ln_s_vec, v_vec, dt)
                v_vec = np.maximum(v_vec, prod.callput.value * (np.exp(self.ln_s_vec) - prod.strike))

            x = np.log(np.array([spot]))
            value = self.fft_step_backward(x, self.ln_s_vec, v_vec, dt)[0]
            return value

        raise NotImplementedError(f"不支持的行权方式{prod.exercise_type.value}")
