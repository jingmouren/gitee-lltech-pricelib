#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import CallPut, ExerciseType, PaymentType
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.time import global_evaluation_date


class QuadDigitalEngine(QuadEngine):
    """二元(数字)期权-数值积分法定价引擎"""

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

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        dt = prod.discrete_obs_interval
        self.backward_steps = round(tau / dt)
        self._check_method_params()

        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)

        upper_barrier = self.n_max * spot
        lower_barrier = 1
        s_vec = np.linspace(lower_barrier, upper_barrier, self.n_points)
        # 行权价在价格格点上的对应格点
        strike_point = np.where(s_vec >= prod.strike)[0][0]
        # 欧式：
        if prod.exercise_type == ExerciseType.European:
            v_grid = np.zeros(self.n_points)
            if prod.callput == CallPut.Call:
                v_grid[strike_point:] = prod.rebate
            elif prod.callput == CallPut.Put:
                v_grid[:strike_point] = prod.rebate
            y = s_vec
            x = np.array(spot)
            value = self.step_backward(x, y, v_grid, tau)[0]
        # 美式：
        elif prod.exercise_type == ExerciseType.American:
            v_grid = np.zeros(shape=(self.n_points, self.backward_steps + 1))
            s_vec = np.tile(s_vec, reps=(self.backward_steps + 1, 1)).T

            if prod.callput == CallPut.Call:
                v_grid[strike_point:, -1] = prod.rebate
            elif prod.callput == CallPut.Put:
                v_grid[:strike_point, -1] = prod.rebate

            for step in range(self.backward_steps - 1, 0, -1):
                y = s_vec[:, step + 1]

                if prod.callput == CallPut.Call:
                    x = s_vec[:strike_point, step]
                    v_grid[:strike_point, step] = self.step_backward(x, y, v_grid[:, step + 1], dt)
                    if prod.payment_type == PaymentType.Hit:
                        v_grid[strike_point:, step] = prod.rebate
                    elif prod.payment_type == PaymentType.Expire:
                        v_grid[strike_point:, step] = prod.rebate * math.exp(-r * dt * (self.backward_steps - step))

                elif prod.callput == CallPut.Put:
                    x = s_vec[strike_point:, step]
                    v_grid[strike_point:, step] = self.step_backward(x, y, v_grid[:, step + 1], dt)
                    if prod.payment_type == PaymentType.Hit:
                        v_grid[:strike_point, step] = prod.rebate
                    elif prod.payment_type == PaymentType.Expire:
                        v_grid[:strike_point, step] = prod.rebate * math.exp(-r * dt * (self.backward_steps - step))

            y = s_vec[:, 1]
            x = np.array(spot)
            value = self.step_backward(x, y, v_grid[:, 1], dt)[0]
        else:
            raise ValueError("ExerciseType must be American or European")
        return value
