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

        # 初始化fft的对数价格向量及边界的对数价格向量
        self.init_grid(spot, vol, tau)
        # 行权价在价格格点上的对应格点
        strike_point = np.where(np.exp(self.ln_s_vec) >= prod.strike)[0][0]
        v_vec = np.zeros(len(self.ln_s_vec))
        if prod.callput == CallPut.Call:
            v_vec[strike_point:] = prod.rebate
        elif prod.callput == CallPut.Put:
            v_vec[:strike_point] = prod.rebate
        # 欧式：
        if prod.exercise_type == ExerciseType.European:
            value = self.fft_step_backward(np.log(np.array([spot])), self.ln_s_vec, v_vec, tau)[0]
        # 美式：
        elif prod.exercise_type == ExerciseType.American:
            for step in range(self.backward_steps - 1, 0, -1):
                if prod.payment_type == PaymentType.Hit:
                    rebate_v = prod.rebate
                elif prod.payment_type == PaymentType.Expire:
                    rebate_v = prod.rebate * math.exp(-r * dt * (self.backward_steps - step))
                else:
                    raise ValueError("Invalid payment type")
                if prod.callput == CallPut.Call:
                    v_vec[:strike_point] = self.fft_step_backward(self.ln_s_vec[:strike_point], self.ln_s_vec, v_vec, dt)
                    v_vec[strike_point:] = rebate_v
                elif prod.callput == CallPut.Put:
                    v_vec[strike_point:] = self.fft_step_backward(self.ln_s_vec[strike_point:], self.ln_s_vec, v_vec, dt)
                    v_vec[:strike_point] = rebate_v
                else:
                    raise ValueError("Invalid call put type")

            value = self.fft_step_backward(np.array(np.log([spot])), self.ln_s_vec, v_vec, dt)[0]
        else:
            raise ValueError("ExerciseType must be American or European")
        return value
