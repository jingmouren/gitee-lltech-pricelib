#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from scipy.optimize import leastsq
from pricelib.common.utilities.enums import ExerciseType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import McEngine


# pylint: disable=invalid-name
def quadratic_func(x, p):
    """二次函数"""
    a, b, c = p
    return a * (x ** 2) + b * x + c


# pylint: disable=invalid-name
def residuals_func(p, y, x):
    """残差函数"""
    return y - quadratic_func(x, p)


class MCVanillaEngine(McEngine):
    """香草期权 Monte Carlo 模拟定价引擎
        支持欧式期权和美式期权，美式期权为LSMC方法"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        tau = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        n_step = int(tau * prod.t_step_per_year)
        paths = self.path_generator(n_step=n_step, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()
        r = self.process.interest(tau)
        if prod.exercise_type == ExerciseType.European:
            price = np.mean(np.maximum(prod.callput.value * (paths[-1, :] - prod.strike), 0)) * np.exp(-r * tau)
            return price
        if prod.exercise_type == ExerciseType.American:  # 美式期权，LSMC方法
            v_grid = np.zeros(shape=paths.shape)
            v_grid[-1, :] = np.maximum(prod.callput.value * (paths[-1, :] - prod.strike), 0)
            if n_step == 0:  # 如果估值日就是到期日
                return np.mean(v_grid[-1, :])
            for i in range(n_step - 1, -1, -1):
                v_grid[i, :] = v_grid[i + 1, :] * np.exp(-r * 1 / prod.t_step_per_year)
                lsm_id = (prod.callput.value * (paths[i, :] - prod.strike) >= 0)
                if np.array(lsm_id).size <= 2:  # 拟合曲线有三个参数，若实值点<3则无法拟合
                    continue
                xdata = np.squeeze(paths[i, lsm_id])
                ydata = np.squeeze(v_grid[i, lsm_id])
                lsm_params = leastsq(residuals_func, x0=np.array([0.01, -1.0, spot]), args=(ydata, xdata))
                hold_value = quadratic_func(paths[i, lsm_id], lsm_params[0])  # 最小二乘拟合的持有价值
                exercise_value = prod.callput.value * (paths[i, lsm_id] - prod.strike)
                # v_grid[i, lsm_id] = np.where(exercise_value >= hold_value, exercise_value, v_grid[i, lsm_id])
                v_grid[i, lsm_id] = np.maximum(hold_value, exercise_value)
            return np.mean(v_grid[0, :])
        raise ValueError("不支持的行权方式, 香草mc定价引擎仅支持欧式期权和美式期权")
