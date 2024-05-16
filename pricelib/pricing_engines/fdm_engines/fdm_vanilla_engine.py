#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import CallPut, ExerciseType
from pricelib.common.pricing_engine_base import FdmEngine, FdmGridwithBound, FdmGrid
from pricelib.common.time import global_evaluation_date


class FdmVanillaEngine(FdmEngine):
    """香草期权PDE有限差分法定价引擎
    支持欧式期权和美式期权"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值，PDE有限差分定价, 时间逆向迭代
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        smax = self.n_smax * spot  # 价格网格上界, 默认为4倍初始价格
        # 返回PDE系数组件abc的函数
        fn_pde_coef = self.process.get_fn_pde_coef(maturity, spot)

        # 生成边界条件：在s=0处，call近似为0; 在smax处，put近似为0。根据看涨看跌平价 call-put = S*exp(-qT) - K*exp(-rT),
        # 可得对应的put(0) = K * exp(-rT), call(smax) = smax * exp(-qT) - K * exp(-rT)
        if prod.callput == CallPut.Call:
            fn_callput = [lambda u: 0, lambda u: smax * self.process.div.disc_factor(maturity, u) -
                          prod.strike * self.process.interest.disc_factor(maturity, u)]
        else:  # CallPut.Put
            fn_callput = [lambda u: prod.strike * self.process.interest.disc_factor(maturity, u),
                          lambda u: 0]
        if prod.exercise_type == ExerciseType.European:
            fdm = FdmGridwithBound(smax=smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                   fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta)
            fdm.set_boundary_condition(fn_callput)
            # 初始化到期日payoff价值yv=max(callput * (F-K), 0)。fdm.xv是log(F)
            yv = np.maximum(prod.callput.value * (fdm.s_vec - prod.strike), 0)
            yv = fdm.evolve_with_interval(start=maturity, end=0, yv=yv)  # 时间逆向迭代求解
        elif prod.exercise_type == ExerciseType.American:
            t_step = round(maturity * prod.t_step_per_year)  # 时间格点数
            t_vec = np.linspace(0, maturity, t_step + 1)  # 时间向量
            dt = (t_vec[-1] - t_vec[0]) / (t_vec.size - 1)  # 时间步长
            fdm = FdmGrid(smax=smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step, fn_pde_coef=fn_pde_coef,
                          fdm_theta=self.fdm_theta, maturity=maturity)
            fdm.v_grid[0] = fn_callput[0](t_vec)
            fdm.v_grid[-1] = fn_callput[1](t_vec)
            yv = np.maximum(prod.callput.value * (fdm.s_vec - prod.strike), 0)
            v_vec = prod.callput.value * (fdm.s_vec - prod.strike)
            for j in range(t_step - 1, -1, -1):
                yv = fdm.evolve(j, yv, dt)  # 每组(p, q)更新。如果在障碍观察区间内，更新fd_bound；在观察区间外，更新fd_full
                yv = np.maximum(yv, v_vec)
        else:
            raise ValueError("不支持的期权类型，仅支持欧式或美式香草期权")
        return np.float64((fdm.functionize(yv, kind="cubic")(spot)))
