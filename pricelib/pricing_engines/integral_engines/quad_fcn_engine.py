#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.utilities.patterns import HashableArray
from pricelib.common.time import global_evaluation_date


class QuadFCNEngine(QuadEngine):
    """FCN/DCN 数值积分定价引擎"""

    # pylint: disable=too-many-locals
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        _maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date,
                                                                            prod.end_date) / prod.t_step_per_year
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        obs_dates = np.array([num for num in obs_dates if num >= 0])
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        _barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        _barrier_in = prod.barrier_in[-len(obs_dates):].copy()
        _barrier_yield = prod.barrier_yield[-len(obs_dates):].copy()
        _coupon = prod.coupon[-len(obs_dates):].copy()

        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(_maturity)
        q = self.process.div(_maturity)
        vol = self.process.vol(_maturity, spot)
        self.backward_steps = obs_dates.size
        self._check_method_params()
        self.set_quad_params(r=r, q=q, vol=vol)
        upper_barrier = self.n_max * prod.s0
        lower_barrier = 1
        s_vec = HashableArray(np.linspace(lower_barrier, upper_barrier, self.n_points))
        v_grid = np.zeros(shape=(self.n_points, self.backward_steps + 1))

        barrier_out_idx = np.searchsorted(s_vec, _barrier_out, side='right')
        barrier_yield_idx = np.searchsorted(s_vec, _barrier_yield, side='right')
        dt = _maturity_business_days / self.backward_steps

        for step in range(self.backward_steps, 0, -1):
            if step == self.backward_steps:  # 设置期末价值状态: 预付金 + 派息 + 到期敲入或有亏损
                v_grid[:, -1] = (prod.margin_lvl * prod.s0 +
                                 np.where(s_vec >= _barrier_yield[-1], _coupon[-1] * prod.s0, 0) +
                                 np.where(s_vec >= _barrier_in[-1], 0, prod.parti_in * (- prod.strike_upper +
                                          np.where(s_vec > prod.strike_lower, s_vec, prod.strike_lower))))
            else:
                # 数值积分，计算前一个敲出观察日的期权价值
                v_grid[:, step] = self.step_backward(s_vec, s_vec, v_grid[:, step + 1], dt)
                # 敲出价之上，发生敲出，返还预付金 + 派息
                v_grid[barrier_out_idx[step - 1]:, step] = (prod.margin_lvl + _coupon[step - 1]) * prod.s0
                # 派息线之上，敲出价之下，派息
                v_grid[barrier_yield_idx[step - 1]:barrier_out_idx[step - 1], step] += _coupon[step - 1] * prod.s0
                # 派息线之下，不派息

        y = s_vec
        x = HashableArray(np.array(spot))
        value = self.step_backward(x, y, v_grid[:, 1], dt)[0]
        return value
