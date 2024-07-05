#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import QuadEngine
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
        obs_dates = np.array([num for num in obs_dates if num >= 0]) / prod.t_step_per_year
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        _barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        self._barrier_in = prod.barrier_in[-len(obs_dates):].copy()
        self._barrier_yield = prod.barrier_yield[-len(obs_dates):].copy()
        self._coupon = prod.coupon[-len(obs_dates):].copy()

        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(_maturity)
        q = self.process.div(_maturity)
        vol = self.process.vol(_maturity, spot)
        self._check_method_params()
        self.set_quad_params(r=r, q=q, vol=vol)

        if _maturity_business_days == 0:  # 如果估值日就是到期日
            # 直接计算终止条件
            value = self._init_terminal_condition(prod, np.array([spot]))[0]
            return value

        if obs_dates[0] == 0:
            # 如果估值日就是敲出观察日
            if spot >= _barrier_out[0]:  # 发生敲出
                return (prod.margin_lvl + self._coupon[0]) * prod.s0
            else:  # 没有发生敲出，则积分迭代步数少一次
                s0_dt = obs_dates[1]
                dt_vec = np.diff(obs_dates)[1:]
        else:
            s0_dt = obs_dates[0]
            dt_vec = np.diff(obs_dates)

        backward_steps = dt_vec.size
        # 初始化fft的对数价格向量及边界的对数价格向量
        self.init_grid(spot, vol, _maturity_business_days)
        s_vec = np.exp(self.ln_s_vec)
        v_grid = np.zeros(shape=(self.n_points, backward_steps + 2))

        barrier_out_idx = np.searchsorted(s_vec, _barrier_out, side='right')
        barrier_yield_idx = np.searchsorted(s_vec, self._barrier_yield, side='right')

        for step in range(backward_steps + 1, 0, -1):
            if step == backward_steps + 1:  # 设置期末价值状态: 预付金 + 派息 + 到期敲入或有亏损
                v_grid[:, -1] = self._init_terminal_condition(prod, s_vec)
            else:
                # 数值积分，计算前一个敲出观察日的期权价值
                v_grid[:, step] = self.fft_step_backward(self.ln_s_vec, self.ln_s_vec,
                                                         v_grid[:, step + 1], dt_vec[step - 1])
                # 敲出价之上，发生敲出，返还预付金 + 派息
                v_grid[barrier_out_idx[step - 1]:, step] = (prod.margin_lvl + self._coupon[step - 1]) * prod.s0
                # 派息线之上，敲出价之下，派息
                v_grid[barrier_yield_idx[step - 1]:barrier_out_idx[step - 1], step] += self._coupon[step - 1] * prod.s0
                # 派息线之下，不派息

        x = np.array([np.log(spot)])
        value = self.fft_step_backward(x, self.ln_s_vec, v_grid[:, 1], s0_dt)[0]
        if obs_dates[0] == 0:  # 如果估值日就是敲出观察日，并且spot位于派息线上、敲出线下，则需要加上派息的价值
            value += self._coupon[0] * prod.s0
        return value

    def _init_terminal_condition(self, prod, s_vec):
        """设置期末价值: 预付金 + 派息 + 到期敲入或有亏损
        Args:
            prod: Product产品对象
            s_vec: np.ndarray, 网格的价格格点
        Returns:
            payoff: np.ndarray, 期末价值向量
        """
        payoff = (prod.margin_lvl * prod.s0 +
                  np.where(s_vec > self._barrier_yield[-1], self._coupon[-1] * prod.s0, 0) +
                  np.where(s_vec > self._barrier_in[-1], 0,
                           prod.parti_in * (- prod.strike_upper + np.where(s_vec > prod.strike_lower,
                                                                           s_vec, prod.strike_lower))))
        return payoff
