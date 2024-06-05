#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import CallPut
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.time import global_evaluation_date


class QuadAutoCallEngine(QuadEngine):
    """Autocall Note(二元小雪球) 数值积分定价引擎"""

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
        obs_dates = np.array([num / prod.t_step_per_year for num in obs_dates if num >= 0])
        calculate_start_diff = (calculate_date - prod.start_date).days
        pay_dates = prod.pay_dates.count_calendar_days(prod.start_date)
        pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= calculate_start_diff])
        assert len(obs_dates) == len(pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        diff_obs_pay_dates = np.array([(prod.pay_dates.date_schedule[i] - prod.obs_dates.date_schedule[i]).days
                                       for i in range(len(obs_dates))]) / prod.annual_days.value
        # 经过估值日截断列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        _barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        _coupon_out = prod.coupon_out[-len(obs_dates):].copy()

        if spot is None:
            spot = self.process.spot()

        r = self.process.interest(_maturity)
        q = self.process.div(_maturity)
        vol = self.process.vol(_maturity, spot)

        self._check_method_params()
        self.set_quad_params(r=r, q=q, vol=vol)
        # 初始化fft的对数价格向量及边界的对数价格向量
        self.init_grid(spot, vol, _maturity_business_days)
        s_vec = np.exp(self.ln_s_vec)

        if obs_dates[0] == 0:
            # 如果估值日就是敲出观察日
            if (prod.callput == CallPut.Call and spot >= _barrier_out[0]) or (
                    prod.callput == CallPut.Put and spot <= _barrier_out[0]):  # 发生敲出
                return prod.s0 * (prod.margin_lvl + _coupon_out[0] * pay_dates[
                        0]) * math.exp(-r * diff_obs_pay_dates[0])
            else:  # 没有发生敲出，则积分迭代步数少一次
                s0_dt = obs_dates[1]
                dt_vec = np.diff(obs_dates)[1:]
        else:
            s0_dt = obs_dates[0]
            dt_vec = np.diff(obs_dates)

        self.backward_steps = dt_vec.size
        v_grid = np.zeros(shape=(self.n_points, self.backward_steps + 2))
        for step in range(self.backward_steps + 1, 0, -1):
            y = self.ln_s_vec
            current_barrier = np.searchsorted(s_vec, _barrier_out[step - 1], side='right')  # 行权价在价格格点上的对应格点

            if step == self.backward_steps + 1:
                if prod.callput == CallPut.Call:
                    v_grid[current_barrier:, -1] = (prod.margin_lvl + _coupon_out[-1] * pay_dates[-1]) * prod.s0
                    v_grid[:current_barrier, -1] = (prod.margin_lvl + prod.coupon_div * pay_dates[-1]) * prod.s0
                elif prod.callput == CallPut.Put:
                    v_grid[:current_barrier, -1] = (prod.margin_lvl + _coupon_out[-1] * pay_dates[-1]) * prod.s0
                    v_grid[current_barrier:, -1] = (prod.margin_lvl + prod.coupon_div * pay_dates[-1]) * prod.s0
            else:
                if prod.callput == CallPut.Call:
                    x = self.ln_s_vec[:current_barrier]
                    v_grid[:current_barrier, step] = self.fft_step_backward(x, y, v_grid[:, step + 1], dt_vec[step - 1])
                    v_grid[current_barrier:, step] = prod.s0 * (prod.margin_lvl + _coupon_out[step - 1] * pay_dates[
                        step - 1]) * math.exp(-r * diff_obs_pay_dates[step - 1])

                elif prod.callput == CallPut.Put:
                    x = self.ln_s_vec[current_barrier:]
                    v_grid[current_barrier:, step] = self.fft_step_backward(x, y, v_grid[:, step + 1], dt_vec[step - 1])
                    v_grid[:current_barrier, step] = prod.s0 * (prod.margin_lvl + _coupon_out[step - 1] * pay_dates[
                        step - 1]) * math.exp(-r * diff_obs_pay_dates[step - 1])

        x = np.array([np.log(spot)])
        value = self.fft_step_backward(x, self.ln_s_vec, v_grid[:, 1], s0_dt)[0]
        return value
