#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from functools import lru_cache
import numpy as np
from pricelib.common.utilities.enums import InOut
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import QuadEngine


class QuadDoubleSharkEngine(QuadEngine):
    """双鲨结构PDE有限差分法定价引擎
    只支持美式观察(整个有效期观察)；只支持离散观察(默认为每日观察)；支持现金返还到期支付"""
    inout = InOut.Out  # 双鲨结构属于敲出期权

    @lru_cache(maxsize=1)
    def set_quad_price_range(self, s_sliced):
        """
        设置数值积分的区间(价格范围)
        Args:
            s_sliced: np.ndarray = self.s[:, step], 其中step是从后向前的步数
        Returns:
            quad_range: tuple, (np.ndarray, ), 数值积分的区间(索引)
            upper_an_range: tuple, (np.ndarray, ), 解析式计算积分的区间(索引)
            lower_an_range: tuple, (np.ndarray, ), 解析式计算积分的区间(索引)
        """
        quad_range = np.where((s_sliced < self.prod.bound[1]) & (s_sliced > self.prod.bound[0]))
        upper_an_range = np.where(s_sliced >= self.prod.bound[1])
        lower_an_range = np.where(s_sliced <= self.prod.bound[0])
        return quad_range, upper_an_range, lower_an_range

    # pylint: disable=too-many-locals
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
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        dt = prod.discrete_obs_interval if prod.discrete_obs_interval is not None else 1 / prod.t_step_per_year
        self.backward_steps = int(tau / dt)
        self._check_method_params()

        # v_grid = np.empty(shape=(self.n_points, self.backward_steps + 1))
        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)
        # 初始化fft的对数价格向量及边界的对数价格向量
        self.init_grid(spot, vol, tau)
        s_vec = np.exp(self.ln_s_vec)
        # 从后向前进行积分计算
        # 设定期末价值，在障碍价格内侧，默认设定未敲入，未敲出
        call_put_payoff = (np.maximum(s_vec - prod.strike[1], 0) * prod.parti[1] +
                           np.maximum(prod.strike[0] - s_vec, 0) * prod.parti[0])
        v_vec = np.where(s_vec >= prod.bound[1], prod.rebate[1],
                         np.where(s_vec <= prod.bound[0], prod.rebate[0], call_put_payoff))

        for step in range(self.backward_steps - 1, 0, -1):
            quad_range, upper_an_range, lower_an_range = self.set_quad_price_range(s_vec)
            # 数值积分计算区域
            y = self.ln_s_vec
            x = self.ln_s_vec[quad_range]
            v_vec[quad_range] = self.fft_step_backward(x, y, v_vec, dt)
            # 解析公式计算积分区域
            v_vec[upper_an_range] = prod.rebate[1] * np.exp(-r * dt * (self.backward_steps - step))
            v_vec[lower_an_range] = prod.rebate[0] * np.exp(-r * dt * (self.backward_steps - step))

        x = np.log(np.array([spot]))
        value = self.fft_step_backward(x, self.ln_s_vec, v_vec, dt)[0]
        return value
