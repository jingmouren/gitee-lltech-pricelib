#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from functools import lru_cache
import numpy as np
from pricelib.common.utilities.enums import UpDown, InOut, PaymentType
from pricelib.common.utilities.patterns import HashableArray
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import QuadEngine


class QuadBarrierEngine(QuadEngine):
    """障碍期权数值积分定价引擎
    只支持离散观察(默认为每日观察)；敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付"""

    @lru_cache(maxsize=1)
    def set_quad_price_range(self, s_sliced, barrier, updown):
        """
        设置数值积分的区间(价格范围)
        Args:
            s_sliced: np.ndarray = self.s[:, step], 其中step是从后向前的步数
            barrier: float, 障碍价格
            updown: 向上/向下，UpDown枚举类
        Returns:
            quad_range: tuple, (np.ndarray, ), 数值积分的区间(索引)
            an_range: tuple, (np.ndarray, ), 解析式Vma、Vmb计算积分的区间(索引)
        """
        if updown == UpDown.Up:
            quad_range = np.where(s_sliced < barrier)
            an_range = np.where(s_sliced >= barrier)
        elif updown == UpDown.Down:
            quad_range = np.where(s_sliced > barrier)
            an_range = np.where(s_sliced <= barrier)
        else:
            raise ValueError("updown must be Up or Down")
        return quad_range, an_range

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
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        dt = prod.discrete_obs_interval
        self.backward_steps = int(tau / dt)
        self._check_method_params()

        # v_grid = np.empty(shape=(self.n_points, self.backward_steps + 1))
        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)
        # 初始化fft的对数价格向量及边界的对数价格向量
        self.init_grid(spot, vol, tau)
        s_vec = np.exp(self.ln_s_vec)

        # upper_barrier = self.n_max * spot
        # lower_barrier = 1
        # s_vec = np.linspace(lower_barrier, upper_barrier, self.n_points)
        # s_vec = HashableArray(np.tile(s_vec, reps=(self.backward_steps + 1, 1)).T)

        # 从后向前进行积分计算
        # 设定期末价值，在障碍价格内侧，默认设定未敲入，未敲出
        if prod.inout == InOut.Out:
            v_vec = np.maximum(prod.callput.value * (s_vec - prod.strike), 0) * prod.parti
            if prod.updown == UpDown.Up:
                v_vec[np.where(s_vec > prod.barrier)] = prod.rebate
            elif prod.updown == UpDown.Down:
                v_vec[np.where(s_vec < prod.barrier)] = prod.rebate
            else:
                raise ValueError("updown must be Up or Down")
        elif prod.inout == InOut.In:
            v_vec = prod.rebate * np.ones(len(s_vec))
            if prod.updown == UpDown.Up:
                upper_index = np.where(s_vec > prod.barrier)
                v_vec[upper_index] = np.maximum(prod.callput.value * (s_vec[upper_index] - prod.strike),
                                                0) * prod.parti
            elif prod.updown == UpDown.Down:
                lower_index = np.where(s_vec < prod.barrier)
                v_vec[lower_index] = np.maximum(prod.callput.value * (s_vec[lower_index] - prod.strike),
                                                0) * prod.parti
            else:
                raise ValueError("updown must be Up or Down")
        else:
            raise ValueError("inout must be In or Out")

        for step in range(self.backward_steps - 1, 0, -1):
            quad_range, an_range = self.set_quad_price_range(s_vec, prod.barrier, prod.updown)
            # 数值积分计算区域
            y = self.ln_s_vec
            x = self.ln_s_vec[quad_range]
            v_vec[quad_range] = self.fft_step_backward(x, y, v_vec, dt)
            # v_grid[:, step] = np.minimum(v_grid[:, step], 1e250)  # 防止价格极大和极小时，期权价值趋近于无穷大
            # 解析公式计算积分区域
            if prod.inout == InOut.In:
                x = HashableArray(self.ln_s_vec[an_range])
                v_vec[an_range] = self.Vma(np.exp(x), prod.strike, prod.callput.value,
                                           dt * (self.backward_steps - step)) * prod.callput.value * prod.parti
                v_vec[an_range] -= self.Vmb(np.exp(x), prod.strike, prod.callput.value, dt * (
                        self.backward_steps - step)) * prod.strike * prod.callput.value * prod.parti
            elif prod.inout == InOut.Out:
                if prod.payment_type == PaymentType.Expire:  # 到期支付现金补偿
                    v_vec[an_range] = prod.rebate * np.exp(-r * dt * (self.backward_steps - step))
                else:  # 立即支付现金补偿
                    v_vec[an_range] = prod.rebate
            else:
                raise ValueError("inout must be In or Out")

        x = np.log(np.array([spot]))
        value = self.fft_step_backward(x, self.ln_s_vec, v_vec, dt)[0]
        return value
