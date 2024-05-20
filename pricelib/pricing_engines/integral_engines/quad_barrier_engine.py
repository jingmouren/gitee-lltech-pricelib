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

        v_grid = np.empty(shape=(self.n_points, self.backward_steps + 1))
        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)

        upper_barrier = self.n_max * spot
        lower_barrier = 1
        s_vec = np.linspace(lower_barrier, upper_barrier, self.n_points)
        s_vec = HashableArray(np.tile(s_vec, reps=(self.backward_steps + 1, 1)).T)

        # 从后向前进行积分计算
        # 设定期末价值，在障碍价格内侧，默认设定未敲入，未敲出
        if prod.inout == InOut.Out:
            v_grid[:, -1] = np.maximum(prod.callput.value * (s_vec[:, -1] - prod.strike), 0) * prod.parti
            if prod.updown == UpDown.Up:
                v_grid[np.where(s_vec[:, -1] > prod.barrier), -1] = prod.rebate
            elif prod.updown == UpDown.Down:
                v_grid[np.where(s_vec[:, -1] < prod.barrier), -1] = prod.rebate
            else:
                raise ValueError("updown must be Up or Down")
        elif prod.inout == InOut.In:
            v_grid[:, -1] = prod.rebate
            if prod.updown == UpDown.Up:
                upper_index = np.where(s_vec[:, -1] > prod.barrier)
                v_grid[upper_index, -1] = np.maximum(prod.callput.value * (s_vec[upper_index, -1] - prod.strike),
                                                     0) * prod.parti
            elif prod.updown == UpDown.Down:
                lower_index = np.where(s_vec[:, -1] < prod.barrier)
                v_grid[lower_index, -1] = np.maximum(prod.callput.value * (s_vec[lower_index, -1] - prod.strike),
                                                     0) * prod.parti
            else:
                raise ValueError("updown must be Up or Down")
        else:
            raise ValueError("inout must be In or Out")

        for step in range(self.backward_steps - 1, 0, -1):
            quad_range, an_range = self.set_quad_price_range(s_vec[:, step], prod.barrier, prod.updown)
            # 数值积分计算区域
            y = s_vec[:, step + 1]
            x = s_vec[quad_range, step]
            v_grid[quad_range, step] = self.step_backward(x, y, v_grid[:, step + 1], dt)
            v_grid[:, step] = np.minimum(v_grid[:, step], 1e250)  # 防止价格极大和极小时，期权价值趋近于无穷大
            # 解析公式计算积分区域
            if prod.inout == InOut.In:
                x = HashableArray(s_vec[an_range, step])
                v_grid[an_range, step] = self.Vma(x, prod.strike, prod.callput.value,
                                                  dt * (self.backward_steps - step)) * prod.callput.value * prod.parti
                v_grid[an_range, step] -= self.Vmb(x, prod.strike, prod.callput.value, dt * (
                        self.backward_steps - step)) * prod.strike * prod.callput.value * prod.parti
            elif prod.inout == InOut.Out:
                if prod.payment_type == PaymentType.Expire:  # 到期支付现金补偿
                    v_grid[an_range, step] = prod.rebate * np.exp(-r * dt * (self.backward_steps - step))
                else:  # 立即支付现金补偿
                    v_grid[an_range, step] = prod.rebate
            else:
                raise ValueError("inout must be In or Out")
        y = s_vec[:, 1]
        x = np.array(spot)
        value = self.step_backward(x, y, v_grid[:, 1], dt)[0]
        return value
