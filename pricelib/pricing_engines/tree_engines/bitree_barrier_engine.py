#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import InOut, UpDown, PaymentType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import BiTreeEngine
from pricelib.common.utilities.patterns import HashableArray


class BiTreeBarrierEngine(BiTreeEngine):
    """障碍期权二叉树定价引擎
    只支持离散观察(默认为每日观察)；敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付"""

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

        self.r = self.process.interest(tau)
        self.q = self.process.div(tau)
        self.vol = self.process.vol(tau, spot)

        n_step = round(tau * self.tree_branches)
        s_paths = self.path_generator(n_step, tau, spot).copy()
        v_grid = np.zeros(shape=s_paths.shape)

        # 设置观察时点obs_points
        if prod.discrete_obs_interval is None:
            obs_points = np.arange(0, n_step + 1, 1)[1:]
        else:
            obs_points = np.flip(np.round(np.arange(tau, 0, -prod.discrete_obs_interval) * self.tree_branches))
            obs_points = np.concatenate((np.array([0]), obs_points))
        obs_points = np.round(obs_points).astype(int)

        for j in range(n_step, -1, -1):
            if j == n_step:  # 到期日，设置终止条件
                if prod.inout == InOut.Out:
                    v_grid[:, -1] = np.maximum(prod.callput.value * (s_paths[:, -1] - prod.strike), 0) * prod.parti
                    if prod.updown == UpDown.Up:
                        v_grid[np.where(s_paths[:, -1] > prod.barrier), -1] = prod.rebate
                    elif prod.updown == UpDown.Down:
                        v_grid[np.where(s_paths[:, -1] < prod.barrier), -1] = prod.rebate
                    else:
                        raise ValueError("updown must be Up or Down")
                elif prod.inout == InOut.In:
                    v_grid[:, -1] = prod.rebate
                    if prod.updown == UpDown.Up:
                        upper_index = np.where(s_paths[:, -1] > prod.barrier)
                        v_grid[upper_index, -1] = np.maximum(
                            prod.callput.value * (s_paths[upper_index, -1] - prod.strike), 0) * prod.parti
                    elif prod.updown == UpDown.Down:
                        lower_index = np.where(s_paths[:, -1] < prod.barrier)
                        v_grid[lower_index, -1] = np.maximum(
                            prod.callput.value * (s_paths[lower_index, -1] - prod.strike), 0) * prod.parti
                    else:
                        raise ValueError("updown must be Up or Down")
            else:
                v_grid[:-1, j] = v_grid[:-1, j + 1] * self.p + v_grid[1:, j + 1] * (1 - self.p)
                if j in obs_points:
                    time_remain = self.dt * (n_step - j)
                    if prod.updown == UpDown.Up:
                        knock_inout = np.where(s_paths[:j, j] >= prod.barrier)
                    elif prod.updown == UpDown.Down:
                        knock_inout = np.where(s_paths[:j, j] <= prod.barrier)
                    else:
                        raise ValueError("updown must be Up or Down")
                    if prod.inout == InOut.In:
                        x = HashableArray(s_paths[knock_inout, j])
                        v_grid[knock_inout, j] = self.Vma(x, prod.strike, prod.callput.value,
                                                          time_remain) * prod.callput.value * prod.parti
                        v_grid[knock_inout, j] -= self.Vmb(x, prod.strike, prod.callput.value,
                                                           time_remain) * prod.strike * prod.callput.value * prod.parti
                    elif prod.inout == InOut.Out:
                        if prod.payment_type == PaymentType.Expire:  # 到期支付现金补偿
                            v_grid[knock_inout, j] = prod.rebate * np.exp(-self.r * time_remain)
                        else:  # 立即支付现金补偿
                            v_grid[knock_inout, j] = prod.rebate
                    else:
                        raise ValueError("inout must be In or Out")
                v_grid[:-1, j] *= np.exp(-self.r * self.dt)
        return v_grid[0, 0]
