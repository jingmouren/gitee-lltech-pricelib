#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.utilities.enums import InOut, ExerciseType, PaymentType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import McEngine


class MCDoubleBarrierEngine(McEngine):
    """双边障碍期权 Monte Carlo 模拟定价引擎
    支持欧式观察(仅到期观察)/美式观察(整个有效期观察)；只支持离散观察(默认为每日观察)；
    支持现金返还；敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付"""

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
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        if _maturity_business_days == 0:  # 当估值日就是到期日时，直接计算payoff
            value = self.calc_maturity_payoff(prod, np.array([spot,]))[0]
            return value

        r = self.process.interest(_maturity)
        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        # 美式：
        if prod.exercise_type == ExerciseType.American:
            payoff = prod.callput.value * (paths[-1] - prod.strike) * prod.parti
            if prod.discrete_obs_interval is None:  # 每日观察
                obs_points = np.arange(0, _maturity_business_days + 1, 1)
            else:  # 均匀离散观察
                dt_step = prod.discrete_obs_interval * prod.t_step_per_year
                obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
                obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界

            if prod.inout == InOut.In:  # 美式双边敲入一定是到期支付
                knock_in_bool = np.any((paths[obs_points] <= prod.bound[0]) | (paths[obs_points] >= prod.bound[1]),
                                       axis=0)
                value = np.where(knock_in_bool, np.where(payoff < 0, 0, payoff), prod.rebate[0])
                value = np.mean(value) * math.exp(-r * _maturity)
            elif prod.inout == InOut.Out:  # 美式双边敲出
                step_index = np.tile(obs_points, (self.n_path, 1)).T
                hit_lower = np.min(
                    np.where(paths[1:] <= prod.bound[0], step_index.astype(int), _maturity_business_days + 1),
                    axis=0)
                hit_upper = np.min(
                    np.where(paths[1:] >= prod.bound[1], step_index.astype(int), _maturity_business_days + 1),
                    axis=0)
                if prod.payment_type == PaymentType.Hit:
                    value = np.mean(
                        np.where(hit_lower < hit_upper, prod.rebate[0] * np.exp(-r * hit_lower / prod.t_step_per_year),
                                 np.where(hit_lower > hit_upper,
                                          prod.rebate[1] * np.exp(-r * hit_upper / prod.t_step_per_year),
                                          np.where(payoff < 0, 0, payoff) * math.exp(-r * _maturity))))
                elif prod.payment_type == PaymentType.Expire:
                    value = np.mean(np.where(hit_lower < hit_upper, prod.rebate[0],
                                             np.where(hit_lower > hit_upper, prod.rebate[1],
                                                      np.where(payoff < 0, 0, payoff)))) * math.exp(-r * _maturity)
                else:
                    raise ValueError("PaymentType must be Hit or Expire")
            else:
                raise ValueError("InOut must be In or Out")
        # 欧式：
        elif prod.exercise_type == ExerciseType.European:
            value = self.calc_maturity_payoff(prod, paths[-1])
            value = np.mean(value) * math.exp(-r * _maturity)
        else:
            raise ValueError("ExerciseType must be American or European")
        return value

    @staticmethod
    def calc_maturity_payoff(prod, s_vec):
        """初始化终止时的期权价值，在障碍价格内侧，默认设定未敲入，未敲出
        Args:
            prod: Product产品对象
            s_vec: np.ndarray, 网格的价格格点
        Returns:
            value: np.ndarray, 期末价值向量
        """
        payoff = prod.callput.value * (s_vec - prod.strike) * prod.parti
        if prod.inout == InOut.Out:
            value = np.where(s_vec <= prod.bound[0], prod.rebate[0],
                             np.where(s_vec >= prod.bound[1], prod.rebate[1],
                                      np.where(payoff < 0, 0, payoff)))
        elif prod.inout == InOut.In:
            value = np.where((s_vec > prod.bound[0]) & (s_vec < prod.bound[1]), prod.rebate[0],
                             np.where(payoff < 0, 0, payoff))
        else:
            raise ValueError("InOut must be In or Out")
        return value
