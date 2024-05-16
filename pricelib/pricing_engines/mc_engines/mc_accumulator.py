#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import CallPut
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCAccumulatorEngine(McEngine):
    """累购累沽 Monte Carlo 模拟定价引擎
    仅支持起初估值，不支持持仓估值，因为没有记录历史累积次数"""

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
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        _obs_dates = np.array(prod.obs_dates.count_business_days(calculate_date))
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        # 敲出与行权日期
        barrier_out = np.ones(_obs_dates.size) * prod.barrier_out
        barrier_out = np.tile(barrier_out, (self.n_path, 1)).T
        strike = np.ones(_obs_dates.size) * prod.strike
        strike = np.tile(strike, (self.n_path, 1)).T
        if prod.callput == CallPut.Call:  # 累购
            knock_out_time_idx = np.argmax(paths[_obs_dates, :] >= barrier_out, axis=0)
            knock_in_time_idx = paths[_obs_dates, :] < strike
        else:  # 累沽
            knock_out_time_idx = np.argmax(paths[_obs_dates, :] <= barrier_out, axis=0)
            knock_in_time_idx = paths[_obs_dates, :] > strike

        knock_out_scenario = np.where(knock_out_time_idx > 0, True, False)
        knock_out_time_idx[~knock_out_scenario] = _obs_dates.size - 1
        knock_in_bool = np.where(np.arange(knock_in_time_idx.shape[0])[:, np.newaxis] > knock_out_time_idx, False,
                                 knock_in_time_idx)

        # 贴现因子
        discount_factor = np.empty(_obs_dates.size)
        for i, j in enumerate(_obs_dates):
            discount_factor[i] = self.process.interest.disc_factor(j / prod.t_step_per_year)
        n_underlying = np.sum(knock_in_bool, axis=0) * (prod.leverage_ratio - 1) + knock_out_time_idx
        exercise_price = paths[_obs_dates[knock_out_time_idx], np.arange(self.n_path)]

        payoff = n_underlying * (exercise_price - prod.strike) * discount_factor[knock_out_time_idx]
        return np.mean(payoff) * prod.callput.value
