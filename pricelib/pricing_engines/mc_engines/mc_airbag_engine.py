#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCAirbagEngine(McEngine):
    """安全气囊 Monte Carlo 模拟定价引擎
    安全气囊是美式观察向下敲入期权，到期支付
    只支持离散观察(默认为每日观察)
    """

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
        paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                    t_step_per_year=prod.t_step_per_year).copy()

        if prod.discrete_obs_interval is None:  # 每日观察
            obs_points = np.arange(0, _maturity_business_days + 1, 1)
        else:  # 均匀离散观察
            dt_step = prod.discrete_obs_interval * prod.t_step_per_year
            obs_points = np.flip(np.round(np.arange(_maturity_business_days, 0, -dt_step)).astype(int))
            obs_points = np.concatenate((np.array([0]), obs_points))
            obs_points[obs_points > _maturity_business_days] = _maturity_business_days  # 防止闰年导致的下标越界

        # 期末价值
        payoff = paths[-1] - prod.strike

        # 安全气囊，未敲入是call，已敲入是标的资产（上涨下跌可以设置不同的参与率）
        knock_in_bool = np.any(paths[obs_points] <= prod.barrier, axis=0)
        value = np.where(knock_in_bool,  # 敲入
                         np.where(payoff < 0, payoff * prod.knockin_parti,  # 敲入下跌参与率
                                  payoff * prod.reset_call_parti),  # 敲入重置后看涨参与率
                         np.where(payoff < 0, 0, payoff * prod.call_parti))  # 看涨参与率
        value = np.mean(value) * self.process.interest.disc_factor(_maturity)

        return value
