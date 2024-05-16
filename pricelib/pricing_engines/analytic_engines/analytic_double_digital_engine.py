#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import math
import numpy as np
from pricelib.common.time import global_evaluation_date
from pricelib.common.utilities.enums import TouchType, ExerciseType, PaymentType
from pricelib.common.processes import StochProcessBase
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticDoubleDigitalEngine(AnalyticEngine):
    """双边二元期权闭式级数近似解定价引擎
    Hui(1996) One-Touch Double Barrier Binary Option Values
    只支持美式双接触/双不接触, 到期支付"""

    def __init__(self, stoch_process: StochProcessBase = None, series_num=10, *,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: 随机过程StochProcessBase对象
            series_num: int，计算Hui(1996)近似解时级数项的数量
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.i_vec = np.array(range(1, series_num + 1))  # 用于计算近似解的级数项

    # pylint: disable=invalid-name, too-many-locals
    def calc_present_value(self, prod, t=None, spot=None):
        assert (prod.payment_type == PaymentType.Expire and
                prod.exercise_type == ExerciseType.American), "Warning: Hui(1996)双边二元期权级数近似解, 只支持美式双接触/双不接触, 到期支付"

        calculate_date = global_evaluation_date() if t is None else t
        _maturity = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year

        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(_maturity)
        q = self.process.div(_maturity)
        vol = self.process.vol(_maturity, spot)

        if prod.discrete_obs_interval is not None:
            # 均匀离散观察，M. Broadie, P. Glasserman, S.G. Kou(1997) 在连续障碍期权解析解上加调整项，调整障碍价格水平
            # 指数上的beta = -zeta(1/2) / sqrt(2*pi) = 0.5826, 其中zeta是黎曼zeta函数
            bound = [None, None]
            bound[1] = prod.bound[1] * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            bound[0] = prod.bound[0] * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
        else:  # 连续观察
            bound = prod.bound

        Z = math.log(bound[1] / bound[0])
        alpha = - 0.5 * (2 * (r - q) / vol ** 2 - 1)
        beta = - 0.25 * (2 * (r - q) / vol ** 2 - 1) ** 2 - 2 * r / vol ** 2

        value0 = self.i_vec * np.pi / Z
        value1 = 2 * np.pi * self.i_vec * prod.rebate[0] / Z ** 2
        value2 = (spot / bound[0]) ** alpha - np.power(-1, self.i_vec) * (spot / bound[1]) ** alpha
        value3 = alpha ** 2 + value0 ** 2
        value4 = np.sin(value0 * np.log(spot / bound[0]))
        value5 = np.exp(-0.5 * vol ** 2 * _maturity * (value0 ** 2 - beta))
        value_series = value1 * value2 * value4 * value5 / value3  # 期权价值
        double_no_touch = np.sum(value_series)

        if prod.touch_type == TouchType.NoTouch:
            return double_no_touch
        if prod.touch_type == TouchType.Touch:
            assert prod.rebate[0] == prod.rebate[1], "Error: Hui(1996)双边二元期权闭式解, 双接触期权补偿回报必须相等"
            return prod.rebate[0] * np.exp(-r * _maturity) - double_no_touch
        raise ValueError("Hui(1996)双边二元期权闭式解, 未知的触碰类型")
