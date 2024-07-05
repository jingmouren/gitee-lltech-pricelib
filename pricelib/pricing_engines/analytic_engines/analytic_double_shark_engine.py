#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from pricelib.common.utilities.enums import PaymentType, InOut, CallPut, ExerciseType, TouchType
from pricelib.common.processes import StochProcessBase
from pricelib.common.pricing_engine_base import AnalyticEngine
from ...products.barrier.double_barrier_option import DoubleBarrierOption
from ...products.digital.double_digital_option import DoubleDigitalOption
from .analytic_double_barrier_engine import AnalyticDoubleBarrierEngine
from .analytic_double_digital_engine import AnalyticDoubleDigitalEngine


class AnalyticDoubleSharkEngine(AnalyticEngine):
    """双鲨结构闭式解定价引擎, 由双边敲出看涨与双边敲出看跌组合而成，现金返还使用美式双接触闭式解
        Haug(1998) Put-Call Barrier Transformations 对于标的持有成本为0的期权(期货期权)，利用认购-认沽障碍期权对称性，
                                                    利用单边障碍期权解析解的组合给双障碍期权定价。
        Ikeda and Kunitomo(1992)将双障碍期权表示为加权正态分布函数的无限集，但是仅当行权价在障碍范围内时，公式才是成立的。
    Haug(1998)的解析解与Ikeda and Kunitomo(1992)的结果几乎一致，两者对比，
        Ikeda and Kunitomo(1992)适用于标的持有成本不为0的情形, Haug(1998)适用于行权价在障碍范围之外的情形。
    由于现金返还使用了美式双接触闭式解Hui(1996)，双接触期权的下边界payoff和上边界payoff必须相等，
        所以双鲨闭式解的高低敲出现金返还也必须相等，现金返还到期支付
    只支持美式观察(整个有效期观察)；支持连续观察/离散观察(默认为每日观察)
    """

    def __init__(self, stoch_process: StochProcessBase = None, formula_type="Ikeda&Kunitomo1992",
                 series_num=10, delta1=0, delta2=0, *,
                 s=None, r=None, q=None, vol=None):
        """
        Args:
            stoch_process: 随机过程StochProcessBase对象
            self.formula_type: 解析解类型，Ikeda&Kunitomo1992或Haug1998
        其他三个参数是Ikeda&Kunitomo1992的参数：
            self.series_num: 用于计算近似解的级数项 默认 i = -10 ~ 10
            self.delta1: 上边界的曲率
            self.delta2: 下边界的曲率，
                当delta1 = delta2 = 0时，对应两条平行边界
                当delta1 < 0 < delta2 时，对应下边界随时间呈指数增长，而上百年姐随时间呈指数衰减
                当delta1 > 0 > delta2 时，对应一个向下凸的下边界和一个向上凸的上边界
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process=stoch_process, s=s, r=r, q=q, vol=vol)
        self.formula_type = formula_type
        self.series_num = series_num
        self.delta1 = delta1
        self.delta2 = delta2

    # pylint: disable=invalid-name
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert (prod.exercise_type == ExerciseType.American and prod.payment_type == PaymentType.Expire), \
            "Error: 双鲨解析解仅支持美式观察、到期支付的情形。"
        assert (prod.strike[0] <= prod.strike[1]) and (
                prod.bound[0] < prod.bound[1]), "Error: 双鲨结构输入的行权价和障碍价必须是递增的，不能是None"

        DOC = DoubleBarrierOption(exercise_type=ExerciseType.American, payment_type=prod.payment_type, inout=InOut.Out,
                                  callput=CallPut.Call, strike=prod.strike[1], bound=prod.bound, rebate=(0, 0),
                                  window=prod.window, start_date=prod.start_date, end_date=prod.end_date,
                                  discrete_obs_interval=prod.discrete_obs_interval,
                                  engine=AnalyticDoubleBarrierEngine(self.process, formula_type=self.formula_type,
                                                                     series_num=self.series_num, delta1=self.delta1,
                                                                     delta2=self.delta2))
        DOP = DoubleBarrierOption(exercise_type=ExerciseType.American, payment_type=prod.payment_type, inout=InOut.Out,
                                  callput=CallPut.Put, strike=prod.strike[0], bound=prod.bound, rebate=(0, 0),
                                  window=prod.window, start_date=prod.start_date, end_date=prod.end_date,
                                  discrete_obs_interval=prod.discrete_obs_interval,
                                  engine=AnalyticDoubleBarrierEngine(self.process, formula_type=self.formula_type,
                                                                     series_num=self.series_num, delta1=self.delta1,
                                                                     delta2=self.delta2))
        double_touch = DoubleDigitalOption(bound=prod.bound, rebate=prod.rebate, exercise_type=ExerciseType.American,
                                           payment_type=prod.payment_type, touch_type=TouchType.Touch,
                                           start_date=prod.start_date, end_date=prod.end_date,
                                           discrete_obs_interval=prod.discrete_obs_interval,
                                           engine=AnalyticDoubleDigitalEngine(self.process, series_num=10))
        result = (DOC.price(t=t, spot=spot) * prod.parti[1] +
                  DOP.price(t=t, spot=spot) * prod.parti[0] +
                  double_touch.price(t=t, spot=spot))
        return result
