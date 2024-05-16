#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import CallPut, InOut, UpDown, PaymentType, ExerciseType
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine
from ...products.digital.digital_option import DigitalOption
from ...products.barrier.barrier_option import BarrierOption
from .analytic_barrier_engine import AnalyticBarrierEngine
from .analytic_asset_or_nothing_engine import AnalyticAssetOrNothingEngine
from .analytic_cash_or_nothing_engine import AnalyticCashOrNothingEngine


class AnalyticAirbagEngine(AnalyticEngine):
    """安全气囊闭式解定价引擎
    安全气囊是美式观察向下敲入看涨期权，敲入前是call，敲入后payoff与持有标的资产相同(在参与率为1的情况下)
    相当于向下敲出看涨期权 + 向下触碰资产或无期权 的组合
    因为airbag解析解由障碍与二元解析解组合而成，所以敲入的下跌参与率与重置后看涨参与率必须相等"""

    # pylint: disable=invalid-name
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        assert prod.knockin_parti == prod.reset_call_parti, "Error: 安全气囊解析解由障碍与二元解析解组合而成，敲入的下跌参与率与重置后看涨参与率必须相等"
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        vol = self.process.vol(tau, self.process.spot())

        if prod.discrete_obs_interval is not None:
            # 均匀离散观察障碍期权，M. Broadie, P. Glasserman, S.G. Kou(1997) 在连续障碍期权解析解上加调整项，调整障碍价格水平
            # 指数上的beta = -zeta(1/2) / sqrt(2*pi) = 0.5826, 其中zeta是黎曼zeta函数
            if prod.updown == UpDown.Up:  # 向上
                barrier = prod.barrier * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            elif prod.updown == UpDown.Down:  # 向下
                barrier = prod.barrier * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            else:
                raise ValueError(f"Error: {prod} 无效的updown参数: {prod.updown}")
        else:  # 连续观察障碍期权
            barrier = prod.barrier

        DOC = BarrierOption(strike=prod.strike, barrier=barrier, rebate=0, inout=InOut.Out,
                            updown=UpDown.Down, callput=CallPut.Call, start_date=prod.start_date,
                            end_date=prod.end_date, discrete_obs_interval=None,
                            engine=AnalyticBarrierEngine(self.process))
        DIAsset = DigitalOption(strike=barrier, rebate=0, callput=CallPut.Put, start_date=prod.start_date,
                                end_date=prod.end_date, discrete_obs_interval=None, exercise_type=ExerciseType.American,
                                payment_type=PaymentType.Expire, engine=AnalyticAssetOrNothingEngine(self.process))
        DICash = DigitalOption(strike=barrier, rebate=-prod.strike, callput=CallPut.Put, start_date=prod.start_date,
                               end_date=prod.end_date, discrete_obs_interval=None, exercise_type=ExerciseType.American,
                               payment_type=PaymentType.Expire, engine=AnalyticCashOrNothingEngine(self.process))
        DOC_value = DOC.price(t=t, spot=spot) * prod.call_parti
        # 期权无本金，票据有本金。合约设立时为平值，这里加上rebate为负的行权价的二元看跌期权，相当于减去本金
        DIAsset_value = (DIAsset.price(t=t, spot=spot) + DICash.price(t=t, spot=spot)) * prod.knockin_parti
        return DOC_value + DIAsset_value
