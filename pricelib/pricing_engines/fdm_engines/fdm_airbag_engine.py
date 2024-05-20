#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from .fdm_barrier_engine import FdmBarrierEngine


class FdmAirbagEngine(FdmBarrierEngine):
    """安全气囊PDE有限差分法定价引擎
    安全气囊是向下敲入看涨期权，到期支付
    支持连续观察/离散观察(默认为每日观察)"""

    def _init_boundary_condition(self, smax, maturity):
        """初始化边界条件
        安全气囊，未敲入是call，已敲入是标的资产
        Args:
            smax: float, 有限差分网格上边界的价格最大值
            maturity: float, 到期时间
        Returns:
            self.fn_callput: List[Callable], 0~smax网格的边界条件
            self.fn_bound: List[Callable], 障碍价格网格的边界条件
        """
        # 生成无障碍价格的边界条件： 0~smax
        self.fn_callput = [lambda u: -self.prod.strike * self.prod.knockin_parti *
                                     self.process.interest.disc_factor(maturity, u),
                           lambda u: (smax * self.process.div.disc_factor(maturity, u)
                                      - self.prod.strike * self.process.interest.disc_factor(maturity, u)
                                      ) * self.prod.knockin_parti]
        # 生成障碍价格的Dirichlet边界条件： bound[0]~smax,  第u天触碰障碍产生的payoff，折现到第u天的价值。
        # 返回一个包含两个lambda函数的列表，列表索引分别为[0, 1]，每个lambda函数的形参r=self.rebate中的r补偿值
        self.fn_bound = []
        for i, r in enumerate(self.rebate):
            if self.bound[i] in [-np.inf, np.inf]:  # 安全气囊未敲入call的上边界是smax, 没有上方的障碍，即障碍价无穷
                self.fn_bound.append(lambda u: (smax * self.process.div.disc_factor(maturity, u) -
                                                self.prod.strike * self.process.interest.disc_factor(maturity, u))
                                               * self.prod.call_parti)
            else:
                # 'PaymentMethod.Expire' 触碰下方的障碍后，到期时支付，安全气囊已敲入是标的资产
                if self.prod.discrete_obs_interval is None:  # 连续观察, 价格下边界是self.bound[i]
                    self.fn_bound.append(lambda u: (self.bound[i] - self.prod.strike) * self.prod.knockin_parti
                                                   * self.process.interest.disc_factor(maturity, u))
                else:  # 离散观察，价格下边界是0
                    self.fn_bound.append(
                        lambda u: - self.prod.strike * self.prod.knockin_parti * self.process.interest.disc_factor(
                            maturity, u))

    def _init_terminal_condition(self, s_vec):
        """初始化终止条件
        连续观察（网格上下界是障碍价格）：初始化终止时已敲入的期权价值，安全气囊敲入后是标的资产
        离散观察（网格上下界是0和smax）：初始化终止时的期权价值
        """
        yv = np.where(s_vec - self.prod.strike > 0,
                      (s_vec - self.prod.strike) * self.prod.reset_call_parti,  # 敲入的重置后看涨参与率
                      (s_vec - self.prod.strike) * self.prod.knockin_parti)  # 敲入的下跌参与率
        return yv

    def _init_no_touch_terminal_condition(self, s_vec, p=0, maturity=0):
        """初始化障碍观察区间终点未敲入的期权价值，安全气囊，未敲入是call"""
        if self.prod.discrete_obs_interval is None:  # 连续观察
            yv = np.maximum(s_vec - self.prod.strike, 0) * self.process.interest.disc_factor(maturity,  # 未敲入的看涨参与率
                                                                                             p) * self.prod.call_parti
        else:  # 离散观察
            yv = np.where(s_vec - self.bound[0] > 0,
                          np.maximum(s_vec - self.prod.strike, 0) * self.prod.call_parti,  # 未敲入的看涨参与率
                          (s_vec - self.prod.strike) * self.prod.knockin_parti)  # 敲入的下跌参与率
        return yv
