#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import InOut, PaymentType
from .fdm_barrier_engine import FdmBarrierEngine


class FdmDoubleSharkEngine(FdmBarrierEngine):
    """双鲨结构PDE有限差分法定价引擎"""
    inout = InOut.Out  # 双鲨结构属于敲出期权

    def _init_boundary_condition(self, smax, maturity):
        """初始化边界条件
        双鲨结构，上方边界条件是call，下方边界条件是put
        Args:
            smax: float, 有限差分网格上边界的价格最大值
            maturity: float, 到期时间
        Returns:
            self.fn_callput: List[Callable], 0~smax网格的边界条件
            self.fn_bound: List[Callable], 障碍价格网格的边界条件
        """
        # 生成无障碍价格的边界条件：在s=0处，call近似为0; 在smax处，put近似为0。根据看涨看跌平价 call-put = S*exp(-qT) - K*exp(-rT),
        # 可得对应的put(0) = K * exp(-rT), call(smax) = smax * exp(-qT) - K * exp(-rT)
        self.fn_callput = [
            lambda u: (self.prod.strike[0] * self.process.interest.disc_factor(maturity, u)) * self.prod.parti[0],
            lambda u: (smax * self.process.div.disc_factor(maturity, u) -
                       self.prod.strike[1] * self.process.interest.disc_factor(maturity, u)) * self.prod.parti[1]]
        # 生成障碍价格的Dirichlet边界条件： 第u天触碰障碍产生的补偿，折现到第u天的价值。
        # 返回一个包含两个lambda函数的列表，列表索引分别为[0, 1]，每个lambda函数的形参r=self.rebate中的r补偿值
        # 需要注意的是，敲出期权rebate[0]是下边界补偿，敲出期权rebate[1]是上边界补偿；敲入期权rebate[0]是敲入补偿，敲入期权rebate[1]没有用处。
        self.fn_bound = []
        for r in self.rebate:
            if self.prod.payment_type == PaymentType.Hit:  # 触碰障碍立即支付
                self.fn_bound.append(lambda u: r)
            else:  # 'PaymentMethod.Expire' 触碰障碍后，到期时支付
                self.fn_bound.append(lambda u: r * self.process.interest.disc_factor(maturity, u))

    def _init_terminal_condition(self, s_vec):
        """初始化终止条件
        双鲨结构，上方是call，下方是put"""
        call_put_payoff = (np.maximum(s_vec - self.prod.strike[1], 0) * self.prod.parti[1] +
              np.maximum(self.prod.strike[0] - s_vec, 0) * self.prod.parti[0])
        yv = np.where(s_vec >= self.bound[1], self.rebate[1],
                      np.where(s_vec <= self.bound[0], self.rebate[0], call_put_payoff))
        return yv
