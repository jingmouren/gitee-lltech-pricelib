#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.time import global_evaluation_date


class QuadPortfolioEngine(QuadEngine):
    """香草组合积分法定价引擎"""

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        calculate_date = global_evaluation_date() if t is None else t
        tau = (prod.end_date - calculate_date).days / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, spot)
        self._check_method_params()

        # 设定期末价值
        self.init_grid(spot, vol, tau)
        v_vec = np.zeros(self.ln_s_vec.shape)
        for vanilla, position in prod.vanilla_list:
            v_vec += np.maximum(vanilla.callput.value * (np.exp(self.ln_s_vec) - vanilla.strike), 0) * position
        # 设置积分法engine参数
        self.set_quad_params(r=r, q=q, vol=vol)
        # 回溯计算
        return self.fft_step_backward(np.log(np.array([spot])), self.ln_s_vec, v_vec, tau)[0]
