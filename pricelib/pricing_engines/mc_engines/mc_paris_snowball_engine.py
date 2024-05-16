#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType
from .mc_autocallable_engine import MCAutoCallableEngine


class MCParisSnowballEngine(MCAutoCallableEngine):
    """巴黎雪球MC定价引擎"""

    def _cal_knock_in_scenario(self, calculate_date):
        """计算每条路径敲入时间"""
        prod = self.prod
        if prod.status == StatusType.DownTouch:
            pass
        else:
            knock_in_obs_dates = prod.knock_in_obs_dates.count_business_days(calculate_date)
            knock_in_obs_dates = np.round(np.array([num for num in knock_in_obs_dates if num >= 0])).astype(int)
            # 判断某一条路径是否有敲入
            knock_in_level = np.tile(self._barrier_in, (self.n_path, 1)).T  # 变敲入（与敲出观察日同长度的列表）
            knock_in_bool = np.where(self.s_paths[knock_in_obs_dates - 1] < knock_in_level, 1, 0)
            knock_in_count = np.cumsum(knock_in_bool, axis=0)
            knock_in_scenario = np.max(knock_in_count, axis=0) >= prod.knock_in_times
            self.knock_in_scenario = knock_in_scenario & self.not_knock_out
