#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np

from ..utilities.enums import VolType
from ..utilities.patterns import Observable


class LocalVolSurface(Observable):
    vol_type = VolType.LV

    def __init__(self, expirations: np.ndarray, strikes: np.ndarray,
                 impliedvol: np.ndarray = None, volval: np.ndarray = None):
        """局部波动率曲面
        ivol: BS隐含波动率曲面矩阵
        expirations；时间格点array
        strikes：价格格点array
        lvol: 局部波动率曲面矩阵，行对应时间t，列对应价格s
        """
        super().__init__()
        self.impliedvol = impliedvol
        self.expirations = expirations  # 一维array, 时间格点, 升序排列
        self.strikes = strikes  # 二维矩阵，每行是时间，每列是执行价，执行价必须是升序排列的
        self.volval = volval  # np.array, 存储局部波动率矩阵

    def __call__(self, t, s):
        """返回指定时间t和制定价格s对应的局部波动率。t是float，s是array"""
        # 在已排序的数组self.t_grids中查找插入t的index，以维持数组的排序
        t_idx = np.searchsorted(self.expirations[:-1], t)
        s_idx = np.argmin(np.abs(self.strikes - s.reshape(-1, 1)), axis=1)
        lv = self.volval[t_idx, s_idx]
        return lv

    def dupire_formula(self):
        """TODO: 用Dupire公式计算局部波动率曲面"""
