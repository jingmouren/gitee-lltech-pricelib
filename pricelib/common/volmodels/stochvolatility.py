#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from ..utilities.enums import VolType


class StochLocalVol:
    """随机局部波动率模型基类"""
    vol_type = VolType.SLV


class HestonSLV(StochLocalVol):
    """Heston随机局部波动率模型"""

    def __init__(self):
        pass
