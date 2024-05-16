#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta
from ..utilities.enums import EngineType
from .engine_base import PricingEngineBase


class AnalyticEngine(PricingEngineBase, metaclass=ABCMeta):
    """解析定价引擎基类"""
    engine_type = EngineType.AnEngine
