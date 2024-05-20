#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from pricelib.common.utilities.utility import time_this, logging
from pricelib.common.product_base.option_base import OptionBase


class UnderlyingAsset(OptionBase):
    """标的资产"""

    def __init__(self, s0, *, engine=None):
        super().__init__()
        self.s0 = s0
        if engine is not None:
            self.set_pricing_engine(engine)

    def set_pricing_engine(self, engine):
        self.engine = engine
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def __repr__(self):
        """返回期权的描述"""
        return f"期初价格为{self.s0}的标的资产"

    @time_this
    def price(self, t=None, spot=None):
        if spot is not None:
            return spot
        return self.engine.process.spot()

    def delta(self, *args, **kwargs):
        delta = 1
        return delta

    def gamma(self, *args, **kwargs):
        gamma = 0
        return gamma

    def vega(self, *args, **kwargs):
        vega = 0
        return vega

    def theta(self, *args, **kwargs):
        theta = 0
        return theta

    def rho(self, *args, **kwargs):
        rho = 0
        return rho
