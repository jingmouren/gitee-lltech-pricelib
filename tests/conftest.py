#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import sys
import os
import datetime
from pricelib import *

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

set_logging_handlers(to_file=False, to_console=False)  # 关闭日志输出


def init_bsm_process(evaluation_date: datetime.date, s=100, r=0.02, q=0.0, vol=0.2):
    """设置全局估值日，并初始化一个广义BSM随机过程"""
    # 设置全局估值日
    set_evaluation_date(evaluation_date)
    # 1. 市场数据，包括标的物价格、无风险利率、分红率、波动率
    spot_price = SimpleQuote(value=s, name="测试标的价格")
    riskfree = ConstantRate(value=r, name="无风险利率")
    dividend = ConstantRate(value=q, name="测试标的贴水率")
    volatility = BlackConstVol(vol, name="测试标的波动率")
    # 2. 随机过程，BSM价格动态
    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)
    return process
