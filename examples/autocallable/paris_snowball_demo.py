# !/user/bin/env python
# -*- coding: utf-8 -*-
""" 巴黎雪球 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import *


def run():
    """简易定价接口"""
    # 设置全局估值日
    set_evaluation_date(datetime.date(2022, 1, 5))
    # 定义产品：巴黎雪球
    option = ParisSnowball(maturity=2, lock_term=12, start_date=datetime.date(2022, 1, 5), s0=100,
                           barrier_out=103, barrier_in=75, coupon_out=0.129, knock_in_times=2, knockout_freq="w",
                           s=100, r=0.02, q=0.06, vol=0.16)
    res_dict = option.pv_and_greeks()
    res_dict["结构"] = str(option)
    return res_dict


if __name__ == '__main__':
    print(run())
