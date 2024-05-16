#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 日期处理模块 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import (global_evaluation_date, set_evaluation_date, CN_CALENDAR, Schedule, BusinessConvention,
                      MonthAdjustment, EndOfMonthModify)

if __name__ == "__main__":
    print("默认估值日为最近交易日:", global_evaluation_date())
    set_evaluation_date(datetime.date(2022, 1, 5))
    print("设置估值日:", global_evaluation_date())
    start_date = datetime.date(2022, 1, 5)
    end_date = start_date + datetime.timedelta(days=365) * 1
    print("起止日期分别为", start_date, end_date, "相差自然日天数", (end_date - start_date).days, "相差交易日天数",
          CN_CALENDAR.business_days_between(start_date, end_date))

    obs_dates = Schedule(trade_calendar=CN_CALENDAR, start=start_date, end=end_date, freq='m', lock_term=3,
                         convention=BusinessConvention.FOLLOWING, correction=MonthAdjustment.YES,
                         endofmonthmodify=EndOfMonthModify.YES)
    print("Schedule生成1年期锁3的敲出观察日列表: (遇非交易日延后，跨月时修正为提前)")
    print(obs_dates.date_schedule)
    print("_".join([d.strftime("%Y%m%d") for d in obs_dates.date_schedule]))
    print(obs_dates.count_business_days(start_date))
    print(obs_dates.count_calendar_days(start_date))
    print("Schedule直接输入日期列表:")
    obs_dates2 = Schedule(trade_calendar=CN_CALENDAR, date_schedule=[datetime.date(2023, 5, 28),
                                                                     datetime.date(2023, 6, 30),
                                                                     datetime.date(2023, 7, 30),
                                                                     datetime.date(2023, 8, 31),
                                                                     datetime.date(2023, 9, 30),
                                                                     datetime.date(2023, 10, 31)])
    print(obs_dates2.date_schedule)
    print(obs_dates2.count_business_days(global_evaluation_date()))
    print(obs_dates2.count_calendar_days(global_evaluation_date()))
