#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import datetime
from pricelib import (global_evaluation_date, set_evaluation_date, CN_CALENDAR, Schedule, BusinessConvention,
                      MonthAdjustment, EndOfMonthModify)


def test_calendar_schedule():
    set_evaluation_date(datetime.date(2022, 1, 5))
    assert global_evaluation_date() == datetime.date(2022, 1, 5)
    start_date = datetime.date(2022, 1, 5)
    end_date = start_date + datetime.timedelta(days=365) * 1
    assert end_date == datetime.date(2023, 1, 5)
    assert CN_CALENDAR.business_days_between(start_date, end_date) == 243

    obs_dates = Schedule(trade_calendar=CN_CALENDAR, start=start_date, end=end_date, freq='m',
                         lock_term=3, convention=BusinessConvention.FOLLOWING, correction=MonthAdjustment.YES,
                         endofmonthmodify=EndOfMonthModify.YES)
    assert obs_dates.date_schedule == [datetime.date(2022, 4, 6), datetime.date(2022, 5, 5),
                                       datetime.date(2022, 6, 6), datetime.date(2022, 7, 5),
                                       datetime.date(2022, 8, 5), datetime.date(2022, 9, 5),
                                       datetime.date(2022, 10, 10), datetime.date(2022, 11, 7),
                                       datetime.date(2022, 12, 5), datetime.date(2023, 1, 5)]
    assert obs_dates.count_business_days(start_date) == [58, 76, 97, 118, 141, 162, 181, 201, 221, 243]
    assert obs_dates.count_calendar_days(start_date) == [91, 120, 152, 181, 212, 243, 278, 306, 334, 365]
