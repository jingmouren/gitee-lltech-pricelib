#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from .timeutils import *
from .calendars import *

__all__ = ['global_evaluation_date', 'set_evaluation_date', 'get_last_business_date', 'CN_CALENDAR', 'Schedule',
           'BusinessConvention', 'MonthAdjustment', 'EndOfMonthModify', 'AnnualDays']


def get_last_business_date(trade_calendar=CN_CALENDAR):
    """获取估值日期"""
    today = datetime.datetime.now().date()
    if today in trade_calendar.holidays or today.weekday() > 4:
        evaluationDate = trade_calendar.advance(today, datetime.timedelta(days=-1))
    else:
        evaluationDate = today
    return evaluationDate


# 估值日期全局变量
__EvaluationDate = get_last_business_date()


def set_evaluation_date(date):
    """设置估值日期"""
    global __EvaluationDate
    __EvaluationDate = date


def global_evaluation_date():
    """获取估值日期"""
    return __EvaluationDate
