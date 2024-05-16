#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0

日期处理主要功能实现：

1. Date: 日期类
datetime.datetime(year,month,day,hour=0,minute=0,second=0,microsecond=0,tzinfo=None,*,fold=0)
支持比较大小: ==, !=, >, <, >=, <=
支持获取年月日星期:year,month,day, weekday(), isoweekday()
支持加减时间: + timedelta(days=1) - relativedelta(weeks=1) + relativedelta(months=1) - relativedelta(years=1)

2. Period: 时间段类
2.1 datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
主要使用timedelta的days，计算自然日时间差: (date1 - date2).days  若含头含尾，需+1

2.2 relativedelta: 处理周、月、年的时间差
    (self, dt1=None, dt2=None,
    years=0, months=0, days=0, leapdays=0, weeks=0,
    hours=0, minutes=0, seconds=0, microseconds=0,
    year=None, month=None, day=None, weekday=None,
    yearday=None, nlyearday=None,
    hour=None, minute=None, second=None, microsecond=None)
计算相对时间差: relativedelta(Date1, Date2)

3. Calendar: 交易日历类
自定义，详见下方代码

4. Schedule: 日期序列类
自定义，详见下方代码

todo: 日期规则，DayCounters
"""
from enum import Enum, unique
import datetime
from dateutil.relativedelta import relativedelta
from dateutil import rrule


class Calendar:
    """某个国家或地区的交易日历
        advance方法: new_date = calender.advance(old_date, period)
        addHoliday: 添加节假日
        removeHoliday: 移除节假日
        businessDaysBetween:获取交易日天数
    """

    def __init__(self):
        """初始化交易日历"""
        self.holidays = set()  # 节假日期集合

    def advance(self, date, period):
        """计算在当前交易日历下，指定日期加上一段时间后，生成的新日期
        Args:
            date: datetime.date，指定日期
            period: datetime.timedelta，时间段，单位是天
        Returns:
            new_date: datetime.date，新日期
        """
        if isinstance(date, datetime.datetime):
            date = date.date()
        new_date = date
        while period.days > 0:
            new_date += datetime.timedelta(days=1)
            if new_date.weekday() < 5 and new_date not in self.holidays:
                period -= datetime.timedelta(days=1)
        while period.days < 0:
            new_date -= datetime.timedelta(days=1)
            if new_date.weekday() < 5 and new_date not in self.holidays:
                period += datetime.timedelta(days=1)
        return new_date

    def business_list_between(self, dt1, dt2):
        """计算在当前交易日历下，dt1和dt2之间有哪些交易日(含头含尾)
        Args:
            dt1: datetime.datetime或datetime.date，起始日
            dt2: datetime.datetime或datetime.date，终止日
        Returns:
            business_days: List[datetime.date]，交易日列表(含头含尾)
        """
        if isinstance(dt1, datetime.datetime):
            dt1 = dt1.date()
        if isinstance(dt2, datetime.datetime):
            dt2 = dt2.date()
        days = rrule.rrule(freq=rrule.DAILY, byweekday=(rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR),
                           dtstart=dt1, until=dt2)  # rrule生成的是datetime.datetime对象
        business_days = [day.date() for day in days if day.date() not in self.holidays]  # 0-4代表周一到周五
        return business_days

    def business_days_between(self, dt1, dt2):
        """计算在当前交易日历下，dt1和dt2之间有多少个交易日(不含头，含尾)
        Args:
            dt1: datetime.datetime或datetime.date，起始日
            dt2: datetime.datetime或datetime.date，终止日
        Returns:
            int，交易日天数(不含头，含尾)
        """
        return len(self.business_list_between(dt1, dt2)) - 1

    def add_holiday(self, date):
        """添加节假日
        Args:
            date: datetime.date，节假日
        Returns: None
        """
        self.holidays.add(date)

    def remove_holiday(self, date):
        """移除非节假日
        Args:
            date: datetime.date，非节假日
        Returns: None
        """
        self.holidays.remove(date)


@unique
class BusinessConvention(Enum):
    """枚举类，用于指定日期调整规则
    例如Scheduler生成敲出观察日时，遇到非交易日，需要提前/延后一个工作日"""
    PRECEDING = "提前"
    FOLLOWING = "延后"


@unique
class MonthAdjustment(Enum):
    """遇到非交易日调整时，如果跨月，是否调整为不换月"""
    YES = 0
    NO = 1


@unique
class EndOfMonthModify(Enum):
    """到期日是非交易日，调整到期日时，恰逢起始日是某月最后一个交易日，终止日是非交易日，向前和向后调整终止日后月份不同，选择向前调整"""
    YES = 0
    NO = 1


# 每周 - 7日，每双周 - 14日，每月 - 30.416日，每季度 - 91.25日
FREQS = {'w': 7, 'd': 14, 'm': 30.416, 's': 91.25}
FREQS_RELATIVEDELTA = {'w': relativedelta(weeks=1), 'd': relativedelta(weeks=2),
                       'm': relativedelta(months=1), 's': relativedelta(months=3)}


class Schedule:
    """日期序列类
    根据起止日期、观察频率和交易日规则，生成观察日列表
    根据某个基准日期，计算self.date_schedule中的每个日期距离基准日期几个自然日(交易日)，返回一个List[int]，列表中是自然日(交易日)数量
    """

    def __init__(self, trade_calendar, start=None, end=None, freq='m', lock_term=1,
                 convention=BusinessConvention.FOLLOWING, correction=MonthAdjustment.YES,
                 endofmonthmodify=EndOfMonthModify.YES, date_schedule=None):
        """初始化Schedule
        Args:
            trade_calendar: 某个市场的Calendar交易日历对象
            start: datetime.date，起始日期
            end: datetime.date，终止日期
            freq: str，观察频率，'w' - 每周，'d' - 每双周，'m' - 每月，'s' - 每季度
            lock_term: int，锁定期，单位是观察频率，比较常见的是敲出观察锁3月
            convention: 日期调整规则，BusinessConvention枚举类，'提前'或'延后'
            correction: 非交易日跨月调整规则，MonthAdjustment枚举类，YES - 调整为不换月，NO - 不调整
            endofmonthmodify: 到期日跨月调整规则，EndOfMonthModify枚举类，YES - 向前调整，NO - 不调整
            date_schedule: List[datetime.date]，日期列表，可以直接外部输入
                           默认为None，会根据start, end, freq, lock_term, convention, correction, endofmonthmodify生成
        """
        self.start = start
        self.end = end
        self.freq = freq
        self.lock_term = lock_term
        self.convention = convention
        self.correction = correction
        self.endofmonthmodify = endofmonthmodify
        self.trade_calendar = trade_calendar
        args = (start, end, freq, lock_term, convention, correction, endofmonthmodify)
        if date_schedule is None:  # 如果没有直接输入日期列表，检查参数是否有None，生成date_schedule
            assert any(arg is not None for arg in args), "Error: Schedule输入参数不全，有None"
            self.date_schedule = self.generate_schedule()
        else:  # 直接输入日期列表
            self.date_schedule = date_schedule
        self.count_business, self.count_calendar = None, None

    def generate_schedule(self):
        """根据起止日期、频率和交易日规则，生成日期列表"""
        # 如果到期日非交易日，先按照月终规则调整到期日：如果起始日是某月最后一个交易日，终止日是非交易日，向前和向后调整终止日后月份不同，选择向前调整
        if (self.endofmonthmodify == EndOfMonthModify.YES and (
                self.end in self.trade_calendar.holidays or self.end.weekday() > 4)):
            next_startday = self.trade_calendar.advance(self.start, datetime.timedelta(days=1))
            # 如果起始日是该月最后一个交易日
            if self.start.month != next_startday.month:
                prev_endday = self.trade_calendar.advance(self.end, datetime.timedelta(days=-1))
                next_endday = self.trade_calendar.advance(self.end, datetime.timedelta(days=1))
                # 如果到期日向前调整和向后调整跨月，向前调整
                if prev_endday.month != next_endday.month:
                    self.end = prev_endday
        # 起始日（含）和终止日（含）之间所有自然日期，统计总日数N
        n_dates = (self.end - self.start).days + 1
        # 根据敲出观察频率确定敲出观察日之间间隔日数f：每月 - 30日，每季度 - 90，每周 - 7，每双周 - 14
        fq = FREQS[self.freq]
        # 根据间隔日数计算敲出观察次数
        n_obs = round(n_dates / fq) + 1
        # 根据观察次数和总日数，使用relativedelta每次加上一个间隔，推算各个日期
        n_obs_dates = [self.start + i * FREQS_RELATIVEDELTA[self.freq] for i in range(n_obs)][self.lock_term:]
        # 根据观察开始时间i，筛选观察日序号
        self.date_schedule = []
        for i, obs_date in enumerate(n_obs_dates):
            if obs_date in self.trade_calendar.holidays or obs_date.weekday() > 4:  # 如果 obs_date不是交易日
                if self.convention == BusinessConvention.PRECEDING:  # 提前
                    obs_date_adj = self.trade_calendar.advance(obs_date, datetime.timedelta(days=-1))
                    if self.correction == MonthAdjustment.YES and obs_date_adj.month != obs_date.month:
                        # 如果提前导致换月，改为延后
                        obs_date_adj = self.trade_calendar.advance(obs_date, datetime.timedelta(days=1))
                else:  # 延后
                    obs_date_adj = self.trade_calendar.advance(obs_date, datetime.timedelta(days=1))
                    if self.correction == MonthAdjustment.YES and obs_date_adj.month != obs_date.month:
                        # 如果延后导致换月，改为提前
                        obs_date_adj = self.trade_calendar.advance(obs_date, datetime.timedelta(days=-1))
                self.date_schedule.append(obs_date_adj)
            else:
                self.date_schedule.append(obs_date)
        # 若最后一个观察日为非交易日，将其延后之后，如果最后一个观察日超过终止日，最后一个观察日改为提前
        while self.date_schedule[-1] > self.end:
            self.date_schedule[-1] = self.trade_calendar.advance(self.date_schedule[-1], datetime.timedelta(days=-1))
        # 使用 dict.fromkeys() 去重，并保持顺序
        self.date_schedule = list(dict.fromkeys(self.date_schedule))
        return self.date_schedule

    def count_business_days(self, base_date):
        """根据某个基准日期，计算self.date_schedule中的每个日期距离基准日期几个交易日
           返回一个列表，列表中是交易日数量int
        Args:
            base_date: datetime.date，基准日期
        Returns:
            self.count_business: List[int]，交易日数量列表
        """
        self.count_business = [self.trade_calendar.business_days_between(base_date, date)
                               for date in self.date_schedule]
        return self.count_business

    def count_calendar_days(self, base_date):
        """根据某个基准日期，计算self.date_schedule中的每个日期距离基准日期几个自然日，
           返回一个列表，列表中是自然日数量int
        Args:
            base_date: datetime.date，基准日期
        Returns:
            self.count_calendar: List[int]，自然日数量列表
        """
        self.count_calendar = [(date - base_date).days for date in self.date_schedule]
        return self.count_calendar


@unique
class AnnualDays(Enum):
    """年化系数枚举类"""
    N365 = 365
    N360 = 360
    N243 = 243
    N244 = 244
