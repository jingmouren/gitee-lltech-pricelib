#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType, ExerciseType
from pricelib.common.time import CN_CALENDAR, AnnualDays
from .autocallable_base import PhoenixBase


class Phoenix(PhoenixBase):
    """Phoenix 凤凰票据产品类
    有派息价格，每个观察日如果价格高于派息价格，派固定利息；每日观察是否敲入"""

    def __init__(self, s0, barrier_out, barrier_yield, barrier_in, coupon, obs_dates=None, pay_dates=None, lock_term=1,
                 parti_in=1, margin_lvl=1, status=StatusType.NoTouch, engine=None, maturity=None, start_date=None,
                 end_date=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            s0: float，标的初始价格
            barrier_out: float，敲出障碍价，绝对值/百分比，长度需要与敲出观察日数一致
            barrier_yield: float，派息边界价格，绝对值/百分比，长度需要与派息观察日数一致
            barrier_in: float，敲入障碍价，绝对值/百分比
            coupon: float，派息票息，百分比，非年化
            lock_term: int，锁定期，单位为月，锁定期内不触发敲出
            parti_in: float，敲入后参与率，限损
            margin_lvl: float，保证金比例，默认为1，即无杠杆
            status: 敲入敲出状态，StatusType枚举类，默认为NoTouch未敲入未敲出，UpTouch为已敲出，DownTouch为已敲入
            engine: 定价引擎，PricingEngine类
                    蒙特卡洛: MCPhoenixEngine
                    PDE: FdmPhoenixEngine
        时间参数: 要么输入年化期限，要么输入起始日和到期日；敲出观察日和票息支付日可缺省
            maturity: float，年化期限
            start_date: datetime.date，起始日
            end_date: datetime.date，到期日
            obs_dates: List[datetime.date]，敲出观察日，可缺省，缺省时会自动生成每月观察的敲出日期序列(已根据节假日调整)
            pay_dates: List[datetime.date]，票息支付日，可缺省，长度需要与敲出观察日数一致，如不指定则默认为敲出观察日
            trade_calendar: 交易日历，Calendar类，默认为中国内地交易日历
            annual_days: int，每年的自然日数量
            t_step_per_year: int，每年的交易日数量
        可选参数:
            若未提供引擎的情况下，提供了标的价格、无风险利率、分红/融券率、波动率，
            则默认使用 PDE 定价引擎 FdmPhoenixEngine
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, status=status,
                         annual_days=annual_days, parti_in=parti_in, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year, engine=engine, s=s, r=r, q=q, vol=vol)
        len_pay_dates = len(self.pay_dates.date_schedule)
        # obs_dates无锁定期，因为迭代过程中paydates和barrier in的改变都发生在敲出日obs_dates，锁定期靠敲出价+inf实现
        # 锁定期敲出价格 为+inf
        self.barrier_out = np.ones(len_pay_dates) * barrier_out
        self.barrier_out[:(lock_term - 1)] = np.inf
        # 凤凰全程观察敲入
        self.barrier_in = np.ones(len_pay_dates) * barrier_in
        # 凤凰，默认派息价等于敲入价
        self.barrier_yield = np.ones(len_pay_dates) * barrier_yield if barrier_yield is not None else self.barrier_in
        self.coupon = np.ones(len_pay_dates) * coupon
        self.strike_upper = s0
        self.strike_lower = 0
        self.in_obs_type = ExerciseType.American

    def __repr__(self):
        """返回期权的描述"""
        return "Phoenix 凤凰票据"
