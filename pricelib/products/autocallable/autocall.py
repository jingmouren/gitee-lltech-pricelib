#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from contextlib import suppress
import datetime
import numpy as np
from pricelib.common.utilities.enums import CallPut, EngineType
from pricelib.common.time import AnnualDays, CN_CALENDAR, global_evaluation_date
from pricelib.common.utilities.utility import time_this, logging
from pricelib.pricing_engines.integral_engines import QuadAutoCallEngine
from .autocallable_base import AutocallableBase


class AutoCall(AutocallableBase):
    """Autocall Note(二元小雪球)，不带敲入的自动赎回结构，每月观察敲出，到期未敲出获得红利票息(保本产品)"""

    def __init__(self, s0, *, barrier_out, coupon_out, coupon_div=0, callput=CallPut.Call, lock_term=1,
                 maturity=None, start_date=None, end_date=None, obs_dates=None, pay_dates=None, margin_lvl=1,
                 engine=None, trade_calendar=CN_CALENDAR, annual_days=AnnualDays.N365, t_step_per_year=243,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            s0: float，标的初始价格
            barrier_out: float，敲出障碍价，绝对值/百分比
            coupon_out: float，敲出票息，百分比，年化
            coupon_div: float，红利票息，百分比，年化
            callput: 看涨看跌，CallPut枚举类，默认为Call
            lock_term: int，锁定期，单位为月，锁定期内不触发敲出
            margin_lvl:  float，保证金比例，默认为1，即无杠杆
            engine: 定价引擎，PricingEngine类
                    蒙特卡洛: MCAutoCallEngine
                    PDE: FdmSnowBallEngine
                    积分法: QuadAutoCallEngine
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
            则默认使用 积分法 定价引擎 QuadAutoCallEngine
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(s0=s0, maturity=maturity, start_date=start_date, end_date=end_date, lock_term=lock_term,
                         trade_calendar=trade_calendar, obs_dates=obs_dates, pay_dates=pay_dates, margin_lvl=margin_lvl,
                         t_step_per_year=t_step_per_year, annual_days=annual_days)
        len_obs_dates = len(self.obs_dates.date_schedule)
        self.barrier_out = np.ones(len_obs_dates) * barrier_out
        self.coupon_out = np.ones(len_obs_dates) * coupon_out
        self.coupon_div = coupon_div
        self.barrier_in = np.zeros(len_obs_dates)  # 小雪球无敲入
        self.callput = callput
        self.margin_lvl = margin_lvl  # 预付金比例
        if engine is not None:
            self.set_pricing_engine(engine)
        elif s is not None and r is not None and q is not None and vol is not None:
            default_engine = QuadAutoCallEngine(s=s, r=r, q=q, vol=vol)
            self.set_pricing_engine(default_engine)

    def set_pricing_engine(self, engine):
        """设置定价引擎"""
        self.engine = engine
        logging.info(f"{self}当前定价方法为{engine.engine_type.value}")

    def __repr__(self):
        """返回期权的描述"""
        return f"Auto{self.callput.name} Note(二元小雪球)"

    @time_this
    def price(self, t: datetime.date = None, spot=None):
        """计算期权价格
        Args:
            t: datetime.date，计算期权价格的日期
            spot: float，标的价格
        Returns: 期权现值
        """
        self.validate_parameters(t=t)
        if self.engine.engine_type == EngineType.PdeEngine and self.callput == CallPut.Put:
            logging.warning("PDE引擎暂不支持AutoPut定价")
            return np.nan
        else:
            price = self.engine.calc_present_value(prod=self, t=t, spot=spot)
            return price

    def delta(self, spot=None, step=None, *args, **kwargs):
        """求价格spot处的delta值
        会先尝试调用定价引擎的delta方法，如果没有则使用通用的差分法计算
        Args:
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
        Returns: delta = ∂V/∂S
        """
        try:
            if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
                return self.engine.delta(prod=self, spot=spot, step=step, *args, **kwargs)
        except:
            self.validate_parameters()
            spot = self.engine.process.spot() if spot is None else spot
            step = spot * 0.01 if step is None else step
            up_price = self.price(spot=spot + step)
            down_price = self.price(spot=spot - step)
            delta = (up_price - down_price) / (2 * step)
            return delta

    def gamma(self, spot=None, step=None, *args, **kwargs):
        """求价格spot处的gamma值
        会先尝试调用定价引擎的gamma方法，如果没有则使用通用的差分法计算
        Args:
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
        Returns: gamma = ∂2V/∂S2
        """
        try:
            if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
                return self.engine.gamma(prod=self, spot=spot, step=step, *args, **kwargs)
        except:
            self.validate_parameters()
            spot = self.engine.process.spot() if spot is None else spot
            step = spot * 0.01 if step is None else step
            up_price = self.price(spot=spot + step)
            down_price = self.price(spot=spot - step)
            mid_price = self.price(spot=spot)
            gamma = (up_price - 2 * mid_price + down_price) / (step ** 2)
            return gamma

    def vega(self, spot=None, step=0.01, *args, **kwargs):
        """计算波动率上升1%的vega todo:目前只支持常数波动率
        会先尝试调用定价引擎的vega方法，如果没有则使用通用的差分法计算
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
        Returns: vega = ∂V/∂σ
        """
        try:
            if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
                return self.engine.vega(prod=self, spot=spot, step=step, *args, **kwargs)
        except:
            self.validate_parameters()
            spot = self.engine.process.spot() if spot is None else spot
            last_vol = self.engine.process.vol.volval
            last_price = self.price(spot=spot)
            self.engine.process.vol.volval = last_vol + step
            new_price = self.price(spot=spot)
            self.engine.process.vol.volval = last_vol
            vega = (new_price - last_price)
            return vega

    def theta(self, spot=None, step=1, *args, **kwargs):
        """计算一天的theta值
        会先尝试调用定价引擎的theta方法，如果没有则使用通用的差分法计算
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: int，时间步长，默认为1天
        Returns: theta = ∂V/∂t
        """
        try:
            if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
                return self.engine.theta(prod=self, spot=spot, step=step, *args, **kwargs)
        except:
            try:
                spot = self.engine.process.spot() if spot is None else spot
                step = datetime.timedelta(days=step)
                origin_price = self.price(spot=spot)
                next_date = self.trade_calendar.advance(global_evaluation_date(), step)
                new_price = self.price(t=next_date, spot=spot)
                theta = (new_price - origin_price)
                return theta
            except:  # 可能遇到到期日估值，无法计算theta
                calculate_date = global_evaluation_date()
                if calculate_date > self.trade_calendar.advance(self.end_date, datetime.timedelta(days=1)):
                    raise ValueError(f"估值日必须小于等于到期日, 当前估值日为{calculate_date}, 到期日为{self.end_date}.")
                return np.nan  # 到期日估值，无法计算theta,返回nan

    def rho(self, spot=None, step=0.01, *args, **kwargs):
        """计算无风险利率上升1%的rho todo:目前只支持常数无风险利率
        会先尝试调用定价引擎的rho方法，如果没有则使用通用的差分法计算
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
        Returns: rho = ∂V/∂r
        """
        try:
            if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
                return self.engine.rho(prod=self, spot=spot, step=step, *args, **kwargs)
        except:
            self.validate_parameters()
            spot = self.engine.process.spot() if spot is None else spot
            last_r = self.engine.process.interest.data
            last_price = self.price(spot=spot)
            self.engine.process.interest.data = last_r + step
            new_price = self.price(spot=spot)
            self.engine.process.interest.data = last_r
            rho = (new_price - last_price)
            return rho

    def pv_and_greeks(self, spot=None, *args, **kwargs):
        """当一次性计算pv和5个greeks时，可以调用此函数，减少计算量
        会先尝试调用定价引擎的pv_and_greeks方法，如果没有则使用通用的差分法计算
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
        Returns:
            Dict[str:float]: {'pv': pv, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
        """
        if self.engine.engine_type != EngineType.PdeEngine or self.callput != CallPut.Put:  # PDE引擎暂不支持AutoPut定价
            if hasattr(self.engine, "pv_and_greeks"):
                with suppress(AttributeError, ValueError):
                    self.price()
                    return self.engine.pv_and_greeks(prod=self, spot=spot, *args, **kwargs)
        self.validate_parameters()
        spot = self.engine.process.spot() if spot is None else spot
        pv = np.float64(self.price(spot=spot))
        s_step = spot * 0.01
        up_price = self.price(spot=spot + s_step)
        down_price = self.price(spot=spot - s_step)
        delta = (up_price - down_price) / (2 * s_step)
        gamma = (up_price - 2 * pv + down_price) / (s_step ** 2)

        last_vol = self.engine.process.vol.volval
        self.engine.process.vol.volval = last_vol + 0.01
        new_price = self.price(spot=spot)
        self.engine.process.vol.volval = last_vol
        vega = (new_price - pv)

        last_r = self.engine.process.interest.data
        self.engine.process.interest.data = last_r + 0.01
        new_price = self.price(spot=spot)
        self.engine.process.interest.data = last_r
        rho = (new_price - pv)

        try:  # 可能遇到到期日估值，无法计算theta
            t_step = datetime.timedelta(days=1)
            next_date = self.trade_calendar.advance(global_evaluation_date(), t_step)
            new_price = self.price(t=next_date, spot=spot)
            theta = (new_price - pv)
        except:
            theta = np.nan
        return {'pv': pv, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}
