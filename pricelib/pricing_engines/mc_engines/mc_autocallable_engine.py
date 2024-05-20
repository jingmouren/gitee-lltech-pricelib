#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import RandsMethod, LdMethod, StatusType
from pricelib.common.pricing_engine_base import McEngine
from pricelib.common.time import global_evaluation_date


class MCAutoCallableEngine(McEngine):
    """自动赎回结构(雪球类) Monte Carlo 模拟定价引擎
    支持变敲出、变敲入、变票息等要素可变型雪球结构"""

    def __init__(self, stoch_process=None, n_path=100000, rands_method=RandsMethod.LowDiscrepancy,
                 antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0, *,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: 随机过程StochProcessBase对象
            n_path: int，MC模拟路径数
            rands_method: 生成随机数方法，RandsMethod枚举类，Pseudorandom伪随机数/LowDiscrepancy低差异序列
            antithetic_variate: bool，是否使用对立变量法
            ld_method: 若使用了低差异序列，指定低差异序列方法，LdMethod枚举类，Sobol序列/Halton序列
            seed: int，随机数种子
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process, n_path, rands_method=rands_method, antithetic_variate=antithetic_variate,
                         ld_method=ld_method, seed=seed, s=s, r=r, q=q, vol=vol)
        # 以下为计算过程的中间变量
        self.prod = None  # Product产品对象
        self.obs_dates = None  # 根据估值日，将敲出观察日转化为List[int]，交易日期限
        self.pay_dates = None  # 根据估值日，将支付日转化为List[float]，年化自然日期限
        self._maturity = None  # 年化到期时间
        self.knock_out_date = None  # 每条路径的敲出时间
        self.not_knock_out = None  # 每条路径是否未敲出
        self.knock_in_scenario = None  # 每天路径是否敲入
        self.knock_out_profit = None  # 敲出payoff加总
        self.hold_to_maturity_profit = None  # 持有到期payoff加总
        self.knock_in_loss = None  # 敲入payoff加总
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        self._barrier_out = None
        self._barrier_in = None
        self._coupon_out = None

    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        self.prod = prod
        calculate_date = global_evaluation_date() if t is None else t
        self._maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        self.obs_dates = np.array([num for num in obs_dates if num >= 0])
        pay_dates = prod.pay_dates.count_calendar_days(calculate_date)
        self.pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= 0])
        assert len(self.obs_dates) == len(self.pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        # 经过估值日截断的列表，例如prod.barrier_out有22个，存续一年时估值，_barrier_out只有12个
        self._barrier_out = prod.barrier_out[-len(obs_dates):].copy()
        self._barrier_in = prod.barrier_in[-len(obs_dates):].copy()
        self._coupon_out = prod.coupon_out[-len(obs_dates):].copy()
        if spot is None:
            spot = self.process.spot()
        else:
            self.reset_paths_flag()  # 重置路径标志位，重新生成路径

        self.s_paths = self.path_generator(n_step=_maturity_business_days, spot=spot,
                                           t_step_per_year=prod.t_step_per_year).copy()
        # 统计各个情景的payoff
        self._cal_knock_out_date()
        self._cal_knock_out_payoff()
        self._cal_knock_in_scenario(calculate_date)
        self._cal_hold_to_maturity_payoff()
        self._cal_knock_in_payoff()
        result = (self.knock_out_profit + self.hold_to_maturity_profit + self.knock_in_loss) / self.n_path
        return result

    def _cal_knock_out_date(self):
        """计算每条路径敲出时间"""
        # 敲出部分
        knock_out_scenario = np.tile(self.obs_dates, (self.n_path, 1)).T
        barrier_out = np.array(self._barrier_out)
        barrier_out = np.tile(barrier_out, (self.n_path, 1)).T
        # 记录每条路径的具体敲出日，如果无敲出则保留inf      注意：路径矩阵包含了期初S0的行
        knock_out_scenario = np.where(self.s_paths[self.obs_dates] >= barrier_out, knock_out_scenario.astype(int),
                                      np.inf)
        self.knock_out_date = np.min(knock_out_scenario, axis=0)
        self.not_knock_out = self.knock_out_date == np.inf

    def _cal_knock_out_payoff(self):
        """计算敲出部分的payoff"""
        prod = self.prod
        is_knock_out = self.knock_out_date != np.inf
        # 每个交易日敲出票息
        coupon_call_array = np.array(self._coupon_out).repeat(
            np.diff(np.append(np.zeros((1,)), self.obs_dates)).astype(int))
        # 每条路径敲出交易日序列
        knock_out_time = self.knock_out_date[is_knock_out].astype(int)
        coupon_call = coupon_call_array[knock_out_time - 1]
        # 每个交易日对应的自然日的下一个敲出观察自然日
        next_calendar_day = np.array(self.pay_dates).repeat(
            np.diff(np.append(np.zeros((1,)), self.obs_dates)).astype(int))
        # 所有敲出路径的敲出自然日，年化，贴现用
        knock_out_time_annual = next_calendar_day[knock_out_time - 1]
        # 计算敲出价格，用于上涨参与payoff
        if isinstance(prod.strike_call, (int, float)):
            strike_call = prod.strike_call
        elif isinstance(prod.strike_call, (list, np.ndarray)) and len(prod.strike_call) == len(self.obs_dates):
            strike_call_array = (np.array(prod.strike_call)).repeat(
                np.diff(np.append(np.zeros((1,)), self.obs_dates)).astype(int))
            strike_call = strike_call_array[knock_out_time - 1]
        else:
            raise ValueError("敲出看涨执行价设置错误")
        # 把payoff先折现再求和,计算敲出总所入
        value = np.sum((prod.s0 * (coupon_call * (knock_out_time_annual if not prod.trigger else 1) + prod.margin_lvl)
                        + prod.parti_out * (self.s_paths[knock_out_time, is_knock_out] - strike_call))
                       * self.process.interest.disc_factor(knock_out_time_annual))
        self.knock_out_profit = value

    def _cal_knock_in_scenario(self, *args, **kwargs):
        """统计敲入的路径"""
        prod = self.prod
        if prod.status == StatusType.DownTouch:
            pass
        else:
            # 判断某一条路径是否有敲入
            if isinstance(prod.barrier_in, (int, float)):
                knock_in_level = prod.barrier_in
            elif isinstance(prod.barrier_in, (list, np.ndarray)):
                knock_in_level = np.array(self._barrier_in).repeat(
                    np.diff(np.append(np.zeros((1,)), self.obs_dates)).astype(int))
            else:
                raise ValueError("敲入线设置错误")
            knock_in_level = np.tile(knock_in_level, (self.n_path, 1)).T
            self.knock_in_scenario = np.any(self.s_paths[1:] <= knock_in_level, axis=0)

    def _cal_hold_to_maturity_payoff(self):
        """计算持有到期的payoff"""
        # 到期红利部分
        prod = self.prod
        if prod.status == StatusType.DownTouch:  # 已敲入
            self.hold_to_maturity_profit = 0
        else:
            # 持有到期，没有敲入也没有敲出
            hold_to_maturity = (~self.knock_in_scenario) & self.not_knock_out
            # 平稳持有到期路径条数
            hold_to_maturity_count = np.count_nonzero(hold_to_maturity)
            # 未敲出未敲入到期收入
            value = (hold_to_maturity_count * (prod.coupon_div * (self.pay_dates[-1] if not prod.trigger else 1)
                                               + prod.margin_lvl) * prod.s0 * self.process.interest.disc_factor(
                self._maturity))
            self.hold_to_maturity_profit = value

    def _cal_knock_in_payoff(self):
        """计算敲入部分的payoff"""
        prod = self.prod
        if prod.status == StatusType.DownTouch:  # 已敲入
            value = np.sum(np.maximum(np.minimum(self.s_paths[-1, self.not_knock_out] - prod.strike_upper, 0),
                                      prod.strike_lower - prod.strike_upper) * prod.parti_in
                           + prod.margin_lvl * prod.s0) * self.process.interest.disc_factor(self._maturity)
        else:
            value = np.sum(np.maximum(np.minimum(self.s_paths[-1, self.not_knock_out & self.knock_in_scenario]
                                                 - prod.strike_upper, 0),
                                      prod.strike_lower - prod.strike_upper) * prod.parti_in
                           + prod.margin_lvl * prod.s0) * self.process.interest.disc_factor(self._maturity)
        self.knock_in_loss = value
