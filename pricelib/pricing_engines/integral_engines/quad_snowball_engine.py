#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import StatusType, QuadMethod
from pricelib.common.pricing_engine_base import QuadEngine
from pricelib.common.time import global_evaluation_date


class QuadSnowballEngine(QuadEngine):
    """雪球数值积分法定价引擎，仅支持非到期观察敲入(到期观察敲入如FCN、DCN等另有更快的算法)"""

    def __init__(self, stoch_process=None, quad_method=QuadMethod.Simpson, n_points=801, n_max=4, *,
                 s=None, r=None, q=None, vol=None):
        """初始化数值积分法定价引擎
        Args:
            stoch_process: StochProcessBase，随机过程
            quad_method: 数值积分方法，QuadMethod枚举类，Trapezoid梯形法则/Simpson辛普森法则
            n_points: int，积分点个数 (需要注意，辛普森法则的n_points必须是奇数)
            n_max: int，价格网格上界，设为n_smax倍初始价格
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process, quad_method, n_points, n_max, s=s, r=r, q=q, vol=vol)
        # 以下为计算过程的中间变量
        self.prod = None  # Product产品对象
        self.obs_dates = None  # 根据估值日，将敲出观察日转化为List[int]，交易日期限
        self.pay_dates = None  # 根据估值日，将支付日转化为List[float]，年化自然日期限
        self.diff_obs_pay_dates = None  # 观察日与付息日的差值，List[float]，年化自然日期限
        self.smax = None  # 价格网格上界，float
        self.smin = None  # 价格网格下界，float
        self.sy_vec = None  # 价格格点(完整)，np.ndarray
        self.s_vec = None  # 价格格点(不含smax和smin)，np.ndarray
        self.dt = None  # 时间步长，float
        self.j_vec = None  # 时间格点向量，np.ndarray
        self.v_not_in = None  # 未敲入时的期权价值矩阵，np.ndarray
        self.v_knock_in = None  # 敲入时的期权价值矩阵，np.ndarray
        self.next_paydate = None
        self.out_dates = None  # 逆序敲出观察日，List[int]，交易日期限
        self.next_diff_obspaydate = None  # 每个交易日对应的下一个敲出观察日对应的付息日与观察日的自然日年化之差
        self.next_barrier_out = None
        self.next_barrier_in = None
        self.out_idxs = None
        self.in_idx = None
        self.itm = None
        self.adj_upper_idx = None
        self.adj_lower_idx = None
        self.next_coupon_out = None
        self.reversed_barrier_out = None
        self.reversed_barrier_in = None
        self.reversed_coupon_out = None
        self.coupon_outs = None
        self._barrier_in = None  # 是经过估值日截断之后的列表，例如prod.barrier_in有22个，存续一年时估值，self._barrier_in只有12个
        self._barrier_out = None  # 是经过估值日截断之后的列表，例如prod.barrier_out有22个，存续一年时估值，self._barrier_out只有12个
        self._coupon_out = None  # 是经过估值日截断之后的列表，例如prod.coupon_out有22个，存续一年时估值，self._coupon_out只有12个

    # pylint: disable=too-many-locals, too-many-statements
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
        _maturity = (prod.end_date - calculate_date).days / prod.annual_days.value
        self.backward_steps = prod.trade_calendar.business_days_between(calculate_date, prod.end_date)
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        obs_dates = np.array([num for num in obs_dates if num >= 0])
        pay_dates = prod.pay_dates.count_calendar_days(calculate_date)
        self.pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= 0])
        assert len(obs_dates) == len(self.pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        self.diff_obs_pay_dates = np.array([(prod.pay_dates.date_schedule[i] - prod.obs_dates.date_schedule[i]).days
                                            for i in range(len(obs_dates))]) / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()
        r = self.process.interest(_maturity)
        q = self.process.div(_maturity)
        vol = self.process.vol(_maturity, spot)
        self._check_method_params()
        self.set_quad_params(r=r, q=q, vol=vol)
        self.smax = self.n_max * spot
        self.smin = 0.0001
        self.sy_vec = np.linspace(self.smin, self.smax, self.n_points)
        self.s_vec = self.sy_vec[1:-1].copy()
        self.dt = 1 / prod.t_step_per_year
        self.j_vec = np.linspace(0, self.backward_steps, self.backward_steps + 1)  # 生成时间格点
        self.v_not_in = np.zeros(shape=(self.n_points, self.backward_steps + 1))
        self.v_knock_in = np.zeros(shape=(self.n_points, self.backward_steps + 1))

        # 初始化逆序敲出观察日
        if isinstance(obs_dates, (list, np.ndarray)):  # 敲出观察日是具体第n天数的列表，如[11,22,32,43,53,64]
            self.out_dates = np.round(np.flip(np.array(obs_dates))).astype(int)
        elif isinstance(obs_dates, (int, np.int32)):  # 敲出观察日是整数， 敲出观察日个数，如6代表6个均匀分布的观察日
            out_dates = (self.backward_steps - np.round(np.linspace(0, self.backward_steps,
                                                                    round(obs_dates + 1)))[:-1])
            self.out_dates = out_dates.astype(int)
        else:
            raise ValueError('敲出观察日类型设置错误，请检查')
        self.next_diff_obspaydate = self.diff_obs_pay_dates.repeat(
            np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))  # 计息时间数
        self.next_diff_obspaydate = np.append(self.next_diff_obspaydate[0],
                                              self.next_diff_obspaydate)  # 每个交易日对应的下一个敲出观察日对应的付息日与观察日的自然日年化之差

        # 设置期末价值状态
        self._init_terminal_condition(self.s_vec, _maturity)
        # 最近一次敲出观察日的已敲入矩阵价格
        days_to_last_knock_out_day = 0  # 距离最近一次敲出观察的天数
        knock_out_payoff = [self.next_coupon_out * _maturity]
        # 初始化已敲入、未敲入的边界条件
        self._init_boundary_condition(self.smax, _maturity)
        # 最近一次敲出观察日的已敲入矩阵价格
        last_v_knock_in = self.v_knock_in[1: self.adj_upper_idx, -1]
        for j in range(self.backward_steps - 1, 0, -1):
            days_to_last_knock_out_day += 1
            x_not_in = self.s_vec[self.in_idx:]
            x_in = self.sy_vec[1:self.adj_upper_idx]
            v_not_in = self.v_not_in[:, j + 1]
            v_in = self.v_knock_in[:, j + 1]

            if prod.status == StatusType.NoTouch:  # 未敲入
                self.v_not_in[(1 + self.in_idx):-1, j] = self.step_backward(x_not_in, self.sy_vec, v_not_in, self.dt)
            # 如果敲出payoff全部相同，即没有看涨call，只有敲出票息，则可以用Vmb简化计算
            if np.any(knock_out_payoff != knock_out_payoff[0]):
                self.v_knock_in[(1 + self.adj_lower_idx):, j] = self.step_backward(
                    self.sy_vec[(1 + self.adj_lower_idx):],
                    self.sy_vec, v_in,
                    self.dt)
            else:  # 如果敲出payoff全部相同，即没有看涨call，只有敲出票息，则可以用Vmb简化计算
                # 只需计算敲出线下方的点
                x_in_adjusted = self.sy_vec[(1 + self.adj_lower_idx):self.adj_upper_idx]
                self.v_knock_in[(1 + self.adj_lower_idx):self.adj_upper_idx, j] = (
                        self.step_backward(x_in_adjusted, x_in, last_v_knock_in,
                                           self.dt * days_to_last_knock_out_day) + self.Vmb(x_in_adjusted,
                                           self.smax * self.adj_upper_idx / self.n_points, 1,
                                           self.dt * days_to_last_knock_out_day) *
                                           knock_out_payoff[0])

            if j in self.out_dates:
                coupon_t = 1 if prod.trigger else self.next_paydate[j]
                # 如果敲出价是浮动的，调整敲出价
                self.next_barrier_out = (self.reversed_barrier_out[np.where(self.out_dates == j)[0][0]]
                                         if isinstance(prod.barrier_out, (list, np.ndarray)) else prod.barrier_out)
                self.out_idxs = 1 + np.array(np.where(self.s_vec >= self.next_barrier_out)[0])
                self.adj_upper_idx = int(self.n_points / self.n_max * (self.next_barrier_out / prod.s0 + 0.25))
                # 如果敲出票息是浮动的，调整敲出票息
                self.next_coupon_out = (self.reversed_coupon_out[np.where(self.out_dates == j)[0][0]]
                                        if isinstance(prod.coupon_out, (list, np.ndarray)) else prod.coupon_out)
                # 修改敲出边界条件
                knock_out_payoff = self.itm[self.out_idxs - 1] + (
                    prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0 * self.process.interest.disc_factor(
                    self.next_paydate[j], self.next_paydate[j] - self.next_diff_obspaydate[j])
                self.v_not_in[self.out_idxs, j] = self.v_knock_in[self.out_idxs, j] = knock_out_payoff

                # 如果敲入价是浮动的，调整敲入价
                self.next_barrier_in = self.reversed_barrier_in[np.where(self.out_dates == j)[0][0]] if isinstance(
                    prod.barrier_in, (list, np.ndarray)) else prod.barrier_in
                self.in_idx = 1 + np.where(self.s_vec <= self.next_barrier_in)[0][
                    -1] if self.next_barrier_in > 0 else 0

                last_v_knock_in = self.v_knock_in[1: self.adj_upper_idx, j]  # 最近一次敲出观察日的已敲入矩阵价格
                days_to_last_knock_out_day = 0  # 距离最近一次敲出观察的天数

            if prod.status == StatusType.NoTouch:  # 未敲入
                # 考虑敲入的情况
                self.v_not_in[(1 + self.adj_lower_idx):(self.in_idx + 1), j] = self.v_knock_in[
                                                                               (1 + self.adj_lower_idx):(
                                                                                       self.in_idx + 1), j]

        x = np.array(spot)
        if prod.status == StatusType.NoTouch:
            value = self.step_backward(x, self.sy_vec, self.v_not_in[:, 1], self.dt)[0]
        else:
            value = self.step_backward(x, self.sy_vec, self.v_knock_in[:, 1], self.dt)[0]
        return value

    def _init_terminal_condition(self, s_vec, maturity):
        """初始化终止时已敲入且未敲出、未敲入且未敲出的期权价值
        Args:
            s_vec: List[float], 网格的价格格点
            maturity: float, 到期时间，年化自然日期限
        Returns: None
        """
        prod = self.prod
        # 初始化next_barrier_in、next_barrier_out、next_coupon_out为最后一期的敲入价、敲出价、敲出票息。若变敲出变敲入，需要动态调整
        # 设置到期日的敲入边界位置
        if isinstance(prod.barrier_in, (int, float, np.int32, np.float64)):
            # 常数敲入雪球
            self.next_barrier_in = prod.barrier_in
        elif isinstance(prod.barrier_in, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._barrier_in = prod.barrier_in[-len(self.out_dates):].copy()
            # 浮动敲入雪球
            self.next_barrier_in = self._barrier_in[-1]
            self.reversed_barrier_in = np.flip(np.array(self._barrier_in))
        else:
            raise ValueError(f'敲入障碍价类型为{type(prod.barrier_in)}，仅支持int/float/list/np.ndarray，请检查')

        # 设置到期日的敲出边界位置
        if isinstance(prod.barrier_out, (int, float, np.int32, np.float64)):
            # 常数敲出雪球
            self.next_barrier_out = prod.barrier_out
        elif isinstance(prod.barrier_out, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._barrier_out = prod.barrier_out[-len(self.out_dates):].copy()
            # 阶梯式雪球
            self.next_barrier_out = self._barrier_out[-1]
            self.reversed_barrier_out = np.flip(np.array(self._barrier_out))
        else:
            raise ValueError(f'敲出障碍价类型为{type(prod.barrier_out)}，仅支持int/float/list/np.ndarray，请检查')

        # 敲出票息
        if isinstance(prod.coupon_out, (int, float, np.int32, np.float64)):
            # 常数敲出票息
            self.next_coupon_out = prod.coupon_out
            # N天的敲出票息列表，用于生成mspot价格上边界条件
            self.coupon_outs = np.full(self.backward_steps + 1, prod.coupon_out)
        elif isinstance(prod.coupon_out, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._coupon_out = prod.coupon_out[-len(self.out_dates):].copy()
            # 浮动敲出票息
            self.next_coupon_out = self._coupon_out[-1]
            self.reversed_coupon_out = np.flip(np.array(self._coupon_out))
            # N天的敲出票息列表，用于生成smax价格上边界条件
            coupon_outs = np.array(self._coupon_out).repeat(
                np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))
            self.coupon_outs = np.append(coupon_outs[0], coupon_outs)
        else:
            raise ValueError(f'敲出票息类型为{type(prod.coupon_out)}，仅支持int/float/list/np.ndarray，请检查')

        # 为了避免积分法在s=0的边缘多次迭代后异常大，直接设定边缘附近的值
        self.adj_lower_idx = int(self.n_points / self.n_max * 0.4)
        self.adj_upper_idx = int(self.n_points / self.n_max * (self.next_barrier_out / prod.s0 + 0.25))
        # in_idx：小于到期日敲入线的最大的标的价格索引。如果是变敲入雪球，此变量in_idx应该随敲入价动态调整。
        self.in_idx = 1 + np.where(s_vec <= self.next_barrier_in)[0][-1] if self.next_barrier_in > 0 else 0
        # out_idxs：大于到期日敲出线的标的价格数组的索引。如果是变敲出雪球，此变量out_idxs应该随敲出价动态调整。
        self.out_idxs = 1 + np.array(np.where(s_vec >= self.next_barrier_out)[0])
        # 虚值call在值度
        self.itm = np.where(s_vec - prod.strike_call > 0, (s_vec - prod.strike_call) * prod.parti_out, 0)

        coupon_t = 1 if prod.trigger else maturity  # 票息是否年化
        # 矩阵最后一列：到期payoff。考虑到了保底、保证金\纯期权模式、以及敲出边界上方增加虚值call
        """已敲入：如果到期小于敲出边界，在100%本金模式下，小于保底边界，获得保底价；
                                                  大于保底边界，获得到期价格；
                                   在保证金模式下，小于保底边界，支付期初价（高行权价）与保底价之间价差；
                                                  大于保底边界，支付到期价格与期初价之间价差；
                  如果到期大于敲出边界，大于看涨行权价，获得看涨收益、本金和票息；
                                    在敲出边界和看涨行权价之间，获得本金和票息"""
        self.v_knock_in[1:-1, -1] = np.where(s_vec < self.next_barrier_out,
                                             np.minimum(
                                                 np.where(s_vec <= prod.strike_lower, prod.strike_lower,
                                                          s_vec) -
                                                 prod.strike_upper, 0) + prod.s0 * prod.margin_lvl,
                                             np.where(s_vec > prod.strike_call,
                                                      (s_vec - prod.strike_call) * prod.parti_out + (
                                                          prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                      (
                                                          prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0))
        """未敲入：如果到期小于敲入边界，在100%本金模式下，小于保底边界，获得保底价；
                                                  大于保底边界，获得实际价格；
                                  在保证金模式下，小于保底边界，支付保底价与期初价之间价差；
                                                  大于保底边界，支付到期价格与期初价之间价差；
                 如果到期大于敲入边界，且大于看涨行权价，获得看涨收益、本金和票息；
                                  在敲入边界和看涨行权价之间，获得本金和票息
                                          如果在敲出边界和看涨行权价之间，获得本金和敲出票息
                                          如果在敲入边界和敲出边界之间，获得本金和红利票息"""
        self.v_not_in[1:-1, -1] = np.where(s_vec <= self.next_barrier_in,
                                           np.where(s_vec <= prod.strike_lower, prod.strike_lower, s_vec) -
                                           prod.strike_upper + prod.s0 * prod.margin_lvl,
                                           np.where(s_vec > prod.strike_call,
                                                    (s_vec - prod.strike_call) * prod.parti_out + (
                                                        prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                    np.where(s_vec >= self.next_barrier_out,
                                                             (
                                                         prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                             (
                                                         prod.margin_lvl + prod.coupon_div * coupon_t) * prod.s0)))

    def _init_boundary_condition(self, smax, maturity):
        """初始化边界条件
        考虑到了保底、OTM、保证金/纯期权模式、以及敲出边界上方增加虚值call(增强型)
        Args:
            smax: float, 标的价格上界
            maturity: float, 到期时间，年化自然日期限
        Returns: None
        """
        prod = self.prod
        # 矩阵第一行: 标的价格为0时结构的价值
        t_vec = self.dt * self.j_vec
        interest_maturity_discount_factor = self.process.interest.disc_factor(maturity, t_vec)
        div_maturity_discount_factor = self.process.div.disc_factor(maturity, t_vec)

        # 由看涨看跌平价 call-put=S*exp(-qt)-K*exp(-rt)，S=0时call=0，put=K*exp(-rt)
        for i in range(self.adj_lower_idx + 1):
            self.v_not_in[i, :] = self.v_knock_in[i, :] = ((prod.margin_lvl * prod.s0 - prod.strike_upper
                                                            + prod.strike_lower) * interest_maturity_discount_factor)
            if self.sy_vec[i] > prod.strike_lower:
                self.v_knock_in[i, :] += self.sy_vec[i] * div_maturity_discount_factor
                self.v_not_in[i, :] = self.v_knock_in[i, :]

        # 矩阵最后一行: 每一个观察日前，股价上界能得到的本金加票息贴现
        self.next_paydate = self.pay_dates.repeat(
            np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))  # 计息时间数
        self.next_paydate = np.append(self.next_paydate[0], self.next_paydate)  # 每个交易日对应的下一个敲出观察日对应的付息日

        # 触发器：上边界为绝对票息贴现/ 非触发器：上边界为年化票息贴现
        coupon_t = 1 if prod.trigger else self.next_paydate
        # 由看涨看跌平价 call-put = S*exp(-qt) - K*exp(-rt)，S=Smax时put=0，call = Smax*exp(-qt) - K*exp(-rt)
        interest_next_paydate_discount_factor = self.process.interest.disc_factor(self.next_paydate, t_vec)
        div_next_paydate_discount_factor = self.process.div.disc_factor(self.next_paydate, t_vec)
        v_upper = (prod.parti_out * (self.smax * div_next_paydate_discount_factor
                                     - prod.strike_call * interest_next_paydate_discount_factor)
                   + (prod.margin_lvl + self.coupon_outs * coupon_t) * prod.s0
                   * interest_next_paydate_discount_factor)
        self.v_knock_in[-1, :] = self.v_not_in[-1, :] = v_upper
