#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from pricelib.common.utilities.enums import StatusType, ExerciseType
from pricelib.common.pricing_engine_base import FdmEngine, FdmGrid
from pricelib.common.time import global_evaluation_date


class FdmAutoCallableEngine(FdmEngine):
    """AutoCallable PDE有限差分法定价引擎基类
        雪球/凤凰/FCN/DCN, 支持变敲出、变敲入、变票息等要素可变型结构"""

    def __init__(self, stoch_process=None, s_step=800, n_smax=2, fdm_theta=1, *,
                 s=None, r=None, q=None, vol=None):
        """初始化有限差分法定价引擎
        Args:
            stoch_process: StochProcessBase随机过程
            s_step: int，价格步数
            n_smax: int，价格网格上界，设为n_smax倍初始价格
            fdm_theta: float，时间方向有限差分theta，0：explicit, 1: implicit, 0.5: Crank-Nicolson
        在未设置stoch_process时，(stoch_process=None)，会默认创建BSMprocess，需要输入以下变量进行初始化
            s: float，标的价格
            r: float，无风险利率
            q: float，分红/融券率
            vol: float，波动率
        """
        super().__init__(stoch_process, s_step=s_step, n_smax=n_smax, fdm_theta=fdm_theta, s=s, r=r, q=q, vol=vol)
        # 计算过程的中间变量
        self.prod = None  # Product产品对象
        self.out_dates = None  # 根据估值日，将敲出观察日转化为List[int]，交易日期限
        self.pay_dates = None  # 根据估值日，将支付日转化为List[float]，年化自然日期限
        self.diff_obs_pay_dates = None  # 观察日与付息日的差值，List[float]，年化自然日期限
        self.smax = None  # 价格网格上界，float
        self.t_step = None  # 网格的时间步数，int
        self.dt = None  # 时间步长，float
        self.j_vec = None  # 时间格点，List[int]
        self.fd_not_in = None  # 未敲入的有限差分网格FdmGrid对象
        self.fd_knockin = None  # 已敲入的有限差分网格FdmGrid对象
        self._barrier_in = None  # 是经过估值日截断之后的列表，例如prod.barrier_in有22个，存续一年时估值，self._barrier_in只有12个
        self._barrier_out = None  # 是经过估值日截断之后的列表，例如prod.barrier_out有22个，存续一年时估值，self._barrier_out只有12个
        self.next_paydate = None
        self.next_barrier_in = None
        self.reversed_barrier_in = None
        self.next_barrier_out = None
        self.reversed_barrier_out = None
        self.in_idx = None
        self.out_idxs = None
        # 雪球独有
        self.next_coupon_out = None
        self.reversed_coupon_out = None
        self._coupon_out = None  # 是经过估值日截断之后的列表，例如prod.coupon_out有22个，存续一年时估值，self._coupon_out只有12个
        self.coupon_outs = None
        self.itm = None
        self.next_diff_obspaydate = None
        # 凤凰独有
        self._barrier_yield = None  # 是经过估值日截断之后的列表，例如prod.barrier_yield有24个，存续一年时估值，self._barrier_yield只有12个
        self.next_barrier_yield = None
        self.reversed_barrier_yield = None
        self.next_coupon = None
        self._coupon = None  # 是经过估值日截断之后的列表，例如prod.coupon有24个，存续一年时估值，self._coupon只有12个
        self.coupons = None
        self.reversed_coupon = None
        self.yld_idxs = None

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
        _maturity_business_days = prod.trade_calendar.business_days_between(calculate_date,
                                                                            prod.end_date) / prod.t_step_per_year
        obs_dates = prod.obs_dates.count_business_days(calculate_date)
        obs_dates = np.array([num for num in obs_dates if num >= 0])
        calculate_start_diff = (calculate_date - prod.start_date).days
        pay_dates = prod.pay_dates.count_calendar_days(prod.start_date)
        self.pay_dates = np.array([num / prod.annual_days.value for num in pay_dates if num >= calculate_start_diff])
        assert len(obs_dates) == len(self.pay_dates), f"Error: {prod}的观察日和付息日长度不一致"
        self.diff_obs_pay_dates = np.array([(prod.pay_dates.date_schedule[i] - prod.obs_dates.date_schedule[i]).days
                                            for i in range(len(obs_dates))]) / prod.annual_days.value
        if spot is None:
            spot = self.process.spot()

        # 返回PDE系数组件abc的函数
        fn_pde_coef = self.process.get_fn_pde_coef(_maturity, spot)

        # 统一已敲入、未敲入矩阵的格点
        self.t_step = round(prod.t_step_per_year * _maturity_business_days)  # 时间步数
        self.dt = 0 if self.t_step == 0 else _maturity_business_days / self.t_step  # 时间步长
        self.j_vec = np.linspace(0, self.t_step, self.t_step + 1)  # 生成时间格点

        if self.t_step == 0:  # 如果估值日是到期日
            self.smax = smin = spot  # 价格网格上下界都取估值日标的价格 - 直接计算到期日payoff
        else:
            smin = 0
            self.smax = self.n_smax * prod.s0  # 价格网格上界, 默认为n_smax倍初始价格

        # 初始化已敲入、未敲入的FdmGrid对象
        self.fd_not_in = FdmGrid(smax=self.smax, maturity=_maturity_business_days,
                                 t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                 fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta, smin=smin)  # 未敲入
        self.fd_knockin = FdmGrid(smax=self.smax, maturity=_maturity_business_days,
                                  t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                  fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta, smin=smin)  # 已敲入

        # 初始化逆序敲出观察日
        if isinstance(obs_dates, (list, np.ndarray)):  # 敲出观察日是具体第n天数的列表，如[11,22,32,43,53,64]
            out_dates = np.round(np.flip(np.array(obs_dates))).astype(int)
        elif isinstance(obs_dates, (int, np.int32)):  # 敲出观察日是整数， 敲出观察日个数，如6代表6个均匀分布的观察日
            out_dates = (self.t_step - np.round(np.linspace(0, self.t_step, round(obs_dates + 1)))[:-1])
            out_dates = out_dates.astype('int')
        else:
            raise ValueError('敲出观察日类型设置错误，请检查')
        self.out_dates = out_dates

        # 初始化到期日payoff价值
        self._init_terminal_condition(self.fd_not_in.s_vec, _maturity)

        if self.t_step == 0:  # 如果估值日是到期日
            # 直接返回到期日payoff
            if prod.status == StatusType.DownTouch:
                result = self.fd_knockin.v_grid[1, 0]
            else:
                result = self.fd_not_in.v_grid[1, 0]
            return np.float64(result)

        # 初始化已敲入、未敲入的边界条件
        self._init_boundary_condition(self.smax, _maturity)

        # N为时间步数，从倒数第二天向第一天反向递推
        self._backward_induction()

        if prod.status == StatusType.DownTouch:
            result = self.fd_knockin.functionize(self.fd_knockin.v_grid[1:-1, 0], kind="cubic")(spot)
        else:
            result = self.fd_not_in.functionize(self.fd_not_in.v_grid[1:-1, 0], kind="cubic")(spot)
        return np.float64(result)

    @abstractmethod
    def _init_terminal_condition(self, *args, **kwargs):
        """初始化终止条件"""

    @abstractmethod
    def _init_boundary_condition(self, *args, **kwargs):
        """初始化边界条件"""

    @abstractmethod
    def _backward_induction(self, *args, **kwargs):
        """反向递推"""

    def delta(self, prod, t=0, spot=None, step=None, status: StatusType = None):
        """求t时刻，价格spot的delta值,
            spot是价格的绝对值
            t是期权价值矩阵的列(时间)的索引
            step为微扰步长，默认为价格spot的1.0%，
            status为敲入状态"""
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        status = prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        delta = (f(min(spot + step, self.smax)) - f(min(spot - step, self.smax))) / (2 * step)
        return delta

    def gamma(self, prod, t=0, spot=None, step=None, status: StatusType = None):
        """求t时刻，价格spot的gamma值,
            spot是价格的绝对值
            t是期权价值矩阵的列(时间)的索引
            step为微扰步长，默认为价格spot的1.0%，
            status为敲入状态"""
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        status = prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = interp1d(fdm_grid.s_vec, fdm_grid.v_grid[1:-1, t], kind='cubic')
        gamma = (f(spot + step) - 2 * f(spot) + f(spot - step)) / (step ** 2)
        return gamma

    def theta(self, prod, t=0, spot=None, step=1, status: StatusType = None):
        '''默认: 一天的theta'''
        spot = self.process.spot() if spot is None else spot
        status = prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        try:
            g = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t + step], kind='cubic')
            theta = (g(spot) - f(spot))
            return theta
        except IndexError:  # 可能遇到到期日估值，无法计算theta
            return np.nan

    def vega(self, prod, t=0, spot=None, step=0.01, status: StatusType = None):
        """vega todo:目前只支持常数波动率"""
        spot = self.process.spot() if spot is None else spot
        status = prod.status if status is None else status
        last_vol = self.process.vol.volval
        last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()
        fdm_grid0 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.vol.volval = last_vol + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        vega = (g(spot) - f(spot))
        self.process.vol.volval = last_vol
        self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid
        return vega

    def rho(self, prod, t=0, spot=None, step=0.01, status: StatusType = None):
        """rho todo:目前只支持常数无风险利率"""
        spot = self.process.spot() if spot is None else spot
        status = prod.status if status is None else status
        last_r = self.process.interest.data
        last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()
        fdm_grid0 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.interest.data = last_r + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        rho = (g(spot) - f(spot))
        self.process.interest.data = last_r
        self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid
        return rho

    # pylint: disable=too-many-locals
    def pv_and_greeks(self, prod, t=0, spot=None, status: StatusType = None):
        """当一次性计算pv和5个greeks时，可以调用此函数
        Returns:
            Dict[str:float]: {'pv': pv, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}"""
        spot = self.process.spot() if spot is None else spot
        status = prod.status if status is None else status
        s_step = spot * 0.01
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        pv = np.float64(f(spot))
        delta = (f(min(spot + s_step, self.smax)) - f(min(spot - s_step, self.smax))) / (2 * s_step)
        gamma = (f(spot + s_step) - 2 * f(spot) + f(spot - s_step)) / (s_step ** 2)
        if fdm_grid.v_grid.shape[1] > 1:
            g = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t + 1], kind='cubic')
            theta = (g(spot) - f(spot))
        else:
            theta = np.nan
        last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()

        last_vol = self.process.vol.volval
        self.process.vol.volval = last_vol + 0.01
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g1 = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        vega = (g1(spot) - f(spot))
        self.process.vol.volval = last_vol

        last_r = self.process.interest.data
        self.process.interest.data = last_r + 0.01
        self.calc_present_value(prod=prod)
        fdm_grid2 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g2 = fdm_grid2.functionize(fdm_grid2.v_grid[1:-1, t], kind='cubic')
        rho = (g2(spot) - f(spot))
        self.process.interest.data = last_r

        self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid

        return {'pv': pv, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def value_matrix(self, status: StatusType = None):
        """根据是否敲入状态，返回S-t矩阵"""
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        return fdm_grid.v_grid, fdm_grid.s_vec

    def delta_matrix(self, step=0, status: StatusType = None):
        """计算S-t矩阵中，每个t时刻所有s格点的delta
            step: 标的价步长，默认为0，即选择PDE有限差分网格的价格步长
            status为敲入状态"""
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        if step == 0:
            # 默认PDE模拟时的步长 step = fdm_grid.ds
            delta_s_t = (fdm_grid.v_grid[2:] - fdm_grid.v_grid[:-2]) / (2 * fdm_grid.ds)
            return delta_s_t, fdm_grid.s_vec  # 中心差分，价格向量掐头0去尾smax，正好是fdm_grid.s_vec
        else:
            # 重新定义步长，在模拟矩阵的基础上插值
            n_step = np.round(self.smax / step).astype('int')
            spots = np.arange(0, n_step, 1) * step
            # 下方n_step和spots用于避免计算远离敲入水平的场景，降低运算量
            # n_step = 2 * np.round((self.spot - self.barrier_in + 1) / step)
            # spots = np.arange(0, n_step, 1) * step + self.barrier_in - 1
            delta_s_t = np.zeros((n_step - 1, fdm_grid.v_grid.shape[1]))
            # 分别在fdm_grid.s_vec价格向量首位添加0和smax
            x_vec = np.insert(fdm_grid.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            for j in range(fdm_grid.v_grid.shape[1]):
                f = interp1d(x_vec, fdm_grid.v_grid[:, j], kind='cubic')
                for i, spot in enumerate(spots[1:]):
                    delta_s_t[i, j] = (f(spot + step) - f(spot - step)) / (step * 2)
            return delta_s_t, spots

    def gamma_matrix(self, step=0, status: StatusType = None):
        """计算S-t矩阵中，每个t时刻所有s格点的gamma
            step: 标的价步长，默认为0，即选择PDE有限差分网格的价格步长
            status为敲入状态"""
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        if step == 0:
            # 默认PDE模拟时的步长 step = fdm_grid.ds
            gamma_s_t = np.diff(fdm_grid.v_grid, n=2, axis=0) / fdm_grid.ds ** 2
            return gamma_s_t, fdm_grid.s_vec
        else:
            # 重新定义步长，在模拟矩阵的基础上插值
            n_step = np.round(self.smax / step).astype('int')
            step = self.smax / n_step
            spots = np.arange(0, n_step, 1) * step

            gamma_s_t = np.zeros((n_step - 2, fdm_grid.v_grid.shape[1]))
            # 分别在fdm_grid.s_vec价格向量首位添加0和smax
            x_vec = np.insert(fdm_grid.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            for j in range(fdm_grid.v_grid.shape[1]):
                f = interp1d(x_vec, fdm_grid.v_grid[:, j], kind='cubic')
                for i, spot in enumerate(spots[1:-2]):
                    gamma_s_t[i, j] = (f(spot + step) - 2 * f(spot) + f(spot - step)) / (step ** 2)
            return gamma_s_t, spots

    def theta_matrix(self, step=0, status: StatusType = None):
        """计算S-t矩阵中，每个s价格所有t时刻的theta'
            step: 时间步长，默认为0，即选择PDE有限差分网格的时间步长
            status为敲入状态"""
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        if step == 0:
            # 默认PDE模拟时的步长
            theta_s_t = np.diff(fdm_grid.v_grid, axis=1)  # / self.dt
            x_vec = np.insert(fdm_grid.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            return theta_s_t, x_vec
        else:
            raise NotImplementedError("todo: 时间步长与pde有限差分网格不同的theta.")


class FdmSnowBallEngine(FdmAutoCallableEngine):
    """ 雪球类 AutoCallable PDE 有限差分法定价引擎
        自动赎回Autocall Note(无敲入)/雪球式Snowball Autocallable/触发器Trigger Autocallable
            自动赎回：无敲入（敲入价为-1），有红利票息（未敲出有保底收益），为保本产品
            雪球式：有敲入，敲入后下跌部分保护，敲出年化票息
            触发器：有敲入，敲入后下跌部分保护，敲出绝对票息
        支持变敲出、变票息、变敲入、雪球plus、保底雪球、OTM、触发式等多种结构
    """

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

        # 由看涨看跌平价 call-put=S*exp(-qt)-K*exp(-rt)，S=0时call=0，put=K*exp(-rt)
        self.fd_not_in.v_grid[0, :] = self.fd_knockin.v_grid[0, :] = (
                (prod.margin_lvl * prod.s0 - prod.strike_upper + prod.strike_lower)
                * interest_maturity_discount_factor)

        # 矩阵最后一行: 每一个观察日前，股价上界能得到的本金加票息贴现
        self.next_paydate = self.pay_dates.repeat(
            np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))  # 计息时间数
        self.next_paydate = np.append(self.pay_dates[0], self.next_paydate)  # 每个交易日对应的下一个敲出观察日对应的付息日

        # 触发器：上边界为绝对票息贴现/ 非触发器：上边界为年化票息贴现
        coupon_t = 1 if prod.trigger else self.next_paydate
        # 由看涨看跌平价 call-put = S*exp(-qt) - K*exp(-rt)，S=Smax时put=0，call = Smax*exp(-qt) - K*exp(-rt)
        interest_next_paydate_discount_factor = self.process.interest.disc_factor(self.next_paydate, t_vec)
        div_next_paydate_discount_factor = self.process.div.disc_factor(self.next_paydate, t_vec)
        v_smax = ((smax * div_next_paydate_discount_factor
                   - prod.strike_call * interest_next_paydate_discount_factor)
                  * prod.parti_out + (prod.margin_lvl + self.coupon_outs * coupon_t) * prod.s0
                  * interest_next_paydate_discount_factor)
        self.fd_knockin.v_grid[self.s_step, :] = self.fd_not_in.v_grid[self.s_step, :] = v_smax

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
            self.coupon_outs = np.full(self.t_step + 1, prod.coupon_out)  # N天的敲出票息列表，用于生成mspot价格上边界条件
        elif isinstance(prod.coupon_out, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._coupon_out = prod.coupon_out[-len(self.out_dates):].copy()
            # 浮动敲出票息
            self.next_coupon_out = self._coupon_out[-1]
            self.reversed_coupon_out = np.flip(np.array(self._coupon_out))
            # N天的敲出票息列表，用于生成smax价格上边界条件
            coupon_outs = np.array(self._coupon_out).repeat(
                np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))
            self.coupon_outs = np.append(self._coupon_out[0], coupon_outs)
        else:
            raise ValueError(f'敲出票息类型为{type(prod.coupon_out)}，仅支持int/float/list/np.ndarray，请检查')

        # in_idx：小于到期日敲入线的最大的标的价格索引。如果是变敲入雪球，此变量in_idx应该随敲入价动态调整。
        if self.next_barrier_in > 0 and s_vec[0] < self.next_barrier_in:
            self.in_idx = 1 + np.where(s_vec <= self.next_barrier_in)[0][-1]
        else:
            self.in_idx = 0
        # out_idxs：大于到期日敲出线的标的价格数组的索引。如果是变敲出雪球，此变量out_idxs应该随敲出价动态调整。
        self.out_idxs = 1 + np.array(np.where(s_vec >= self.next_barrier_out)[0])
        # 虚值call在值度
        self.itm = np.where(s_vec - prod.strike_call > 0, (s_vec - prod.strike_call) * prod.parti_out, 0)

        coupon_t = 1 if prod.trigger else self.pay_dates[-1]  # 票息是否年化
        # 矩阵最后一列：到期payoff。考虑到了保底、保证金\纯期权模式、以及敲出边界上方增加虚值call
        """已敲入：如果到期小于敲出边界，在100%本金模式下，小于保底边界，获得保底价；
                                                  大于保底边界，获得到期价格；
                                   在保证金模式下，小于保底边界，支付期初价（高行权价）与保底价之间价差；
                                                  大于保底边界，支付到期价格与期初价之间价差；
                  如果到期大于敲出边界，大于看涨行权价，获得看涨收益、本金和票息；
                                    在敲出边界和看涨行权价之间，获得本金和票息"""
        self.fd_knockin.v_grid[1:-1, self.t_step] = np.where(s_vec < self.next_barrier_out,
                                             np.minimum(np.where(s_vec <= prod.strike_lower, prod.strike_lower, s_vec) -
                                                 prod.strike_upper, 0) + prod.s0 * prod.margin_lvl,
                                             np.where(s_vec > prod.strike_call,
                                                      (s_vec - prod.strike_call) * prod.parti_out + (
                                                           prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                      (prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0))
        """未敲入：如果到期小于敲入边界，在100%本金模式下，小于保底边界，获得保底价；
                                                  大于保底边界，获得实际价格；
                                  在保证金模式下，小于保底边界，支付保底价与期初价之间价差；
                                                  大于保底边界，支付到期价格与期初价之间价差；
                 如果到期大于敲入边界，且大于看涨行权价，获得看涨收益、本金和票息；
                                  在敲入边界和看涨行权价之间，获得本金和票息
                                          如果在敲出边界和看涨行权价之间，获得本金和敲出票息
                                          如果在敲入边界和敲出边界之间，获得本金和红利票息"""
        self.fd_not_in.v_grid[1:-1, self.t_step] = np.where(s_vec <= self.next_barrier_in,
                                        np.where(s_vec <= prod.strike_lower, prod.strike_lower, s_vec) -
                                            prod.strike_upper + prod.s0 * prod.margin_lvl,
                                        np.where(s_vec > prod.strike_call,
                                                 (s_vec - prod.strike_call) * prod.parti_out + (
                                                     prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                 np.where(s_vec >= self.next_barrier_out,
                                                          (prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0,
                                                          (prod.margin_lvl + prod.coupon_div * coupon_t) * prod.s0)))

    def _backward_induction(self):
        """从倒数第二天向第一天逆向迭代"""
        prod = self.prod
        self.next_diff_obspaydate = self.diff_obs_pay_dates.repeat(
            np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))  # 计息时间数
        self.next_diff_obspaydate = np.append(self.diff_obs_pay_dates[0],
                                              self.next_diff_obspaydate)  # 每个交易日对应的下一个敲出观察日对应的付息日与观察日的自然日年化之差

        if prod.in_obs_type == ExerciseType.European:  # 敲入观察为欧式，仅到期观察敲入
            self.next_barrier_in = -1
            self.in_idx = 0
        for j in range(self.t_step - 1, -1, -1):
            self.fd_knockin.v_grid[1:-1, j] = self.fd_knockin.evolve(j, self.fd_knockin.v_grid[1:-1, j + 1], self.dt)
            self.fd_not_in.v_grid[1:-1, j] = self.fd_not_in.evolve(j, self.fd_not_in.v_grid[1:-1, j + 1], self.dt)

            # 考虑敲出的情况: 敲出观察日边界条件需更改
            if j in self.out_dates:
                coupon_t = 1 if prod.trigger else self.next_paydate[j]
                # 如果敲出价是浮动的，调整敲出价
                self.next_barrier_out = (self.reversed_barrier_out[np.where(self.out_dates == j)[0][0]]
                                         if isinstance(prod.barrier_out, (list, np.ndarray)) else prod.barrier_out)
                self.out_idxs = 1 + np.array(np.where(self.fd_not_in.s_vec >= self.next_barrier_out)[0])
                # 如果敲出票息是浮动的，调整敲出票息
                self.next_coupon_out = (self.reversed_coupon_out[np.where(self.out_dates == j)[0][0]]
                                        if isinstance(prod.coupon_out, (list, np.ndarray)) else prod.coupon_out)
                # 发生敲出的payoff
                knock_out_payoff = (self.itm[self.out_idxs - 1] + (
                        prod.margin_lvl + self.next_coupon_out * coupon_t) * prod.s0
                                    ) * self.process.interest.disc_factor(self.next_paydate[j], self.next_paydate[j] -
                                                                          self.next_diff_obspaydate[j])
                # 用knock_out_payoff覆盖敲出价上方的衍生品价值
                self.fd_not_in.v_grid[self.out_idxs, j] = self.fd_knockin.v_grid[self.out_idxs, j] = knock_out_payoff

                if prod.in_obs_type != ExerciseType.European:
                    # 如果敲入价是浮动的，调整敲入价
                    self.next_barrier_in = self.reversed_barrier_in[np.where(self.out_dates == j)[0][0]] if isinstance(
                        prod.barrier_in, (list, np.ndarray)) else prod.barrier_in
                    self.in_idx = 1 + np.where(self.fd_not_in.s_vec <= self.next_barrier_in)[0][
                        -1] if self.next_barrier_in > 0 else 0

            # 考虑敲入的情况: 向下越过敲入边界时，下一时间点敲入边界以上部分是标准autocall价格
            self.fd_not_in.v_grid[:(self.in_idx + 1), j] = self.fd_knockin.v_grid[:(self.in_idx + 1), j]


class FdmPhoenixEngine(FdmAutoCallableEngine):
    """ 凤凰类 AutoCallable PDE 有限差分法定价引擎：
            较雪球式AutoCallable，增加派息价格，少了敲出和红利票息
            每个派息（敲出）观察日，如果价格高于派息价格，派固定利息，如果发生敲出，合约提前结束；
            发生敲入后，派息方式不变，到期如果未敲出，结构为看跌空头或熊市价差空头
        FCN(Fixed Coupon Note, 固定派息票据)/DCN(Digital Coupon Note, 二元派息票据)/Phoenix凤凰票据
            FCN：无派息价格（派息价格为0），每月固定派息；到期日观察是否敲入
            DCN：有派息价格，每个观察日如果价格高于派息价格，派固定利息；到期日观察是否敲入
            Phoenix：有派息价格，每个观察日如果价格高于派息价格，派固定利息；每日观察是否敲入
    """

    def _init_boundary_condition(self, smax, maturity):
        """初始化边界条件
        Args:
            smax: float, 标的价格上界
            maturity: float, 到期时间，年化自然日期限
        Returns: None
        """
        prod = self.prod
        # 矩阵第一行: 标的价格为0时结构的价值
        t_vec = self.dt * self.j_vec
        interest_maturity_discount_factor = self.process.interest.disc_factor(maturity, t_vec)
        if np.any(prod.barrier_yield) > 0:
            # 如果派息价格不为0，则无派息, (预付金+敲入亏损)*贴现
            self.fd_not_in.v_grid[0, :] = self.fd_knockin.v_grid[0, :] = ((prod.strike_lower - prod.strike_upper)
                                                                          * prod.parti_in + prod.margin_lvl * prod.s0
                                                                          ) * interest_maturity_discount_factor
        else:
            # 如果派息价格为0，即FCN，则派息随着每个派息日累积，(累计派息+预付金+敲入亏损)*贴现
            self.fd_not_in.v_grid[0, self.t_step] = self.fd_knockin.v_grid[0, self.t_step] = (
                    (prod.strike_lower - prod.strike_upper)
                    * prod.parti_in + (prod.margin_lvl + self.next_coupon) * prod.s0)
            for j in range(self.t_step - 1, -1, -1):  # todo: 对常数无风险利率做优化，避免重复计算折现因子
                self.fd_not_in.v_grid[0, j] = self.fd_knockin.v_grid[0, j] = (self.fd_knockin.v_grid[0, j + 1]
                                                                              * self.process.interest.disc_factor(
                            j * self.dt, (j + 1) * self.dt))
                if j in self.out_dates:
                    self.fd_not_in.v_grid[0, j] = self.fd_knockin.v_grid[0, j] = (self.fd_knockin.v_grid[0, j]
                                                                                  + self.coupons[j] * prod.s0)

        # 矩阵最后一行: 每一个观察日前，股价上界能得到的本金加票息贴现
        self.next_paydate = self.pay_dates.repeat(
            np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))  # 计息时间数
        self.next_paydate = np.append(self.pay_dates[0], self.next_paydate)  # 每个交易日对应的下一个敲出观察日对应的付息日
        # 每一个观察日前，股价上界能得到：(派息+预付金)*贴现。这里派息次数取经过的敲出观察日的次数，这是近似的取值，因为派息次数是路径依赖的。
        # 由于股价上界很大，近似认为之前的每次派息日，价格都足够高，每次都可以获得派息。
        interest_next_paydate_discount_factor = self.process.interest.disc_factor(self.next_paydate, t_vec)
        v_smax = (prod.margin_lvl + self.coupons) * prod.s0 * interest_next_paydate_discount_factor
        self.fd_knockin.v_grid[self.s_step, :] = self.fd_not_in.v_grid[self.s_step, :] = v_smax

    def _init_terminal_condition(self, s_vec, maturity):
        """初始化终止时已敲入且未敲出、未敲入且未敲出的期权价值
        Args:
            s_vec: List[float], 网格的价格格点
            maturity: float, 到期时间，年化自然日期限
        Returns: None
        """
        prod = self.prod
        # 初始化next_barrier_in、next_barrier_out、next_coupon为最后一期的敲入价、敲出价、敲出票息。若变敲出变敲入，需要动态调整
        # 设置到期日的敲入边界位置
        if isinstance(prod.barrier_in, (int, float, np.int32, np.float64)):
            # 常数敲入
            self.next_barrier_in = prod.barrier_in
        elif isinstance(prod.barrier_in, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._barrier_in = prod.barrier_in[-len(self.out_dates):].copy()
            # 浮动敲入
            self.next_barrier_in = self._barrier_in[-1]
            self.reversed_barrier_in = np.flip(np.array(self._barrier_in))
        else:
            raise ValueError(f'敲入障碍价类型为{type(prod.barrier_in)}，仅支持int/float/list/np.ndarray，请检查')

        # 设置到期日的敲出边界位置
        if isinstance(prod.barrier_out, (int, float, np.int32, np.float64)):
            # 常数敲出
            self.next_barrier_out = prod.barrier_out
        elif isinstance(prod.barrier_out, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._barrier_out = prod.barrier_out[-len(self.out_dates):].copy()
            # 阶梯式敲出
            self.next_barrier_out = self._barrier_out[-1]
            self.reversed_barrier_out = np.flip(np.array(self._barrier_out))
        else:
            raise ValueError(f'敲出障碍价类型为{type(prod.barrier_out)}，仅支持int/float/list/np.ndarray，请检查')

        # 设置到期日的派息边界位置
        if isinstance(prod.barrier_yield, (int, float, np.int32, np.float64)):
            # 常数派息
            self.next_barrier_yield = prod.barrier_yield
        elif isinstance(prod.barrier_yield, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._barrier_yield = prod.barrier_yield[-len(self.out_dates):].copy()
            # 浮动派息
            self.next_barrier_yield = self._barrier_yield[-1]
            self.reversed_barrier_yield = np.flip(np.array(self._barrier_yield))

        # 敲出票息
        if isinstance(prod.coupon, (int, float, np.int32, np.float64)):
            # 常数敲出票息
            self.next_coupon = prod.coupon
            self.coupons = np.full(self.t_step + 1, prod.coupon)  # N天的敲出票息列表，用于生成mspot价格上边界条件
        elif isinstance(prod.coupon, (list, np.ndarray)):
            # self.out_dates是估值日之后的、经过截断的敲出观察日，取产品参数列表中最后len(self.out_dates)个，与其对应
            self._coupon = prod.coupon[-len(self.out_dates):].copy()
            # 浮动敲出票息
            self.next_coupon = self._coupon[-1]
            self.reversed_coupon = np.flip(np.array(self._coupon))
            # N天的敲出票息列表，用于生成smax价格上边界条件
            coupons = np.array(self._coupon).repeat(
                np.diff(np.append(np.zeros((1,)), self.out_dates[::-1])).astype(int))
            self.coupons = np.append(self._coupon[0], coupons)
        else:
            raise ValueError(f'敲出票息类型为{type(prod.coupon)}，仅支持int/float/list/np.ndarray，请检查')

        # in_idx：小于到期日敲入线的最大的标的价格索引。如果是变敲入，此变量in_idx应该随敲入价动态调整。
        if self.next_barrier_in > 0 and s_vec[0] <= self.next_barrier_in:
            self.in_idx = 1 + np.where(s_vec <= self.next_barrier_in)[0][-1]
        else:
            self.in_idx = 0
        # out_idxs：大于到期日敲出线的标的价格数组的索引。如果是变敲出，此变量out_idxs应该随敲出价动态调整。
        self.out_idxs = 1 + np.array(np.where(s_vec >= self.next_barrier_out)[0])
        # yld_idxs: 大于到期日派息边界位置的标的价格数组的索引。如果是变派息，此变量yld_idxs应该随敲出价动态调整。
        self.yld_idxs = 1 + np.array(np.where(s_vec >= self.next_barrier_yield)[0])

        # 矩阵最后一列：到期payoff
        # 已敲入：派息 + 敲入或有亏损 + 预付金
        self.fd_knockin.v_grid[1:-1, self.t_step] = np.where(s_vec > self.next_barrier_yield,
                                                             self.next_coupon * prod.s0,
                                                             0) + \
                                                    np.where(s_vec > prod.strike_upper, 0,
                                                             (np.where(s_vec > prod.strike_lower, s_vec,
                                                                       prod.strike_lower) - prod.strike_upper) *
                                                             prod.parti_in) + prod.margin_lvl * prod.s0
        # 未敲入：派息 + 到期敲入或有亏损 + 预付金
        self.fd_not_in.v_grid[1:-1, self.t_step] = np.where(s_vec > self.next_barrier_yield,
                                                            self.next_coupon * prod.s0,
                                                            0) + \
                                                   np.where(s_vec > self.next_barrier_in, 0,
                                                            # 敲入亏损
                                                            (np.where(s_vec > prod.strike_lower, s_vec,
                                                                      prod.strike_lower) - prod.strike_upper) *
                                                            prod.parti_in) + prod.margin_lvl * prod.s0

    def _backward_induction(self):
        """从倒数第二天向第一天逆向迭代"""
        prod = self.prod
        if prod.in_obs_type == ExerciseType.European:  # 敲入观察为欧式，仅到期观察敲入
            self.next_barrier_in = -1
            self.in_idx = 0
        for j in range(self.t_step - 1, -1, -1):
            self.fd_knockin.v_grid[1:-1, j] = self.fd_knockin.evolve(j, self.fd_knockin.v_grid[1:-1, j + 1], self.dt)
            self.fd_not_in.v_grid[1:-1, j] = self.fd_not_in.evolve(j, self.fd_not_in.v_grid[1:-1, j + 1], self.dt)

            # 考虑敲出的情况: 敲出观察日边界条件需更改
            if j in self.out_dates:

                # 如果敲出价是浮动的，调整敲出价
                self.next_barrier_out = (self.reversed_barrier_out[np.where(self.out_dates == j)[0][0]]
                                         if isinstance(prod.barrier_out, (list, np.ndarray)) else prod.barrier_out)
                self.out_idxs = 1 + np.array(np.where(self.fd_not_in.s_vec >= self.next_barrier_out)[0])
                # 如果派息边界是浮动的，调整派息边界
                self.next_barrier_yield = (self.reversed_barrier_yield[np.where(self.out_dates == j)[0][0]]
                                           if isinstance(prod.barrier_yield,
                                                         (list, np.ndarray)) else prod.barrier_yield)
                self.yld_idxs = 1 + np.array(np.where(self.fd_not_in.s_vec >= self.next_barrier_yield)[0])
                # 如果敲出票息是浮动的，调整敲出票息
                self.next_coupon = (self.reversed_coupon[np.where(self.out_dates == j)[0][0]]
                                    if isinstance(prod.coupon, (list, np.ndarray)) else prod.coupon)
                # 在票息边界和敲出边界之间，无敲入phoenix的价格为亏损贴现加上票息
                margin_s = prod.margin_lvl * prod.s0
                self.fd_knockin.v_grid[self.yld_idxs, j] = (self.next_coupon * prod.s0 + np.where(
                    self.fd_not_in.s_vec[self.yld_idxs - 1] < self.next_barrier_out,
                    self.fd_knockin.v_grid[self.yld_idxs, j], margin_s))
                self.fd_not_in.v_grid[self.yld_idxs, j] = (self.next_coupon * prod.s0 + np.where(
                    self.fd_not_in.s_vec[self.yld_idxs - 1] < self.next_barrier_out,
                    self.fd_not_in.v_grid[self.yld_idxs, j], margin_s))

                if prod.in_obs_type != ExerciseType.European:
                    # 如果敲入价是浮动的，调整敲入价
                    self.next_barrier_in = self.reversed_barrier_in[np.where(self.out_dates == j)[0][0]] if isinstance(
                        prod.barrier_in, (list, np.ndarray)) else prod.barrier_in
                    self.in_idx = 1 + np.where(self.fd_not_in.s_vec <= self.next_barrier_in)[0][
                        -1] if self.next_barrier_in > 0 else 0

            # 考虑敲入的情况: 向下越过敲入边界时，下一时间点敲入边界以上部分是标准autocall价格
            self.fd_not_in.v_grid[:(self.in_idx + 1), j] = self.fd_knockin.v_grid[:(self.in_idx + 1), j]
