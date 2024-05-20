#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from contextlib import suppress
import numpy as np
from scipy.interpolate import interp1d
from pricelib.common.utilities.enums import EngineType, CallPut, InOut, PaymentType, StatusType
from pricelib.common.utilities.utility import descending_pairs_for_barrier
from pricelib.common.pricing_engine_base import FdmEngine, FdmGridwithBound, FdmGrid
from pricelib.common.time import global_evaluation_date


class FdmBarrierEngine(FdmEngine):
    """障碍期权PDE有限差分法定价引擎
    只支持美式观察(整个有效期观察是否触碰)；支持连续观察、均匀离散观察(默认为每日观察)；
    敲入现金返还为到期支付；敲出现金返还支持 立即支付/到期支付"""
    engine_type = EngineType.PdeEngine

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
        self.bound = None  # 障碍价格区间
        self.rebate = None  # 补偿回报
        self.window = None  # 观察时间窗口
        self.fd_not_in = None  # 未敲入的有限差分网格FdmGrid对象
        self.fd_knockin = None  # 已敲入的有限差分网格FdmGrid对象
        self.lower = None  # 小于低障碍价的标的价格bool索引
        self.upper = None  # 大于高障碍价的标的价格bool索引
        self.outer = None  # 小于低障碍价或大于高障碍价，即在两个障碍价格之外的bool索引
        self.inner = None  # 大于低障碍价且小于高障碍价，即在两个障碍价格之间的bool索引
        self.smax = None  # 价格网格上界
        self.fn_callput = None  # 无障碍价格的边界条件
        self.fn_bound = None  # 障碍价格的边界条件

    def _init_boundary_condition(self, smax, maturity):
        """生成无障碍价格的边界条件
        在s=0处，call近似为0; 在smax处，put近似为0。根据看涨看跌平价 call-put = S*exp(-qT) - K*exp(-rT),
        可得对应的put(0) = K * exp(-rT), call(smax) = smax * exp(-qT) - K * exp(-rT)
        Args:
            smax: float, 有限差分网格上边界的价格最大值
            maturity: float, 到期时间
        Returns:
            self.fn_callput: List[Callable], 0~smax网格的边界条件，callput = self.fn_callput[0](s) or self.fn_callput[1](s)
            self.fn_bound: List[Callable], 障碍价格网格的边界条件，bound = self.fn_bound[0](s) or self.fn_bound[1](s)
        """
        if self.prod.callput == CallPut.Call:
            self.fn_callput = [lambda u: 0, lambda u: (smax * self.process.div.disc_factor(maturity, u) -
                                                       self.prod.strike * self.process.interest.disc_factor(maturity,
                                                                                                            u)) * self.prod.parti]
        else:
            self.fn_callput = [lambda u: self.prod.strike * self.process.interest.disc_factor(maturity, u)
                                         * self.prod.parti, lambda u: 0]

        # 生成障碍价格的Dirichlet边界条件： 第u天触碰障碍产生的补偿，折现到第u天的价值。
        # 返回一个包含两个lambda函数的列表，列表索引分别为[0, 1]，每个lambda函数的形参r=self.rebate中的r补偿值
        # 需要注意的是，敲出期权rebate[0]是下边界补偿，敲出期权rebate[1]是上边界补偿；敲入期权rebate[0]是敲入补偿，敲入期权rebate[1]没有用处。
        self.fn_bound = []
        for i, r in enumerate(self.rebate):
            if self.bound[i] in [-np.inf, np.inf]:  # 障碍价无穷，即障碍价没有设置
                if self.prod.inout == InOut.Out:  # 敲出期权，未设置障碍时，边界条件是香草期权的payoff
                    self.fn_bound.append(self.fn_callput[i])
                else:  # 敲入期权，未设置障碍时，边界条件是未敲入补偿rebate
                    self.fn_bound.append(lambda u: self.rebate[0] * self.process.interest.disc_factor(maturity, u))
            else:  # 这里只设置敲出期权的边界条件，因为敲入期权的边界条件是由已敲入的期权价值覆盖的
                if self.prod.inout == InOut.Out:
                    if self.prod.payment_type == PaymentType.Hit:  # 触碰障碍, 立即支付现金补偿
                        self.fn_bound.append(lambda u: r)
                    else:  # 'PaymentMethod.Expire' 触碰障碍后，到期时支付现金补偿
                        self.fn_bound.append(lambda u: r * self.process.interest.disc_factor(maturity, u))
                else:
                    self.fn_bound.append(self.fn_callput[i])

    def _init_terminal_condition(self, s_vec):
        """终止条件
        连续观察（网格上下界是障碍价格）：初始化终止时已敲入/未敲出的期权价值
        离散观察（网格上下界是0和smax）：初始化终止时的期权价值
        """
        yv = np.maximum(self.prod.callput.value * (s_vec - self.prod.strike), 0) * self.prod.parti
        if self.prod.discrete_obs_interval is None:  # 连续观察
            return yv  # 已敲入/未敲出 都是香草
        else:  # 离散观察
            if self.prod.inout == InOut.In:  # 敲入障碍 # 已敲入
                return yv
            else:  # 敲出障碍  # 未敲出香草，已敲出rebate
                yv[self.lower] = self.rebate[0]
                yv[self.upper] = self.rebate[1]
                return yv

    def _init_no_touch_terminal_condition(self, s_vec, p=0, maturity=0):
        """初始化障碍观察区间终点未敲入的期权价值
            连续观察（网格上下界是障碍价格）：此时未敲入期权价值全部是补偿rebate
            离散观察（网格上下界是0和smax）：此时将未敲入部分是rebate, 敲入部分是香草
        """
        if self.prod.discrete_obs_interval is None:  # 连续观察
            yv = s_vec * 0 + self.rebate[0] * self.process.interest.disc_factor(maturity, p)
        else:  # 离散观察
            yv = np.maximum(self.prod.callput.value * (s_vec - self.prod.strike), 0) * self.prod.parti
            yv[self.inner] = self.rebate[0]
        return yv

    # pylint: disable=too-many-locals, too-many-branches, too-many-locals
    def calc_present_value(self, prod, t=None, spot=None, bound=(None, None), rebate=(0, 0),
                           window=(None, None)):
        """计算现值，PDE有限差分定价，时间逆向迭代
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
            bound: (lower_barrier, upper_barrier), 低障碍价/高障碍价，障碍价格绝对数值。如果是默认值None，即为无穷大或无穷小inf。
            rebate: (lower_rebate, upper_rebate), 对敲出期权，元组第一位是低障碍价下方的补偿，第二位是高障碍价上方的补偿，
                                                  对敲入期权，元组第一位是敲入补偿，第二位没有用处。
            window: Tuple/List[float]，障碍观察时间窗口，(start, end)，距离估值日的年化期限
                    如果是默认值None，说明观察区间是整个存续期: 估值日-终止日
        Returns: float，现值
        """
        self.prod = prod
        self.bound = np.array([(2 * i - 1) * np.inf if b is None else b for i, b in enumerate(bound)])  # 障碍价格区间
        self.rebate = np.array(rebate)  # 补偿回报
        self.window = np.array(window)  # 观察时间窗口

        calculate_date = global_evaluation_date() if t is None else t  # 估值日
        # 剩余到期时间（交易日天数）
        maturity = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()
        self.smax = self.n_smax * spot  # 价格网格上界, n_smax倍初始价格

        window_start = 0 if self.window[0] is None else max(self.window[0], 0)  # 观察时间窗口开始start时间
        window_end = maturity if self.window[1] is None else max(self.window[1], 0)  # 观察时间窗口截止end时间

        # 返回PDE系数组件abc的函数
        fn_pde_coef = self.process.get_fn_pde_coef(maturity, spot)

        # 如果障碍是连续观察，网格上下界应该是障碍价格，如果障碍是离散观察，网格上下界应该是0~4倍初始价格
        if prod.discrete_obs_interval is None:  # 连续观察
            bound_rate = (min(self.smax, self.bound[1]) - max(0, self.bound[0])) / (self.smax - 0)  # 障碍价格占价格网格上下界的比例
            fd_full = FdmGridwithBound(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                       fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta)  # 无障碍价格，[0, smax]
            fd_bound = FdmGridwithBound(smax=self.smax, t_step_per_year=prod.t_step_per_year,
                                        s_step=round(self.s_step * bound_rate),
                                        fn_pde_coef=fn_pde_coef, bound=self.bound,
                                        fdm_theta=self.fdm_theta)  # 有障碍价格, Dirichlet边界条件
            self._init_boundary_condition(self.smax, maturity)
            fd_full.set_boundary_condition(self.fn_callput)

            self.lower = fd_full.s_vec <= self.bound[0]  # 小于低障碍价的标的价格bool索引
            self.upper = fd_full.s_vec >= self.bound[1]  # 大于高障碍价的标的价格bool索引
            self.outer = self.lower | self.upper  # 小于低障碍价或大于高障碍价，即在两个障碍价格之外的bool索引

            time_pairs = descending_pairs_for_barrier(0, window_start, window_end,
                                                      maturity)  # 将一系列时间从大到小排序，去重，包装成相邻的降序对
            fdm = fd_full  # fdm是当前时间点的fdm，初始值是fd_full，后面会根据时间点的变化，切换成fd_full或fd_bound
            # 从到期日从后向前倒推，此时无论敲入还是敲出，未敲入和已敲出的payoff已经固定是rebate无需计算，只有已敲入或未敲出的期权还需要计算价值。
            # df_diri代表未敲入或已敲出。到期日的fd_full代表已敲入或未敲出。

            if prod.inout == InOut.In:  # 敲入障碍。
                # 初始化到期日已敲入payoff价值yv=max(callput * (F-K), 0)
                yv = self._init_terminal_condition(fdm.s_vec)
                # 从到期日开始向前迭代，fd_full是已敲入。fd_bound是未敲入。
                # 由于障碍观察区间之后，未敲入的期权价值固定是rebate，所以不需要计算fd_bound这一部分，只需要计算fd_full。
                for p, q in time_pairs:  # p是起点t{j}, q是终点t{j-1}, p>q由后向前倒推
                    if p == window_end:  # 进入障碍观察区间，开始计算未敲入的期权价值fd_bound，此时fd_full是已敲入的。
                        yb = yv  # 记录进入观察区间之前的yv，称为yb(boundaries)，yb是已敲入的期权价值
                        # 初始化p时刻(观察区间终点)仍然未敲入的fd_bound的期权价值向量yv，yv全部是未敲入的补偿rebate
                        yv = self._init_no_touch_terminal_condition(fd_bound.s_vec, p, maturity)
                        # 每步更新的fdm切换成fd_bound，因为未敲入的fd_bound的边界条件需要被已敲入的fd_full覆盖，所以要先算fd_full，再算fd_bound
                        fdm = fd_bound
                    if window_end >= p > window_start:  # 处于障碍观察期间内。注意p == w_end也算在内。
                        # 在观察区间内。此时fd_full和fd_bound都在更新: fd_full迭代yb，fd_bound迭代yv。将未敲入的fd_bound的边界条件用已敲入的fd_full覆盖。
                        yb, bc_fn = fd_full.evolve_with_interval(p, q, yb, self.bound)
                        for i, b in enumerate(self.bound):  # 如果向上/向下敲入未设置边界，则边界条件应该是未敲入状态
                            if b in [-np.inf, np.inf]:
                                bc_fn[i] = self.fn_bound[i]
                        fd_bound.set_boundary_condition(bc_fn)
                    if p == window_start:
                        # 退出观察区间，此时在障碍价格之外部分的yv覆盖为已敲入或已敲出的yb，每步只更新fd_full。
                        # 这里只更新fd_full的原因，是因为在观察区间起点之前，不可能敲入，所有的价格路径都是未敲入的状态；最早可能敲入的点在观察区间起点。
                        # 而在敲入观察区间第一天，在outer外的部分payoff会变成rebate，因此将障碍价格之外部分的yv覆盖为已敲入或已敲出的yb。
                        yv = fd_bound.functionize(yv)(fd_full.s_vec)
                        yv[self.outer] = yb[self.outer]
                        fdm = fd_full
                    yv = fdm.evolve_with_interval(p, q, yv)  # 每组(p, q)更新。如果在障碍观察区间内，更新fd_bound；在观察区间外，更新fd_full
            else:  # 敲出障碍
                # 初始化到期日未敲出payoff价值yv=max(callput * (F-K), 0)
                yv = self._init_terminal_condition(fdm.s_vec)
                fd_bound.set_boundary_condition(self.fn_bound)
                for p, q in time_pairs:  # 倒序时间对，p>=q
                    if p == window_end:
                        # 进入观察区间，后面的fd_full都是未敲出的，因为已敲出的期权已经失效。fd_bound也是未敲出，只是比df_full加上了敲出障碍的边界条件。
                        yv = fd_full.functionize(yv)(fd_bound.s_vec)
                        fdm = fd_bound
                    if p == window_start:
                        # 退出观察区间，前面的fd_full都是不可能敲出的。需要将yv的lower和upper部分替换为rebate。
                        # 因为敲出观察区间第一天，在outer的部分会立即变成敲出，payoff会变成rebate
                        yv = fd_bound.functionize(yv)(fd_full.s_vec)
                        yv[self.lower] = self.fn_bound[0](p)
                        yv[self.upper] = self.fn_bound[1](p)
                        fdm = fd_full
                    yv = fdm.evolve_with_interval(p, q, yv)  # 每组(p, q)更新。如果在障碍观察区间内，更新fd_bound；在观察区间外，更新fd_full

            return np.float64(fdm.functionize(yv)(spot))
        else:  # 离散观察
            t_step = round(prod.t_step_per_year * maturity)  # 时间步数
            dt = 1 / prod.t_step_per_year
            fn_pde_coef = self.process.get_fn_pde_coef(maturity, spot)
            self.fd_not_in = FdmGrid(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                     fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta,
                                     maturity=maturity)
            self._init_boundary_condition(smax=self.smax, maturity=maturity)
            self.fd_not_in.v_grid[0, :] = self.fn_bound[0](self.fd_not_in.tv)
            self.fd_not_in.v_grid[-1, :] = self.fn_bound[1](self.fd_not_in.tv)

            self.lower = self.fd_not_in.s_vec <= self.bound[0]  # 小于低障碍价的标的价格bool索引
            self.upper = self.fd_not_in.s_vec >= self.bound[1]  # 大于高障碍价的标的价格bool索引
            self.outer = self.lower | self.upper  # 小于低障碍价或大于高障碍价，即在两个障碍价格之外的bool索引
            self.inner = ~self.lower & ~self.upper

            obs_points = np.flip(np.round(prod.t_step_per_year *
                                          np.arange(maturity, 0, -prod.discrete_obs_interval)).astype(int))
            obs_points = np.concatenate((np.array([0]), obs_points))

            if prod.inout == InOut.In:  # 敲入障碍
                self.fd_knockin = FdmGrid(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                          fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta, maturity=maturity)
                self.fd_knockin.v_grid[-1, :] = self.fn_callput[1](self.fd_knockin.tv)
                self.fd_knockin.v_grid[0, :] = self.fn_callput[0](self.fd_knockin.tv)
                # 初始化到期日payoff价值
                yv = self._init_no_touch_terminal_condition(self.fd_not_in.s_vec)
                yv_in = self._init_terminal_condition(self.fd_knockin.s_vec)

                self.fd_not_in.v_grid[1:-1, -1] = yv
                self.fd_knockin.v_grid[1:-1, -1] = yv_in

                for j in range(t_step - 1, -1, -1):
                    yv = self.fd_not_in.evolve(j, yv, dt)  # 每组(p, q)更新。如果在障碍观察区间内，更新fd_bound；在观察区间外，更新fd_full
                    yv_in = self.fd_knockin.evolve(j, yv_in, dt)
                    if j in obs_points:
                        yv[self.outer] = yv_in[self.outer]
                    self.fd_not_in.v_grid[1:-1, j] = yv
                    self.fd_knockin.v_grid[1:-1, j] = yv_in
            else:  # 敲出障碍
                # 初始化到期日payoff价值
                yv = self._init_terminal_condition(self.fd_not_in.s_vec)
                self.fd_not_in.v_grid[1:-1, -1] = yv
                for j in range(t_step - 1, -1, -1):
                    yv = self.fd_not_in.evolve(j, yv, dt)  # 每组(p, q)更新。如果在障碍观察区间内，更新fd_bound；在观察区间外，更新fd_full
                    if j in obs_points:
                        if prod.payment_type == PaymentType.Expire:  # 到期支付现金补偿
                            yv[self.lower] = self.fn_bound[0](j / t_step * maturity)
                            yv[self.upper] = self.fn_bound[1](j / t_step * maturity)
                        else:  # 立即支付现金补偿
                            yv[self.lower] = self.fn_bound[0](maturity)
                            yv[self.upper] = self.fn_bound[1](maturity)
                    self.fd_not_in.v_grid[1:-1, j] = yv
            return np.float64(self.fd_not_in.functionize(yv, kind="cubic")(spot))

    def delta(self, prod, t=0, spot=None, step=None, status: StatusType = None):
        """求t时刻，价格spot处的delta值
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
            status: StatusType，期权是否已敲入
        Returns: delta = ∂V/∂S
        """
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        delta = (f(min(spot + step, self.smax)) - f(min(spot - step, self.smax))) / (2 * step)
        return delta

    def gamma(self, prod, t=0, spot=None, step=None, status: StatusType = None):
        """求t时刻，价格spot处的gamma值
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为价格spot的1%
            status: StatusType，期权是否已敲入
        Returns: gamma = ∂2V/∂S2
        """
        spot = self.process.spot() if spot is None else spot
        step = spot * 0.01 if step is None else step
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = interp1d(fdm_grid.s_vec, fdm_grid.v_grid[1:-1, t], kind='cubic')
        gamma = (f(spot + step) - 2 * f(spot) + f(spot - step)) / (step ** 2)
        return gamma

    def theta(self, prod, t=0, spot=None, step=1, status: StatusType = None):
        """计算一天的theta值
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: int，时间步长，默认为1天
            status: StatusType，期权是否已敲入
        Returns: theta = ∂V/∂t
        """
        spot = self.process.spot() if spot is None else spot
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        g = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t + step], kind='cubic')
        theta = (g(spot) - f(spot)) / (step * spot)  # ( step * self.dt)
        return theta

    def vega(self, prod, t=0, spot=None, step=0.01, status: StatusType = None):
        """计算波动率上升1%的vega todo:目前只支持常数波动率
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
            status: StatusType，期权是否已敲入
        Returns: vega = ∂V/∂σ
        """
        spot = self.process.spot() if spot is None else spot
        status = self.prod.status if status is None else status
        last_vol = self.process.vol.volval
        last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()
        fdm_grid0 = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.vol.volval = last_vol + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        vega = (g(spot) - f(spot)) / spot
        self.process.vol.volval = last_vol
        self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid
        return vega

    def rho(self, prod, t=0, spot=None, step=0.01, status: StatusType = None):
        """计算无风险利率上升1%的rho todo:目前只支持常数无风险利率
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            step: float，微扰步长，默认为1%
            status: StatusType，期权是否已敲入
        Returns: rho = ∂V/∂r
        """
        spot = self.process.spot() if spot is None else spot
        status = self.prod.status if status is None else status
        last_r = self.process.interest.data
        last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()
        fdm_grid0 = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = fdm_grid0.functionize(fdm_grid0.v_grid[1:-1, t], kind='cubic')
        self.process.interest.data = last_r + step
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        g = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        rho = (g(spot) - f(spot)) / spot
        self.process.interest.data = last_r
        self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid
        return rho

    def pv_and_greeks(self, prod, t=0, spot=None, status: StatusType = None):
        """当一次性计算pv和5个greeks时，可以调用此函数，减少计算量
        Args:
            t: int，期权价值矩阵的列(时间)的索引
            spot: float，价格的绝对值
            status: StatusType，期权是否已敲入
        Returns:
            Dict[str:float]: {'pv': pv, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
        """
        spot = self.process.spot() if spot is None else spot
        status = self.prod.status if status is None else status
        s_step = spot * 0.01
        fdm_grid = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        f = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t], kind='cubic')
        pv = np.float64(f(spot))
        delta = (f(min(spot + s_step, self.smax)) - f(min(spot - s_step, self.smax))) / (2 * s_step)
        gamma = (f(spot + s_step) - 2 * f(spot) + f(spot - s_step)) / (s_step ** 2)

        g = fdm_grid.functionize(fdm_grid.v_grid[1:-1, t + 1], kind='cubic')
        theta = (g(spot) - f(spot)) / (1 * spot)
        with suppress(AttributeError):
            last_in_grid = self.fd_knockin.v_grid.copy()
        last_not_grid = self.fd_not_in.v_grid.copy()

        last_vol = self.process.vol.volval
        self.process.vol.volval = last_vol + 0.01
        self.calc_present_value(prod=prod)
        fdm_grid1 = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        g1 = fdm_grid1.functionize(fdm_grid1.v_grid[1:-1, t], kind='cubic')
        vega = (g1(spot) - f(spot)) / spot
        self.process.vol.volval = last_vol

        last_r = self.process.interest.data
        self.process.interest.data = last_r + 0.01
        self.calc_present_value(prod=prod)
        fdm_grid2 = self.fd_knockin if (status != StatusType.NoTouch) else self.fd_not_in
        g2 = fdm_grid2.functionize(fdm_grid2.v_grid[1:-1, t], kind='cubic')
        rho = (g2(spot) - f(spot)) / spot
        self.process.interest.data = last_r
        with suppress(AttributeError, UnboundLocalError):
            self.fd_knockin.v_grid = last_in_grid
        self.fd_not_in.v_grid = last_not_grid

        return {'pv': pv, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def value_matrix(self, status: StatusType = None):
        """根据是否敲入状态，返回S-t矩阵
        Args:
            status: StatusType，期权是否已敲入
        Returns:
            v_grid: np.ndarray，S-t矩阵
            s_vec: np.ndarray，S轴格点
        """
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        return fdm_grid.v_grid, fdm_grid.s_vec

    def delta_matrix(self, step=0, status: StatusType = None):
        """计算S-t矩阵中，每个t时刻所有s格点的delta
        Args:
            step: float, 标的价步长，默认为0，即选择PDE有限差分网格的价格步长
            status: StatusType，期权是否已敲入
        Returns:
            delta_s_t: np.ndarray，Delta-t矩阵
            s_vec: np.ndarray，S轴格点
        """
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
        Args:
            step: float, 标的价步长，默认为0，即选择PDE有限差分网格的价格步长
            status: StatusType，期权是否已敲入
        Returns:
            gamma_s_t: np.ndarray，Gamma-t矩阵
            s_vec: np.ndarray，S轴格点
        """
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
        """计算S-t矩阵中，每个s价格所有t时刻的theta
        Args:
            step: int, 时间步长，默认为0，即选择PDE有限差分网格的时间步长
            status: StatusType，期权是否已敲入
        Returns:
            theta_s_t: np.ndarray，Theta-t矩阵
            s_vec: np.ndarray，S轴格点
        """
        status = self.prod.status if status is None else status
        fdm_grid = self.fd_knockin if (status == StatusType.DownTouch) else self.fd_not_in
        if step == 0:
            # 默认PDE模拟时的步长
            spot = self.process.spot()
            theta_s_t = np.diff(fdm_grid.v_grid, axis=1) / spot  # / self.dt
            x_vec = np.insert(fdm_grid.s_vec, 0, 0)
            x_vec = np.append(x_vec, self.smax)
            return theta_s_t, x_vec
        else:  # 重新定义步长，在模拟矩阵的基础上插值
            raise NotImplementedError("todo: 时间步长与pde有限差分网格不同的theta.")
