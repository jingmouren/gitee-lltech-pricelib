#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from pricelib.common.utilities.enums import TouchType, PaymentType, ExerciseType
from pricelib.common.pricing_engine_base import FdmEngine, FdmGridwithBound, FdmGrid
from pricelib.common.time import global_evaluation_date


class FdmDigitalEngine(FdmEngine):
    """二元(数字)期权PDE有限差分法定价引擎，整个存续期观察
    支持结构:
        单边二元: 欧式二元/美式二元/一触即付
        双边二元:
            欧式 - 二元凹式/二元凸式
            美式 - 美式双接触/美式双不接触
    """

    def __init__(self, stoch_process=None, s_step=800, n_smax=2, fdm_theta=1, *,
                 s=None, r=None, q=None, vol=None):
        """构造函数
        Args:
            stoch_process: StochProcessBase 随机过程
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
        # 以下为计算过程的中间变量
        self.prod = None  # Product产品对象
        self.fdm = None  # 有限差分网格对象，若连续观察，则为FdmGridwithBound；若离散观察则为FdmGrid
        self.bound = None  # 低/高障碍价格绝对数值
        self.rebate = None  # 低/高障碍价格对应补偿绝对数值
        self.fn_bound = None  # 边界条件函数

    def _gen_func_bound(self, bound_flag, rebate, maturity):
        """生成边界条件函数
        Args:
            bound_flag: 0或1，0代表边界上的期权价值是0，1代表边界上的期权价值是rebate
            rebate: float，二元期权的行权收益
            maturity: float，到期时间
        Returns:
            func_bound: function，边界条件函数，输入为年化时间，返回为期权价值
        """
        if self.prod.payment_type == PaymentType.Hit:  # 触碰障碍立即支付
            def func_bound(u):
                return bound_flag * rebate
        else:  # 'PaymentMethod.Expire' 触碰障碍后，到期时支付
            def func_bound(u):
                return bound_flag * rebate * self.process.interest.disc_factor(maturity, u)
        return func_bound

    def _init_boundary_condition(self, maturity):
        """
        生成障碍价格的Dirichlet边界条件： 第u天触碰障碍产生的payoff，折现到第u天的价值。
        返回一个包含两个lambda函数的列表，列表索引分别为[0, 1]，对应低障碍价和高障碍价。
        Args:
            maturity: float, 到期时间
        Returns: None
        """
        self.fn_bound = []
        flag_list = [1, 1]
        for i, rebate in enumerate(self.rebate):
            if self.bound[i] in [-np.inf, np.inf]:  # 障碍价无穷，即障碍价没有设置
                flag_list[i] *= -1  # 反转
            if self.prod.touch_type == TouchType.NoTouch:
                flag_list[i] *= -1  # 反转
            bound_flag = (1 + flag_list[i]) * 0.5  # 将±1转化为0或1
            self.fn_bound.append(self._gen_func_bound(bound_flag, rebate, maturity))

    def _init_terminal_condition(self, s_vec):
        """初始化终止条件
        Args:
            s_vec: List[float], 标的价格向量
        Returns:
            yv: List[float], 终止条件 - 期权价值向量
        """
        # 美式, 且连续观察
        if self.prod.exercise_type == ExerciseType.American and self.prod.discrete_obs_interval is None:
            # 连续观察，网格边界是障碍价，留到最后一列的都是一直没碰到边界的payoff
            if self.prod.touch_type == TouchType.Touch:
                yv = s_vec * 0
            else:  # NoTouch
                yv = s_vec * 0 + self.rebate[0]
            return yv
        if self.prod.exercise_type == ExerciseType.European or (  # 欧式或离散观察美式
                self.prod.exercise_type == ExerciseType.American and self.prod.discrete_obs_interval is not None):
            lower = s_vec <= self.bound[0]  # 小于低障碍价的标的价格bool索引
            upper = s_vec >= self.bound[1]  # 大于高障碍价的标的价格bool索引
            inner = (~lower) & (~upper)  # 大于低障碍价且小于于高障碍价，即在两个障碍价格之间的bool索引
            if self.prod.touch_type == TouchType.Touch:
                yv = s_vec * 0
                yv[lower] = self.rebate[0]
                yv[upper] = self.rebate[1]
            else:  # NoTouch
                yv = s_vec * 0
                yv[inner] = self.rebate[0]
            return yv

        raise ValueError(f"{self.prod.exercise_type}必须是欧式或美式")

    def calc_present_value(self, prod, t=None, spot=None, bound=(None, None), rebate=(0, 0)):
        """计算现值，PDE有限差分定价, 时间逆向迭代
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
            bound: (lower_barrier, upper_barrier), 低障碍价/高障碍价，障碍价格绝对数值。如果是默认值None，即为无穷大或无穷小inf。
            rebate: (lower_rebate, upper_rebate), 对敲出期权，元组第一位是低障碍价下方的补偿，第二位是高障碍价上方的补偿，
                                                        对敲入期权，元组第一位是敲入补偿，第二位没有用处。
        Returns: float，现值
        """
        self.prod = prod
        self.bound = np.array([(2 * i - 1) * np.inf if b is None else b for i, b in enumerate(bound)])  # 障碍价格区间
        self.rebate = np.array(rebate)  # 补偿回报

        calculate_date = global_evaluation_date() if t is None else t
        # 剩余到期时间，交易日
        _maturity = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()
        self.smax = self.n_smax * spot  # 价格网格上界, n_smax倍初始价格
        # 返回PDE系数组件abc的函数
        fn_pde_coef = self.process.get_fn_pde_coef(_maturity, spot)

        if prod.exercise_type == ExerciseType.American:
            if prod.discrete_obs_interval is None:  # 连续观察美式
                self.fdm = FdmGridwithBound(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                            fn_pde_coef=fn_pde_coef, bound=self.bound, fdm_theta=self.fdm_theta)
                self._init_boundary_condition(_maturity)
                self.fdm.set_boundary_condition(self.fn_bound)

                # 初始化到期日payoff价值
                yv = self._init_terminal_condition(self.fdm.s_vec)
                # 从后向前逆推
                yv = self.fdm.evolve_with_interval(_maturity, 0, yv)

            else:  # 离散观察美式
                self.fdm = FdmGrid(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                                   fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta, maturity=_maturity)
                self._init_boundary_condition(_maturity)
                self.fdm.v_grid[0, :] = self.fn_bound[0](self.fdm.tv)
                self.fdm.v_grid[-1, :] = self.fn_bound[1](self.fdm.tv)

                lower = self.fdm.s_vec <= self.bound[0]  # 小于低障碍价的标的价格bool索引
                upper = self.fdm.s_vec >= self.bound[1]  # 大于高障碍价的标的价格bool索引
                t_step = round(prod.t_step_per_year * _maturity)  # 时间步数
                dt = 1 / prod.t_step_per_year
                obs_points = np.flip(np.round(prod.t_step_per_year *
                                              np.arange(_maturity, 0, -prod.discrete_obs_interval)).astype(int))
                obs_points = np.concatenate((np.array([0]), obs_points))

                yv = self._init_terminal_condition(self.fdm.s_vec)
                self.fdm.v_grid[1:-1, -1] = yv

                for j in range(t_step - 1, -1, -1):
                    yv = self.fdm.evolve(j, yv, dt)  # 每组(p, q)更新
                    if j in obs_points:
                        if prod.touch_type == TouchType.Touch:
                            if prod.payment_type == PaymentType.Expire:  # 到期支付
                                yv[lower] = self.rebate[0] * self.process.interest.disc_factor(_maturity,
                                                                                            j / t_step * _maturity)
                                yv[upper] = self.rebate[1] * self.process.interest.disc_factor(_maturity,
                                                                                            j / t_step * _maturity)
                            else:  # 立即支付
                                yv[lower] = self.rebate[0]
                                yv[upper] = self.rebate[1]
                        else:
                            yv[lower] = 0
                            yv[upper] = 0
                    self.fdm.v_grid[1:-1, j] = yv
        elif prod.exercise_type == ExerciseType.European:  # 欧式
            self.fdm = FdmGrid(smax=self.smax, t_step_per_year=prod.t_step_per_year, s_step=self.s_step,
                               fn_pde_coef=fn_pde_coef, fdm_theta=self.fdm_theta, maturity=_maturity)
            self._init_boundary_condition(_maturity)
            self.fdm.v_grid[0, :] = self.fn_bound[0](self.fdm.tv)
            self.fdm.v_grid[-1, :] = self.fn_bound[1](self.fdm.tv)

            t_step = round(prod.t_step_per_year * _maturity)  # 时间步数
            dt = 1 / prod.t_step_per_year
            yv = self._init_terminal_condition(self.fdm.s_vec)
            self.fdm.v_grid[1:-1, -1] = yv

            for j in range(t_step - 1, -1, -1):
                yv = self.fdm.evolve(j, yv, dt)  # 每组(p, q)更新
                self.fdm.v_grid[1:-1, j] = yv
        else:
            raise ValueError(f"{prod.exercise_type}必须是欧式或美式")
        return np.float64(self.fdm.functionize(yv, kind="cubic")(spot))
