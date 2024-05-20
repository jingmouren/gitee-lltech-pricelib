#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from scipy.stats import norm
from pricelib.common.utilities.enums import UpDown, BarrierType, PaymentType, InOut
from pricelib.common.time import global_evaluation_date
from pricelib.common.pricing_engine_base import AnalyticEngine


class AnalyticBarrierEngine(AnalyticEngine):
    """障碍期权闭式解定价引擎
    敲入期权现在支付期权费，但是当到期前资产价格触及障碍水平时，期权才生效。若一直没有发生敲入，则到期时支付现金返还rebate
    敲出期权现在支付期权费，但是当到期前资产价格触及障碍水平时，期权就失效了。若到期前发生敲出事件，则立刻支付现金返还rebate
    Merton(1973), Reiner & Rubinstein(1991a)提出障碍期权解析解，
    Broadie, Glasserman和Kou(1995)提出均匀离散观察障碍期权近似解"""

    # pylint: disable=invalid-name, too-many-locals, missing-docstring, too-many-branches, too-many-statements
    def calc_present_value(self, prod, t=None, spot=None):
        """计算现值
        Args:
            prod: Product产品对象
            t: datetime.date，估值日; 如果是None，则使用全局估值日globalEvaluationDate
            spot: float，估值日标的价格，如果是None，则使用随机过程的当前价格
        Returns: float，现值
        """
        if prod.inout == InOut.In:
            assert prod.payment_type == PaymentType.Expire, "ValueError: 敲入期权解析解，一直未敲入，到期时支付现金返还。payment_type应该是Expire"
        if prod.inout == InOut.Out:
            assert prod.payment_type == PaymentType.Hit, "ValueError: 敲出期权解析解，发生敲出，立刻支付现金返还。payment_type应该是Hit"
        calculate_date = global_evaluation_date() if t is None else t
        tau = prod.trade_calendar.business_days_between(calculate_date, prod.end_date) / prod.t_step_per_year
        if spot is None:
            spot = self.process.spot()

        r = self.process.interest(tau)
        q = self.process.div(tau)
        vol = self.process.vol(tau, self.process.spot())
        drift = r - q
        if prod.discrete_obs_interval is not None:
            # 均匀离散观察障碍期权，M. Broadie, P. Glasserman, S.G. Kou(1997) 在连续障碍期权解析解上加调整项，调整障碍价格水平
            # 指数上的beta = -zeta(1/2) / sqrt(2*pi) = 0.5826, 其中zeta是黎曼zeta函数
            if prod.updown == UpDown.Up:  # 向上
                barrier = prod.barrier * np.exp(0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            elif prod.updown == UpDown.Down:  # 向下
                barrier = prod.barrier * np.exp(-0.5826 * vol * np.sqrt(prod.discrete_obs_interval))
            else:
                raise ValueError("self.updown 只能是Up或Down")
        else:  # 连续观察障碍期权
            barrier = prod.barrier

        mu = drift / vol ** 2 - 0.5
        la = np.sqrt(mu ** 2 + 2 * r / vol ** 2)
        a = (barrier / spot) ** (2 * mu)
        b = (barrier / spot) ** (2 * mu + 2)
        c = (barrier / spot) ** (mu + la)
        d = (barrier / spot) ** (mu - la)
        a1 = np.log(spot / prod.strike)
        a2 = np.log(spot / barrier)
        a3 = np.log(spot * prod.strike / barrier ** 2)
        a4 = drift + 0.5 * vol ** 2
        a5 = drift - 0.5 * vol ** 2
        a6 = tau
        a7 = vol * np.sqrt(a6)
        d1 = (a1 + a4 * a6) / a7
        d2 = (a1 + a5 * a6) / a7
        d3 = (a2 + a4 * a6) / a7
        d4 = (a2 + a5 * a6) / a7
        d5 = (a2 - a5 * a6) / a7
        d6 = (a2 - a4 * a6) / a7
        d7 = (a3 - a5 * a6) / a7
        d8 = (a3 - a4 * a6) / a7
        d9 = -a2 / a7 + la * a7
        d10 = -a2 / a7 - la * a7

        def A(phi):
            return phi * spot * np.exp(-q * a6) * norm.cdf(phi * d1) - \
                phi * prod.strike * np.exp(-r * a6) * norm.cdf(phi * d2)

        def B(phi):
            return phi * spot * np.exp(-q * a6) * norm.cdf(phi * d3) - \
                phi * prod.strike * np.exp(-r * a6) * norm.cdf(phi * d4)

        def C(phi, eta):
            return phi * spot * np.exp(-q * a6) * b * norm.cdf(-eta * d8) - \
                phi * prod.strike * np.exp(-r * a6) * a * norm.cdf(-eta * d7)

        def D(phi, eta):
            return phi * spot * np.exp(-q * a6) * b * norm.cdf(-eta * d6) - \
                phi * prod.strike * np.exp(-r * a6) * a * norm.cdf(-eta * d5)

        def E(eta):
            return prod.rebate * np.exp(-r * a6) * (norm.cdf(eta * d4) - a * norm.cdf(-eta * d5))

        def F(eta):
            return prod.rebate * (c * norm.cdf(eta * d9) + d * norm.cdf(eta * d10))

        if prod.barrier_type == BarrierType.UIC:  # 向上敲入看涨
            phi = 1
            eta = -1
            if prod.strike >= barrier:
                price = A(phi) + E(eta)
            else:
                price = (B(phi) - C(phi, eta) + D(phi, eta)) + E(eta)
        elif prod.barrier_type == BarrierType.UIP:  # 向上敲入看跌
            phi = -1
            eta = -1
            if prod.strike >= barrier:
                price = (A(phi) - B(phi) + D(phi, eta)) + E(eta)
            else:
                price = C(phi, eta) + E(eta)
        elif prod.barrier_type == BarrierType.UOC:  # 向上敲出看涨
            phi = 1
            eta = -1
            if prod.strike >= barrier:
                price = F(eta)
            else:
                price = (A(phi) - B(phi) + C(phi, eta) - D(phi, eta)) + F(eta)
        elif prod.barrier_type == BarrierType.UOP:  # 向上敲出看跌
            phi = -1
            eta = -1
            if prod.strike >= barrier:
                price = (B(phi) - D(phi, eta)) + F(eta)
            else:
                price = (A(phi) - C(phi, eta)) + F(eta)
        elif prod.barrier_type == BarrierType.DIC:  # 向下敲入看涨
            phi = 1
            eta = 1
            if prod.strike >= barrier:
                price = C(phi, eta) + E(eta)
            else:
                price = (A(phi) - B(phi) + D(phi, eta)) + E(eta)
        elif prod.barrier_type == BarrierType.DIP:  # 向下敲入看跌
            phi = -1
            eta = 1
            if prod.strike >= barrier:
                price = (B(phi) - C(phi, eta) + D(phi, eta)) + E(eta)
            else:
                price = A(phi) + E(eta)
        elif prod.barrier_type == BarrierType.DOC:  # 向下敲出看涨
            phi = 1

            eta = 1
            if prod.strike >= barrier:
                price = (A(phi) - C(phi, eta)) + F(eta)
            else:
                price = (B(phi) - D(phi, eta)) + F(eta)
        elif prod.barrier_type == BarrierType.DOP:  # 向下敲出看跌
            phi = -1
            eta = 1
            if prod.strike >= barrier:
                price = (A(phi) - B(phi) + C(phi, eta) - D(phi, eta)) + F(eta)
            else:
                price = F(eta)
        else:
            raise ValueError("self.barrier_type 障碍类型错误")
        return price * prod.parti
