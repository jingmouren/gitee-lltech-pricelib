#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@File : snowball_greeks_plot.py
@author : MRX
@email : marx@galatech.com.cn 
@Time : 2024/05/10 下午9:12
"""
import datetime
from pricelib import *
from pricelib.common.utilities import pde_plot


def plot_greeks(pde_engine, barrier_in, spot):
    # 绘制delta和gamma曲面图
    delta_matrix, s_vec = pde_engine.delta_matrix(status=StatusType.NoTouch)
    fig1 = pde_plot.draw_greeks_surface("delta", delta_matrix, spot_range=(65, 115), show_plot=True)
    gamma_matrix, s_vec = pde_engine.gamma_matrix(status=StatusType.NoTouch)
    fig2 = pde_plot.draw_greeks_surface("gamma", gamma_matrix, show_plot=True)
    # 绘制t=0时的delta和gamma曲线
    fig3 = pde_plot.draw_greeks_curve(delta_matrix, gamma_matrix, t=0, spot_range=(65, 115), show_plot=True)
    # 绘制delta上下限的等高线
    pde_plot.draw_greeks_contour(delta_matrix, barrier_in=barrier_in * pde_engine.s_step / (pde_engine.n_smax * spot),
                                 high_value=1.2, low_value=0.01)
    return fig1, fig2, fig3


def run():
    start_date = datetime.date(2022, 1, 5)
    set_evaluation_date(start_date)
    engine = FdmSnowBallEngine(s_step=200, n_smax=2, s=100, r=0.02, q=0.01, vol=0.17)
    option = StandardSnowball(maturity=1, start_date=start_date, lock_term=3, s0=100, barrier_out=103, barrier_in=80,
                              coupon_out=0.107, engine=engine)
    option.price()
    plot_greeks(engine, 80, 100)


if __name__ == '__main__':
    run()
