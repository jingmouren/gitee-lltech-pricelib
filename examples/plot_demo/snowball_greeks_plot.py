#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
@File : snowball_greeks_plot.py
@author : MRX
@email : marx@galatech.com.cn 
@Time : 2024/05/10 下午9:12
"""
import datetime
import numpy as np
from pricelib import *
from pricelib.common.utilities import pde_plot


def plot_greeks(pde_engine, s0, barrier_in, barrier_out, s_step, n_smax):
    # 计算未敲入雪球的Delta网格和Gamma网格
    delta_matrix, s_vec = pde_engine.delta_matrix(status=StatusType.NoTouch)
    gamma_matrix, s_vec = pde_engine.gamma_matrix(status=StatusType.NoTouch)
    # 绘制t=0时的Delta和Gamma曲线
    spot_range = np.round(np.array([barrier_in / s0 - 0.1, barrier_out / s0 + 0.1])
                          * s_step / n_smax).astype(int)
    fig1 = pde_plot.draw_greeks_curve(delta_matrix, gamma_matrix, t=0, s_step=s_step, n_smax=n_smax,
                                      spot_range=spot_range, show_plot=True)
    # 绘制Delta和Gamma曲面图
    delta_matrix_ = delta_matrix[1:-1, 0: int(delta_matrix.shape[1] * 0.90)]
    gamma_matrix_ = gamma_matrix[1:-1, 0: int(gamma_matrix.shape[1] * 0.90)]
    fig2 = pde_plot.draw_greeks_surface("delta", delta_matrix_, s_step=s_step, n_smax=n_smax,
                                        spot_range=spot_range, show_plot=True)
    fig3 = pde_plot.draw_greeks_surface("gamma", gamma_matrix_, s_step=s_step, n_smax=n_smax,
                                        spot_range=spot_range, show_plot=True)
    # 绘制delta上下限的等高线
    pde_plot.draw_greeks_contour(delta_matrix, s0=s0, barrier_in=barrier_in, s_step=s_step, n_smax=n_smax,
                                 high_value=1.5, low_value=0.01, title=f'一年期锁3平敲雪球理论Delta区间: 0.01-1.5')
    return fig1, fig2, fig3


def run():
    start_date = datetime.date(2022, 1, 5)
    set_evaluation_date(start_date)
    s0, barrier_in, barrier_out = 100, 80, 103
    s_step, n_smax = 800, 2
    engine = FdmSnowBallEngine(s_step=s_step, n_smax=n_smax, s=s0, r=0.02, q=0.04, vol=0.16)
    option = StandardSnowball(maturity=1, start_date=start_date, lock_term=3,
                              s0=s0, barrier_out=barrier_out, barrier_in=barrier_in,
                              coupon_out=0.113, engine=engine)
    print(option.price())
    plot_greeks(engine, s0=s0, barrier_in=barrier_in, barrier_out=barrier_out, s_step=s_step, n_smax=n_smax)


if __name__ == '__main__':
    run()
