#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 蒙特卡洛模拟BSM、Heston路径
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(f"ModuleNotFoundError: 请安装matplotlib>=3.5.3, <=3.7.5")
    exit(0)

from pricelib import (GeneralizedBSMProcess, HestonProcess, MCVanillaEngine, SimpleQuote, BlackConstVol,
                      RandsMethod, LdMethod)


def run_mc_paths_plot():
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = SimpleQuote(value=0.02, name="无风险利率")
    dividend = SimpleQuote(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")

    # 1. BSM
    bsm_process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    engine = MCVanillaEngine(bsm_process, n_path=500, rands_method=RandsMethod.Pseudorandom,
                             antithetic_variate=True, ld_method=LdMethod.Sobol, seed=0)

    engine.t_step_per_year = 243
    n_step = int(0.5 * engine.t_step_per_year)

    bsm_path = engine.path_generator(n_step, spot_price.data)

    plt.plot(bsm_path)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.title('BSM Price Simulation')
    plt.grid(True)
    plt.show()

    # 2.Heston
    var_theta = 0.0226
    heston_process = HestonProcess(spot=spot_price, interest=riskfree, div=dividend, v0=0.0251, var_theta=var_theta,
                                   var_kappa=4.01, var_vol=0.302, var_rho=-0.293)
    engine.set_stoch_process(heston_process)

    heston_path = engine.path_generator(n_step, spot_price.data)

    plt.plot(heston_path)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.title('Heston Price Simulation')
    plt.grid(True)
    plt.show()

    var_path = engine.var_paths
    plt.plot(var_path)
    plt.plot([var_theta] * var_path.shape[0], color='black')
    plt.xlabel('time')
    plt.ylabel('vol')
    plt.title('Heston Vol Simulation')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_mc_paths_plot()
