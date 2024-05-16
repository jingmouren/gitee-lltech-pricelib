#!/user/bin/env python
# -*- coding: utf-8 -*-
""" 生成标的价格二叉树 示例
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(f"ModuleNotFoundError: 请安装matplotlib>=3.5.3, <=3.7.5")
    exit(0)

from pricelib.pricing_engines.tree_engines import BiTreeVanillaEngine
from pricelib import GeneralizedBSMProcess, SimpleQuote, BlackConstVol


def run_tree_paths_plot():
    spot_price = SimpleQuote(value=100, name="中证1000指数")
    riskfree = SimpleQuote(value=0.03, name="无风险利率")
    dividend = SimpleQuote(value=0.05, name="中证1000贴水率")
    volatility = BlackConstVol(0.16, name="中证1000波动率")

    process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, div=dividend, vol=volatility)

    engine = BiTreeVanillaEngine(process, tree_branches=500)
    n_step = int(0.016 * engine.tree_branches)
    path = engine.path_generator(n_step)

    fig, ax = plt.subplots()
    for i in range(path.shape[1]):
        for j in range(i + 1):
            plt.plot(i, j, 'o')
            plt.text(i, j, '{:.2f}'.format(path[j, i]))

    # 设置图形属性
    plt.title('Binomial Tree of Stock Prices')
    plt.xlabel('Period')
    plt.ylabel('Number of Downward Movements')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_tree_paths_plot()
