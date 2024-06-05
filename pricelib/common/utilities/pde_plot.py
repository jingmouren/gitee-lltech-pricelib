#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
from .utility import logging

try:
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
except ModuleNotFoundError as e:
    logging.error(e)
    logging.warning('请安装plotly和matplotlib库,\nmatplotlib>=3.5.3, <=3.7.5;\nplotly>=5.16.1, <=5.22.0')
    exit(0)


def draw_greeks_surface(greeks_name, greeks_matrix, spot_range=None, show_plot=True):
    """绘制greeks曲面, spot_range是价格索引的区间
    Args:
        greeks_name: str, greeks名称, 如delta, gamma, theta等
        greeks_matrix: np.ndarray, 希腊值二维数组，行代表价格，列代表时间
        spot_range: tuple(int)=(spot_min, spot_max), 价格索引的区间, 默认为None, 即绘制全部价格范围的图像
        show_plot: bool, 是否显示图像, 默认为True
    Returns:
        fig: plotly.graph_objects.Figure对象，希腊值三维图
    """
    price_grid = np.array(range(1, 1 + greeks_matrix.shape[0]))
    if spot_range is not None:
        spot_min, spot_max = spot_range
        price_grid = price_grid[spot_min:spot_max + 1]
        greeks_matrix = greeks_matrix[spot_min:spot_max + 1, ]
    time_grid = np.array(range(greeks_matrix.shape[1]))
    fig = go.Figure(
        data=[go.Surface(z=greeks_matrix, x=time_grid, y=price_grid, colorscale='Turbo')], )  # Jet,Turbo, Rainbow
    # 轮廓线(可选)
    fig.update_traces(contours_z={  # 轮廓设置
        "show": True,  # 开启是否显示
        "usecolormap": True,  # 颜色设置
        "highlightcolor": "red",  # 高亮 mistyrose
        "project_z": True})
    # 设置标题、图片大小
    fig.update_layout(
        title=f'{greeks_name}-S-t surface',
        width=800, height=900,
        autosize=True,
        scene={
            'xaxis': {'title': '时间格点'},
            'yaxis': {'title': '标的价格格点'},
            'zaxis': {'title': greeks_name},
        },
        # template="plotly_dark",  # 深色主题
        margin={'l': 65, 'r': 50, 'b': 65, 't': 90},
        # margin=dict(r=0, l=0, t=0, b=0, pad=0)
    )
    if show_plot:
        # 将图保存到本地
        # pyo.plot(fig, filename=f'./greeks_surface_plot.html')
        fig.show()  # 显示图片(plotly在线)
    return fig


def draw_greeks_curve(delta_matrix, gamma_matrix=None, t=0, spot_range=None, show_plot=True):
    """绘制t时刻的greeks曲线
    Args:
        delta_matrix: np.ndarray, delta二维数组，行代表价格，列代表时间
        gamma_matrix: np.ndarray, gamma二维数组，行代表价格，列代表时间，默认为None，此时只绘制delta曲线
        t: int, 相对于起始日的时间索引
        spot_range: tuple(int)=(spot_min, spot_max), 价格索引的区间, 默认为None, 即绘制全部价格范围的图像
        show_plot: bool, 是否显示图像, 默认为True
    Returns:
        fig: plotly.graph_objects.Figure对象，希腊值曲线图
    """
    fig = go.Figure()
    price_grid = np.array(range(1, 1 + delta_matrix.shape[0]))
    if spot_range is not None:
        spot_min, spot_max = spot_range
        price_grid = price_grid[spot_min:spot_max + 1]
        delta_matrix = delta_matrix[spot_min:spot_max + 1, ]
        if gamma_matrix is not None:
            gamma_matrix = gamma_matrix[spot_min:spot_max + 1, ]

    delta_line = go.Scatter(
        x=price_grid,
        y=delta_matrix[:, t],
        mode='lines',
        name="Delta"
    )
    fig.add_trace(delta_line)
    if gamma_matrix is not None:
        gamma_line = go.Scatter(
            x=price_grid,
            y=gamma_matrix[:, t],
            mode='lines',
            name="Gamma"
        )
        fig.add_trace(gamma_line)
    fig.update_layout(title='Greeks', xaxis_title='Underlying Price',
                      yaxis_title='Greeks Value')  # 深色主题 , template="plotly_dark",

    if show_plot:
        # pyo.plot(fig, filename=f'./greeks_curve_plot.html')
        fig.show()
    return fig


def draw_greeks_contour(greeks_matrix, barrier_in, high_value=1.2, low_value=0.01, title=None):
    """绘制greeks等高线
    Args:
        greeks_matrix: np.ndarray, 希腊值二维数组，行代表价格，列代表时间
        barrier_in: int, barrier_in是敲入价索引
        high_value: float, greeks的上限
        low_value: float, greeks的下限
        title: str, 图表标题
    Returns: None
    """
    # 绘制Delta等高线图
    high_contour = plt.contour(greeks_matrix, levels=[high_value, ])
    # 清除之前的图像
    plt.clf()
    # 获取等高线的路径点
    for collection in high_contour.collections:
        paths = collection.get_paths()
        for path in paths:
            vertices = path.vertices
            # 筛选出上半边的点（未敲入）
            upper_half_vertices = vertices[vertices[:, 1] > barrier_in]
            # 绘制上半边的点（未敲入）
            plt.plot(upper_half_vertices[:, 0], upper_half_vertices[:, 1], 'orange')
    plt.contour(greeks_matrix, levels=[low_value, ], colors='royalblue')
    plt.xlabel('Days')
    plt.ylabel('Price')
    if title is None:
        title = f'雪球理论Delta区间: {low_value}-{high_value}'
    plt.title(title)
    # 创建虚拟线条对象用于图例
    high_line = mlines.Line2D([], [], color='orange', label=f'Delta={high_value}')
    low_line = mlines.Line2D([], [], color='royalblue', label=f'Delta={low_value}')
    plt.legend(handles=[high_line, low_line])
    # 控制y轴范围
    plt.ylim(barrier_in, 120)
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体
    plt.show()
