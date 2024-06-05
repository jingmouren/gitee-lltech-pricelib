#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.barrier import airbag_demo, barrier_option_demo, double_barrier_option_demo, double_shark_demo


def test_airbag_demo():
    expected_discrete = [['安全气囊', 1.7896829727130363, 1.8028831469331283, 1.7908106935325485]]

    expected_continuous = [['安全气囊', 1.6143977879423792, 1.6104801711634669]]

    expected_discrete_df = pd.DataFrame(expected_discrete,
                                        columns=['安全气囊', '闭式解(每日观察)', 'MonteCarlo', 'PDE'])
    expected_continuous_df = pd.DataFrame(expected_continuous, columns=['安全气囊', '闭式解(连续观察)', 'PDE'])
    df1, df2 = airbag_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)


def test_barrier_demo():
    expected_discrete = [[110, '离散观察向上敲出看涨期权', 0.14541706951380462, 0.1449829265364823, 0.14529938887693322,
                          0.14448855582014658, 0.14569170345643637],
                         [110, '离散观察向上敲出看跌期权', 7.003389638583823, 7.013881690900373, 6.999598266404379,
                          6.970243047865854, 7.0157816550155605],
                         [90, '离散观察向下敲出看涨期权', 5.596471515410604, 5.609916847782886, 5.596918485883809,
                          5.563814103031952, 5.599437648181393],
                         [90, '离散观察向下敲出看跌期权', 0.2157089603993949, 0.20303968340901607, 0.21605428503261587,
                          0.1921376752340182, 0.21476058332335518],
                         [110, '离散观察向上敲入看涨期权', 6.585500579649498, 6.590661212031636, 6.574722422769064,
                          6.584940982094393, 6.580718980409329],
                         [110, '离散观察向上敲入看跌期权', 1.6491389153588862, 1.6440599129752898, 1.634441362937148,
                          1.6810912768961297, 1.6319304101407335],
                         [90, '离散观察向下敲入看涨期权', 1.1344461337526983, 1.1257272907852338, 1.12310234059212,
                          1.1655900255215395, 1.126973035684375],
                         [90, '离散观察向下敲入看跌期权', 8.436819593543314, 8.454901920466648, 8.41798435921207,
                          8.458440322217628, 8.432951481832934]]

    expected_continuous = [[110, '连续观察向上敲出看涨期权', 0.11022890889242376, 0.11089923768497928],
                           [110, '连续观察向上敲出看跌期权', 6.747875496802923, 6.743321507845971],
                           [90, '连续观察向下敲出看涨期权', 5.418921340603482, 5.4150051797626135],
                           [90, '连续观察向下敲出看跌期权', 0.16783376458806742, 0.1688061731782789],
                           [110, '连续观察向上敲入看涨期权', 6.620688740270879, 6.615511446180786],
                           [110, '连续观察向上敲入看跌期权', 1.904653057139786, 1.904390557310326],
                           [90, '连续观察向下敲入看涨期权', 1.31199630855982, 1.3114055041033341],
                           [90, '连续观察向下敲入看跌期权', 8.484694789354641, 8.478905891978254]]

    expected_discrete_df = pd.DataFrame(expected_discrete,
                                        columns=['障碍价', '障碍期权', '闭式解(每日观察)', 'MonteCarlo', 'Quadrature',
                                                 '二叉树', 'PDE'])
    expected_continuous_df = pd.DataFrame(expected_continuous,
                                          columns=['障碍价', '障碍期权', '闭式解(连续观察)', 'PDE'])
    df1, df2 = barrier_option_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)


def test_double_barrier_demo():
    expected_discrete = [
        ['离散观察双边敲入障碍Call期权', 3.2838431976856626, 3.283843197685652, 3.297514053588857, 3.278144698563653],
        ['离散观察双边敲入障碍Put期权', 2.8997877229880125, 2.8997877229880107, 2.9183497754521506, 2.8974543768417216],
        ['离散观察双边敲出障碍Call期权', 0.5812314819532958, 0.5812314819533064, 0.5784203998697937,
         0.5846239999755075],
        ['离散观察双边敲出障碍Put期权', 0.9652869566509459, 0.9652869566509477, 0.9541466295346603, 0.9653143262305219]]

    expected_continuous = [['连续观察双边敲入障碍Call期权', 3.35725094672749, 3.3572509467274787, 3.353127876238339],
                           ['连续观察双边敲入障碍Put期权', 2.9998619027404256, 2.999861902740417, 2.9953922895465137],
                           ['连续观察双边敲出障碍Call期权', 0.5078237329114685, 0.5078237329114796, 0.5096408223008237],
                           ['连续观察双边敲出障碍Put期权', 0.8652127768985327, 0.8652127768985411, 0.8673764135257145]]

    expected_discrete_df = pd.DataFrame(expected_discrete,
                                        columns=['障碍期权(每日观察)', 'I&K1992', 'Haug1998', 'MonteCarlo', 'PDE'])
    expected_continuous_df = pd.DataFrame(expected_continuous,
                                          columns=['障碍期权(连续观察)', 'I&K1992', 'Haug1998', 'PDE'])
    df1, df2 = double_barrier_option_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)


def test_double_shark_demo():
    expected_discrete = [
        ['离散观察到期支付双鲨期权', 1.9487488061336946, 1.9487488061336855, 1.9525538887531206, 1.9411878870746637,
         1.9412052821642596]]

    expected_continuous = [['连续观察到期支付双鲨期权', 1.999789564076469, 1.9997895640764738, 1.9967913331194467]]

    expected_discrete_df = pd.DataFrame(expected_discrete,
                                        columns=['双鲨期权(每日观察)', 'I&K1992', 'Haug1998', 'MonteCarlo', 'PDE', 'Quad'])
    expected_continuous_df = pd.DataFrame(expected_continuous,
                                          columns=['双鲨期权(连续观察)', 'I&K1992', 'Haug1998', 'PDE'])
    df1, df2 = double_shark_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)
