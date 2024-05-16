#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.digital import digital_option_demo, double_touch_notouch_demo


def test_digital_demo():
    expected_discrete = [
        ['离散观察欧式Call二元期权', 0.7813384345830714, 0.7800421042175157, 0.7912396180275464, 0.812813152699443,
         0.7891833475907255],
        ['离散观察欧式Put二元期权', 1.2726911914338912, 1.2630840104230847, 1.2674696024255445, 1.2895615455303617,
         1.2913388609349215],
        ['离散观察美式Call二元期权', 1.6766870002507048, 1.6744733936099303, 1.69633048284921, 1.6688088109426866,
         1.670733761754569],
        ['离散观察美式Put二元期权', 2.1376442967454254, 2.134676670727452, 2.125110746564379, 2.0890054574578123,
         2.1428078891800553],
        ['离散观察美式Call二元期权', 1.6909576756985945, 1.688528407429802, 1.7107735700808155, 1.6828563953693552,
         1.6849260212121293],
        ['离散观察美式Put二元期权', 2.1530488132491263, 2.1497975527515845, 2.1403179127696594, 2.1037244495819074,
         2.1582634329368275]]
    expected_continuous = [['连续观察欧式Call二元期权', 0.7813384345830714, 0.789183],
                           ['连续观察欧式Put二元期权', 1.2726911914338912, 1.291339],
                           ['连续观察美式Call二元期权', 1.8021622711412713, 1.799008],
                           ['连续观察美式Put二元期权', 2.269883592086009, 2.263541],
                           ['连续观察美式Call二元期权', 1.817889647827623, 1.814803],
                           ['连续观察美式Put二元期权', 2.286636690933101, 2.280384]]
    expected_discrete_df = pd.DataFrame(expected_discrete, columns=['期权类型    ', '闭式解(每日观察)', 'MonteCarlo',
                                                                    'Quadrature', '二叉树', 'PDE'])
    expected_continuous_df = pd.DataFrame(expected_continuous, columns=['期权类型    ', '闭式解(连续观察)', 'PDE'])
    df1, df2 = digital_option_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)


def test_double_touch_notouch_demo():
    expected_discrete = [['到期支付二元凹式(欧式双接触期权)', np.nan, 3.084881, 3.123992],
                         ['到期支付二元凸式(欧式双不接触期权)', np.nan, 6.717105, 6.678003],
                         ['到期支付离散观察美式双触碰二元期权', 5.766874556177397, 5.759843, 5.748472],
                         ['到期支付离散观察美式双不触碰二元期权', 4.035112176890156, 4.042143, 4.05352],
                         ['立即支付离散观察美式双触碰二元期权', np.nan, 5.813134, 5.801601],
                         ['立即支付离散观察美式双不触碰二元期权', np.nan, 4.042143, 4.05352]]

    expected_continuous = [['到期支付二元凹式(欧式双接触期权)', np.nan, 3.123992],
                           ['到期支付二元凸式(欧式双不接触期权)', np.nan, 6.678003],
                           ['到期支付连续观察美式双触碰二元期权', 6.095226932808097, 6.083505],
                           ['到期支付连续观察美式双不触碰二元期权', 3.706759800259456, 3.718488],
                           ['立即支付连续观察美式双触碰二元期权', np.nan, 6.141737],
                           ['立即支付连续观察美式双不触碰二元期权', np.nan, 3.718488]]

    expected_discrete_df = pd.DataFrame(expected_discrete,
                                        columns=['期权类型    ', '闭式解(每日观察)', 'MonteCarlo', 'PDE'])
    expected_continuous_df = pd.DataFrame(expected_continuous, columns=['期权类型    ', '闭式解(连续观察)', 'PDE'])
    df1, df2 = double_touch_notouch_demo.run()
    assert_frame_equal(df1, expected_discrete_df, check_dtype=True)
    assert_frame_equal(df2, expected_continuous_df, check_dtype=True)
