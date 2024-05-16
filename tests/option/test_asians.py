#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.asian import asian_demo


def test_asian_demo():
    expected_data = [
        ['算术平均-代替标的结算价-Call亚式期权', 2.893737891784113, 2.9059603925765556, 2.9024890578353175],
        ['算术平均-代替标的结算价-Call增强型亚式期权', np.nan, 3.453030319689411, np.nan],
        ['算术平均-代替标的结算价-Put亚式期权', 4.34944253565525, 4.351327562759141, 4.358121279660046],
        ['算术平均-代替标的结算价-Put增强型亚式期权', np.nan, 4.898397489871997, np.nan],
        ['几何平均-代替标的结算价-Call亚式期权', 2.8091532462766367, 2.82014461530082, np.nan],
        ['几何平均-代替标的结算价-Call增强型亚式期权', np.nan, 3.3777986368091546, np.nan],
        ['几何平均-代替标的结算价-Put亚式期权', 4.474255200621862, 4.476119967804266, np.nan],
        ['几何平均-代替标的结算价-Put增强型亚式期权', np.nan, 5.008070041295551, np.nan]]
    expected_df = pd.DataFrame(expected_data, columns=['类型', '闭式解', 'MonteCarlo', '二叉树'])
    res_df = asian_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)
