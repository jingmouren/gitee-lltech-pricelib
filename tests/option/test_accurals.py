#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from examples.accurals import accumulator_demo, range_accural_demo


def test_accumulator_demo():
    expected_data = [['累购(Accumulator)', 0.07070529486371592], ['累沽(Decumulator)', 0.1020375017596659]]
    expected_df = pd.DataFrame(expected_data, columns=['期权类型', 'MonteCarlo'])
    res_df = accumulator_demo.run()
    assert_frame_equal(res_df, expected_df, check_dtype=True)


def test_range_accural_demo():
    res = range_accural_demo.run()
    assert res == pytest.approx({'结构': '区间累计(Range Accural)', 'MonteCarlo': 10.164338955994404}, rel=1e-10)
