#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import importlib_metadata
from .pricing_engines import *
from .products import *
from .common import *
from .common.utilities import SimpleQuote

__author__ = '上海凌瓴信息科技有限公司: 马瑞祥, 夏鸿翔, 贡献: 张鹏任, 张峻蔚'
__email__ = 'marx@galatech.com.cn'

try:
    __version__ = importlib_metadata.version("pricelib")
except importlib_metadata.PackageNotFoundError:
    __version__ = "dev"
