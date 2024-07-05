#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import importlib_metadata
import importlib
from .pricing_engines import *
from .products import *
from .common import *

__author__ = '上海凌瓴信息科技有限公司: 马瑞祥, 夏鸿翔, 贡献: 张鹏任, 张峻尉'
__email__ = 'marx@galatech.com.cn'

try:  # 获取当前安装的包版本
    __version__ = importlib_metadata.version("pricelib")
except importlib_metadata.PackageNotFoundError:
    __version__ = "dev"


def check_for_updates(package_name):
    """检查PyPI是否有新版本的包可用"""
    try:  # 从PyPI获取最新版本信息
        requests = importlib.import_module("requests")
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        response.raise_for_status()
        data = response.json()
        latest_version = data['info']['version']
        if latest_version > __version__:
            logging.warning(f"A new version of {package_name} is available: {latest_version} "
                            f"(installed version: {__version__}),\nPlease update your package: "
                            f"pip install --upgrade {package_name} -i https://pypi.org/simple\n"
                            f"-------------------------------------------------------------------------------------")
        elif __version__ == "dev":
            logging.info(f"You are using a development version of {package_name}.")
        else:
            logging.info(f"You are using the latest version of {package_name} ({__version__}).")
    except Exception as e:
        logging.error(f"Failed to check for updates: {e},\nPlease check your internet connection and make sure "
                      f"you've already installed 'requests': pip install requests\n"
                      f"---------------------------------------------------------------------------------------")


# 当包被导入时自动检查更新
check_for_updates('pricelib')
