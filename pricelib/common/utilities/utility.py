#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import time
from functools import wraps
import sys
import logging
import numpy as np

# 创建一个handler，用于输出到控制台
ch = logging.StreamHandler(sys.stdout)
# 创建一个NullHandler，它不会做任何处理，只是简单地忽略所有的日志消息
nh = logging.NullHandler()
# 默认只将日志输出到控制台
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',  # - %(pathname)s[line:%(lineno)d]
                    level=logging.INFO)


def set_logging_handlers(to_file=False, to_console=True, log_file: str = None):
    """更改 logging 模块的 handlers，控制是否将日志打印到屏幕上、log文件中、或者不打印任何日志
        例如，只输出到log文件，不输出到控制台：set_logging_handlers(to_file=True, to_console=False, log_file='./my_log.log')
             既不输出到log文件，也不输出到控制台: set_logging_handlers(to_file=False, to_console=False)
    Args:
        to_file: bool, 是否将日志打印到文件中
        to_console: bool, 是否将日志打印到屏幕上
        log_file: str, log文件的路径，默认为None，即默认使用"./pricelib_test.log"
    """
    # 获取root logger
    logger = logging.getLogger()
    # 清除所有handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # 设置新的handler
    if to_file:  # 输出到log文件
        # 创建一个handler，用于写入日志文件
        if log_file is not None:  # 自定义log文件路径
            fh = logging.FileHandler(log_file, encoding='utf-8')
        else:
            fh = logging.FileHandler('./pricelib_test.log', encoding='utf-8')
        logger.addHandler(fh)
    if to_console:  # 输出到控制台
        logger.addHandler(ch)
    if not to_file and not to_console:  # 不输出日志
        logger.addHandler(nh)


def time_this(fn):
    """
    报告函数执行耗时的装饰器.
    @wraps(fn)使得@time_this不改变使用装饰器原有函数的结构(如__name__, __doc__)，避免fn.__name__返回"wrapper"，而不是fn本身的名字。
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        logging.debug('开始运行%s', fn.__name__)
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        logging.info("%s执行完成，耗时%.3f秒", fn.__name__, end - start)
        return result

    return wrapper


def ascending_pairs(*args):
    """将多个时间入参*args拼接成数组并去重、排序，例如[t0, t1, t2, t3, t4] 然后打包成升序对 [(t0,t1),(t1,t2),(t2,t3),(t3,t4)]"""
    steps = np.sort(np.unique(np.hstack(args)))  # np.hstack 用于将数组按水平方向进行拼接，然后去重，排序
    return list(zip(steps[:-1], steps[1:]))


def descending_pairs(*args):
    """将多个时间入参*args接成数组并去重、排序，例如[t0, t1, t2, t3, t4] 然后打包成降序对 [(t4,t3),(t3,t2),(t2,t1),(t1,t0)]"""
    steps = np.sort(np.unique(np.hstack(args)))[::-1]
    return list(zip(steps[:-1], steps[1:]))


def descending_pairs_for_barrier(*args):
    """将多个时间入参*args接成数组并去重、排序，例如[t0, t1, t2, t3, t4] 然后打包成降序对 [(t4,t3),(t3,t2),(t2,t1),(t1,t0),(t0,t0)
    最后添加一个多余的(t0,t0)项, 因为障碍期权定价，最后定期初价格时，需要从df_bound切换到fd_full"""
    steps = np.sort(np.unique(np.hstack(args)))[::-1]
    steps = np.hstack((steps, steps[-1]))
    return list(zip(steps[:-1], steps[1:]))


def const_class(cls):
    """装饰器，修改常量类的属性，使其中的属性必须大写，并且值不能被更改"""

    @wraps(cls)
    def new_setattr(self, name, value):
        """重写设置属性方法
        允许加入新的常量，但不能更改已存在的常量的值
        新增常量的所有的字母需要大写
        Args:
            self: 常量类-对象
            name: 常量名，所有字母需要大写
            value: 常量值
        Returns: None
        """
        # # 不允许增加或修改任何常量的值
        # raise Exception('const : {} can not be changed'.format(name))
        # 允许加入新的常量，但不能更改已存在常量的值
        if name in self.__dict__:  # 若该常量已存在，不能二次赋值
            raise Exception(f"Can't change const {name}")
        if not name.isupper():  # 新增常量的所有的字母需要大写
            raise Exception(f"const name {name} is not all uppercase")
        self.__dict__[name] = value  # 允许加入新的常量，但不能更改值

    cls.__setattr__ = new_setattr
    return cls
