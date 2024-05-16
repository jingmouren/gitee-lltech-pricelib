#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
import time
from functools import wraps
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',  # - %(pathname)s[line:%(lineno)d]
                    level=logging.INFO)


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
