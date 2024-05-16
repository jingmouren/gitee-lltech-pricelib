#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from abc import ABCMeta, abstractmethod  # 引入ABCMeta 和 abstractmethod 来定义抽象类和抽象方法
import numpy as np
from .utility import logging


class HashableArray(np.ndarray):
    """自定义HashableArray类，继承自np.ndarray。自定义这个类的目的是让可哈希的array能够和@functools.lru_cache装饰器配合使用。
    lru_cache实现了 least recently used cache 最近最少使用缓存。它会为函数建立一个哈希表，当函数新的input已经存在于哈希表的键，则直接返回哈希表中保存的对应output值。
    但是np.ndarray是不可哈希对象，无法作为哈希表的键。为了当函数输入的参数是array时，能够使用lru_cache，需要自定义一个可哈希的array类。"""

    def __new__(cls, *args):
        """覆写np.ndarray的实例化方法，在原来实例化方法的基础上，将实例转化为HashableArray对象。
        类的实例化包含两个步骤: (1)在内存中创建对象，即开辟一块内存空间来存放类的实例Instance; (2)初始化对象，即给实例的属性赋予初始值，例如全部填0
        在 python 中，第一步由 __new__ 函数负责，第二步由 __init__ 函数负责。
        np.asarray方法可以将所有的*args参数转换为数组np.ndarray，然后view方法将数组np.ndarray转化为cls类(即HashableArray类)。
        view是"视图", 与np的copy方法相反，view是数据的引用，修改view返回的变量的值，会影响原始数据的值，而修改view维数和类型不会影响原始数据的维数和类型
        """
        return np.asarray(*args).view(cls)

    def __getitem__(self, *args):
        """覆写np.ndarray的切片方法__getitem__。array的索引切片本来返回的是array对象，将其转化为HashableArray对象"""
        return HashableArray(super().__getitem__(*args))

    def __hash__(self):
        """__hash__方法用于实现对象的哈希值计算。当对象被用作字典的键或者放入集合中时，__hash__方法会被调用来计算哈希值。
        这里先用np.ndarray的tostring方法将数组的值转化为bytes，然后用hash方法将bytes转化为一个整数，即哈希值。"""
        return hash(self.tobytes())

    def __eq__(self, other):
        """__eq__自定义类型比较，这是为了与__hash__方法配合，当想要看一个新数组other是否在哈希表中时，会与哈希表已有的作为键的数组进行比较。
        只要np.array_equal两个数组每个位置的元素相同，即认为两个数组相同。那么就可以认为新数组other已经在哈希表中，直接返回哈希表对应的值"""
        return np.array_equal(self, other)


class Observer(metaclass=ABCMeta):  # metaclass=ABCMeta: 使创建的类为抽象类
    """观察者的基类"""

    @abstractmethod  # 子类必须实现抽象类的抽象方法，这样可以使子类有统一的类方法
    def update(self, observable, *args, **kwargs):
        """收到被观察者的通知时，更新观察者。
        例: 自动改变被观察者的行情属性的值，更新缓存结果
        def update(self, observable, *args, **kwargs):
            if observable is self._s0:
                print(f"观察者收到来自被观察者 s0 的通知：值更新为{observable.data}")
            elif observable is self._sigma:
                print(f"观察者收到来自被观察者 sigma 的通知：值更新为{observable.data}")
            self.__cache = self._calculate(self._s0.data, self._sigma.data)
            print(f"缓存值更新为{self.__cache}")"""

    @abstractmethod
    def remove_self(self):
        """删除对象自己，del之前，先将自己从被观察者的列表中移除
        例:
        self._s0.remove_observer(self)
        self._sigma.remove_observer(self)
        del self"""


class Observable(metaclass=ABCMeta):
    """被观察者的基类"""

    def __init__(self):
        """初始化观察者列表"""
        self.__observers = []

    @property
    def observers_list(self):
        return self.__observers

    def add_observer(self, observer):
        """在观察者列表中添加新的观察者"""
        self.__observers.append(observer)

    def remove_observer(self, observer):
        """在观察者列表中删除指定的观察者"""
        if observer in self.__observers:
            self.__observers.remove(observer)

    def notify_observers(self, *args, **kwargs):
        """当被观察者的内容或状态变化时，通知所有的观察者自动更新"""
        for o in self.__observers:
            o.update(self, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """被观察者一般是市场行情或估值参数，因此有__call__方法，返回指定时间(及标的价格)下的值"""


class SimpleQuote(Observable):
    """简单行情类，被观察者"""

    def __init__(self, value, name=None):
        super().__init__()
        self.__data = value
        self.__name = name

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, new_value):
        """覆写给对象的属性赋值的方法。在被观察者的值改变时，自动向观察者发送通知"""
        try:
            self.__data = float(new_value)  # 被观察者值变化
        except ValueError as e:
            logging.error(f"ValueError:{e}")
        else:
            self.notify_observers(new_value, name=self.__name)  # 自动向观察者发送通知

    def __call__(self, *args, **kwargs):
        return self.data

    @property
    def name(self):
        return self.__name


if __name__ == "__main__":
    # 举例: 标的价格S和波动率sigma是被观察者，BsEulerEvolve是观察者，被观察者的值变化，BsEulerEvolve的__cachevalue会自动变化
    # 并且，通过使用@lru_cache(maxsize=1)装饰器，BsEulerEvolve是观察者会自动缓存最近1次的计算结果，
    # 当输入不变时，直接返回上一次的结果；当输入变化时，重新计算__cachevalue并更新保存的计算结果。
    from functools import lru_cache


    class BsEulerEvolve(Observer):
        """BS SDE欧拉离散更新, 观察者"""

        def __init__(self, underlying: SimpleQuote, sigma: SimpleQuote, riskfree: float, dt: float):
            assert (isinstance(underlying, Observable) and
                    isinstance(sigma, Observable)), "Error: underlying和sigma不是Observable实例"
            super().__init__()
            self.__cachevalue = None  # 存储结果
            self._riskfree = riskfree
            self._dt = dt
            self._s0 = underlying
            underlying.add_observer(self)  # 将本实例加入underlying观察者列表
            self._sigma = sigma
            sigma.add_observer(self)  # 将本实例加入sigma观察者列表

        def evolve(self):
            self.__cachevalue = self._euler_discretization(self._s0.data, self._sigma.data)
            logging.info(f"evolve result = {self.__cachevalue}")

        @lru_cache(maxsize=1)  # 只缓存最近的一种结果
        def _euler_discretization(self, spot, sigma):
            return spot * (1 + self._riskfree * self._dt + sigma * np.random.rand() * np.sqrt(self._dt))

        def update(self, observable, *args, **kwargs):
            if isinstance(observable, SimpleQuote):
                if observable is self._s0:
                    logging.info(f"观察者收到来自被观察者s0的通知：数据更新为{observable.data}")
                elif observable is self._sigma:
                    logging.info(f"观察者收到来自被观察者sigma的通知：数据更新为{observable.data}")
                logging.info(f"update传入的位置参数: {args}，关键字参数: {kwargs}")
                logging.info(f"被观察者的类名 = {type(observable).__name__}")
                self.evolve()

        def remove_self(self):
            """删除对象自己，del自己之前，先将自己从被观察者的列表中移除"""
            self._s0.remove_observer(self)
            self._sigma.remove_observer(self)
            del self


    # 被观察者——标的价格初始值为100
    S = SimpleQuote(value=100, name="标的价格")
    vol = SimpleQuote(value=0.2, name="波动率")
    # 观察者——Black Scholes SDE欧拉离散迭代一步evolve
    dynamic = BsEulerEvolve(underlying=S, sigma=vol, riskfree=0.02, dt=0.005)
    print("1. 初始值:")
    dynamic.evolve()
    print("1'. 再次调用evolve，由于被观察者无变化，直接返回缓存结果，观察者值不变")
    dynamic.evolve()
    print("2. 被观察者【S】改变，观察者evolve自动重新计算:")
    S.data = 105
    print("2'. 再次调用evolve，由于被观察者无变化，直接返回缓存结果，观察者值不变")
    dynamic.evolve()
    print("3. 被观察者【vol】改变，观察者evolve自动重新计算:")
    vol.data = 0.3
    print(f"4. 删除观察者对象，会从被观察者列表中移除自己\nbefore: {S.observers_list, vol.observers_list}")
    dynamic.remove_self()
    print(f"after: {S.observers_list, vol.observers_list}")
