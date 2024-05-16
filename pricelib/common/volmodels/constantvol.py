#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from ..utilities.enums import VolType
from ..utilities.patterns import Observable
from ..utilities.utility import logging


class BlackConstVol(Observable):
    """BSM常数波动率，被观察者"""
    vol_type = VolType.CV

    def __init__(self, vol, name=None):
        super().__init__()
        self.__volval = vol
        self.__name = name

    def __call__(self, *args, **kwargs):
        return self.__volval

    @property
    def volval(self):
        return self.__volval

    @volval.setter
    def volval(self, new_value):
        """覆写给对象的属性赋值的方法。在被观察者的值改变时，自动向观察者发送通知"""
        try:
            self.__volval = float(new_value)  # 被观察者值变化
        except ValueError as e:
            logging.error(f"ValueError:{e}")
        else:
            self.notify_observers(new_value, name=self.__name)  # 自动向观察者发送通知
