#!/user/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 Galaxy Technologies
Licensed under the Apache License, Version 2.0
"""
from enum import Enum, unique  # @unique装饰器可以防止枚举成员具有相同的值
from collections import namedtuple  # 具名元组，实现枚举成员的自定义属性


@unique
class CallPut(Enum):  # 看涨/看跌
    Call = 1
    Put = -1


@unique
class BuySell(Enum):  # 买入/卖出
    Buy = 1
    Sell = -1


@unique
class InOut(Enum):  # 敲入/敲出
    In = "敲入"
    Out = "敲出"


@unique
class TouchType(Enum):  # 触碰/不触碰
    Touch = "触碰"
    NoTouch = "不触碰"


@unique
class UpDown(Enum):  # 向上/向下
    Up = 1
    Down = -1


# 具名元组: 障碍期权类型
BarrierTuple = namedtuple(typename='BarrierTuple', field_names=['text', 'updown', 'inout', 'callput'])


@unique
class BarrierType(Enum):  # 障碍期权类型
    UOC = BarrierTuple("向上敲出看涨", UpDown.Up, InOut.Out, CallPut.Call)
    UOP = BarrierTuple("向上敲出看跌", UpDown.Up, InOut.Out, CallPut.Put)
    DOC = BarrierTuple("向下敲出看涨", UpDown.Down, InOut.Out, CallPut.Call)
    DOP = BarrierTuple("向下敲出看跌", UpDown.Down, InOut.Out, CallPut.Put)
    UIC = BarrierTuple("向上敲入看涨", UpDown.Up, InOut.In, CallPut.Call)
    UIP = BarrierTuple("向上敲入看跌", UpDown.Up, InOut.In, CallPut.Put)
    DIC = BarrierTuple("向下敲入看涨", UpDown.Down, InOut.In, CallPut.Call)
    DIP = BarrierTuple("向下敲入看跌", UpDown.Down, InOut.In, CallPut.Put)

    @property
    def updown(self):
        return self.value.updown

    @property
    def inout(self):
        return self.value.inout

    @property
    def callput(self):
        return self.value.callput

    @staticmethod
    def get_category(updown, input, callput):
        return BARRIER_TYPE_MAPPING[f"{updown.value}{input.value}{callput.value}"]


BARRIER_TYPE_MAPPING = {"1敲出1": BarrierType.UOC, "1敲出-1": BarrierType.UOP, "-1敲出1": BarrierType.DOC,
                        "-1敲出-1": BarrierType.DOP, "1敲入1": BarrierType.UIC, "1敲入-1": BarrierType.UIP,
                        "-1敲入1": BarrierType.DIC, "-1敲入-1": BarrierType.DIP}


@unique
class RandsMethod(Enum):  # 随机数生成方法
    Pseudorandom = "伪随机"
    LowDiscrepancy = "低差异"


@unique
class LdMethod(Enum):  # 低差异序列生成方法
    Halton = "Halton"
    Sobol = "Sobol"


@unique
class VolType(Enum):  # 波动率类型
    CV = "常数波动率"
    LV = "局部波动率"
    SV = "随机波动率"
    SLV = "随机局部波动率"


@unique
class ProcessType(Enum):  # 目前支持的随机过程类型
    BSProcess1D = "一维BS随机过程"
    Heston = "Heston随机过程"


@unique
class EngineType(Enum):  # 定价引擎类型
    AnEngine = "解析法定价引擎"
    McEngine = "蒙特卡洛模拟定价引擎"
    TreeEngine = "树方法定价引擎"
    PdeEngine = "PDE方法定价引擎"
    QuadEngine = "数值积分定价引擎"


@unique
class QuadMethod(Enum):  # 数值积分方法
    Trapezoid = "梯形法则"
    Simpson = "Simpson法则"


@unique
class ExerciseType(Enum):  # 行权方式
    European = "欧式"
    American = "美式"
    Asian = "亚式"
    Bermudan = "百慕大"


@unique
class PaymentType(Enum):  # 支付方式
    Expire = "到期支付"
    Hit = "立即支付"


@unique
class AverageMethod(Enum):
    Arithmetic = "算术平均"
    Geometric = "几何平均"


@unique
class AsianAveSubstitution(Enum):
    Underlying = "代替标的结算价"
    Strike = "代替执行价"


@unique
class StatusType(Enum):
    DownTouch = "敲入"
    UpTouch = "敲出"
    NoTouch = "未敲入未敲出"
