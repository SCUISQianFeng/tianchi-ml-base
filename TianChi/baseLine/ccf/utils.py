# -*- coding:utf-8 -*-

"""
    工具类方法
"""
from constant import const
import typing
import datetime as dt
from datetime import date


def get_discount_rate(s: typing.AnyStr) -> float:
    """
    计算折扣率，将满减和折扣统一
    :param s:
    :return:
    """
    s = str(s)
    if s == 'null':
        return -1
    s = s.split(const.STR_SEPARATOR)
    if len(s) == 1:
        # 0.5的形式
        return float(s[0])
    else:
        # 50:20 满50减20的形式
        return 1.0 - float(s[1]) / float(s[0])


def get_if_fd(s: typing.AnyStr) -> int:
    """
    获取是否满减: 输入是50:20的形式就是满减
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(const.STR_SEPARATOR)
    if len(s) == 1:
        return -1
    else:
        return int(s[0])


def get_full_value(s: [typing.AnyStr, typing.SupportsFloat]) -> int:
    """
    获取满减的条件
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(const.STR_SEPARATOR)
    if len(s) == 1:
        return -1
    else:
        return int(s[0])


def get_reduction_value(s: typing.AnyStr) -> int:
    """
    获取满减的优惠
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(const.STR_SEPARATOR)
    if len(s) == 1:
        return -1
    else:
        return int(s[1])


def get_month(s: str) -> int:
    """
    获取月份
    :param s:
    :return:
    """
    if s[0] == 'null':
        return -1
    else:
        return int(s[4:6])


def get_day(s: str) -> int:
    """
    获取日期
    :param s:
    :return:
    """
    if s[0] == 'null':
        return -1
    else:
        return int(s[6:8])


def get_day_gap(s: str) -> int:
    """
    获取时间间隔
    :param s:
    :return:
    """
    s = s.split(const.STR_SEPARATOR)
    if s[0] == 'null':
        return -1
    if s[1] == 'null':
        return -1
    else:
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
                date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days
