# -*- coding:utf-8 -*-

"""
    工具类方法
"""
from constant import const
import typing
import datetime as dt
from datetime import date
import pandas as pd
import numpy as np


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


def get_label(s: str) -> int:
    """
    获取label
    :param s:
    :return:
    """
    s = s.split(const.STR_SEPARATOR)
    if s[0] == 'null':
        return 0
    if s[1] == 'null':
        return -1
    elif ((date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
           date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days <= 15):
        # 15天内进行了消费
        return 1
    else:
        return -1


def add_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['if_fd'] = df['discount_rate'].apply(get_if_fd)
    df['full_value'] = df['discount_rate'].apply(get_full_value)
    df['reduction_value'] = df['discount_rate'].apply(get_reduction_value)
    df['discount_rate'] = df['discount_rate'].apply(get_discount_rate)
    df['distance'] = df['distance'].repalce('null', -1).astype(int)
    df['month_received'] = df['date_received'].apply(get_month)
    df['month'] = df['date_received'].apply(get_month)
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df['day_gap'] = df['date'].astype('str') + ":" + df['date_received'].astype('str')
    df['label'] = df['day_gap'].apply(get_label)
    df['day_gap'] = df['day_gap'].apply(get_day_gap)
    return df


def add_agg_feature_names(df: pd.DataFrame, df_group: pd.DataFrame, group_cols: list[str], value_col: list[str],
                          agg_ops, col_names: list[str]) -> pd.DataFrame:
    """
    统计特征处理函数
    :param df: 添加特征的 DataFrame
    :param df_group: 特征生成的数据集
    :param group_cols: group by 的列
    :param value_col: 被统计的列
    :param agg_ops: 处理方式 包括count, mean, sum, std, max, min, nunique
    :param col_names: 新特征的名称
    :return: DataFrame
    """
    df_group[value_col] = df_group[value_col].astype('float')
    df_agg = pd.DataFrame(df_group.groupby(group_cols)[value_col].agg(agg_ops)).reset_index()
    df_agg.columns = group_cols + col_names
    df = df.merge(df_agg, on=group_cols, how='left')
    return df


def add_agg_feature(df: pd.DataFrame, df_group: pd.DataFrame, group_cols: list[str], value_col: list[str],
                    agg_ops: list[str], keyword) -> pd.DataFrame:
    """
    统计特征处理函数 名称按照keyword + '_' + value_col + '_' + op
    :param df:
    :param df_group:
    :param group_cols:
    :param value_col:
    :param agg_ops:
    :param keyword:
    :return:
    """
    col_names = []
    for op in agg_ops:
        col_names.append(keyword + '_' + value_col + '_' + op)
    df = add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names)
    return df


def add_count_new_feature(df: pd.DataFrame, df_group: pd.DataFrame, group_cols: list[str],
                          new_feature_name: str) -> pd.DataFrame:
    """
    因为count特征很多， 专门提取count特征的函数
    :param df:
    :param df_group:
    :param group_cols:
    :param new_feature_name:
    :return:
    """
    df_group[new_feature_name] = 1
    df_group = df_group.groupby(group_cols).agg('sum').reset_index()
    df = df.merge(df_group, on=group_cols, how='left')
    return df


#################################################
# 特征群生成函数  商户相关特征群
#################################################


def get_merchant_feature(feature):
    """
    商户相关特征群
    :param feature:
    :return:
    """
    merchant = feature[['merchant_id', 'coupon_id', 'distance', 'data_received', 'date']].copy()
    t = merchant[['merchant_id']].copy()
    # 删除重复行数据
    t.drop_duplicates(inplace=True)
    # 每个商户的交易总次数
    t1 = merchant[merchant.date != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(t, t1, 'merchant_id', 'total_sales')
    # 在每个商户销售中，使用优惠券的交易次数
    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(merchant_feature, t2, 'merchant_id', 'sales_use_coupon')
    # 每个商户发放的优惠券总数
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(merchant_feature, t3, 'merchant_id', 'total_coupon')
    # 在每个线下商户含有优惠券的交易中，统计和用户距离的最大值、最小值、平均值， 中位值
    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')
                  & (merchant.distance != 'null')][['merchant_id', 'distance']].copy()
    t4.distance = t4.distance.astype('int')
    merchant_feature = add_agg_feature(merchant_feature, t4, ['merchant_id'],
                                       'distance',
                                       ['min', 'max', 'mean', 'median', 'merchant'])
    # 将数据中的NaN用0来替换
    merchant_feature.sales_user_coupon = merchant_feature.sales_user_coupon.replace(np.nan, 0)
    # 商户发放优惠券的使用率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_user_coupon.astype(
        'float') / merchant_feature.total_coupon
    # 在商户交易中，使用优惠券的交易占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_user_coupon.astype('float') / merchant_feature.total_sales
    # 将数据中的NaN用0替换
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)
    return


def get_user_feature(feature):
    # for dataset3
    user = feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']].copy()
    t = user[['user_id']].copy()
    t.drop_duplicates(inplace=True)
    # 每个用户交易的商户数
    t1 = user[user.date != 'null'][['user_id', 'merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1 = t1['user_id']
    user_feature = add_count_new_feature(t, t1, 'user_id', 'count_merchant')
    # 在每个用户线下使用优惠券产生的交易中， 统计和商户距离的最大值

