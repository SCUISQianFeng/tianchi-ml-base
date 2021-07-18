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
import sys
import os
sys.path.append(os.pardir)


def get_discount_rate(s: typing.AnyStr) -> float:
    """
    计算折扣率，将满减和折扣统一
    :param s:
    :return:
    """
    s = str(s)
    if s == 'nan':
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


def get_month(s) -> int:
    """
    获取月份
    :param s:
    :return:
    """
    s = str(s)
    if s == 'nan':
        return -1
    else:
        return int(s[4:6])


def get_day(s: str) -> int:
    """
    获取日期
    :param s:
    :return:
    """
    s = str(s)
    if s == 'nan':
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
    if s[0] == 'nan':
        return -1
    if s[1] == 'nan':
        return -1
    else:
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                            int(s[1][6:8]))).days


def get_label(s: str) -> int:
    """
    获取label
    :param s:
    :return:
    """
    s = s.split(const.STR_SEPARATOR)
    if s[0] == 'nan':
        return 0
    if s[1] == 'nan':
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
    df['distance'] = df['distance'].replace('null', -1).fillna(-1).astype(int)
    df['month_received'] = df['date_received'].apply(get_month)
    if 'date' in df.columns:
        # 测试集没有date特征
        df['month'] = df['date'].apply(get_month)

    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df['day_gap'] = df['date'].astype('str') + ":" + df['date_received'].astype('str')
    df['label'] = df['day_gap'].apply(get_label)
    df['day_gap'] = df['day_gap'].apply(get_day_gap)
    return df


def add_agg_feature_names(df: pd.DataFrame, df_group: pd.DataFrame, group_cols, value_col: list[str],
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


def add_agg_feature(df: pd.DataFrame, df_group: pd.DataFrame, group_cols, value_col,
                    agg_ops, keyword) -> pd.DataFrame:
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


def add_count_new_feature(df: pd.DataFrame, df_group: pd.DataFrame, group_cols,
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
    # 在每个用户线下使用优惠券产生的交易中， 统计和商户距离的最大值、最小值、平均值、中位值
    t2 = user[(user.date != 'null') & (user.coupon_id != 'null') & (user.distance != 'null')][['user_id', 'distance']]
    t2.distance = t2.distance.astype('int')
    user_feature = add_agg_feature(user_feature, t2, ['user_id'], 'distance', ['min', 'max', 'mean', 'median'], 'user')
    # 每个用户使用优惠券的次数
    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    user_feature = add_count_new_feature(user_feature, t7, 'user_id', 'buy_use_coupon')
    # 每个用户消费的总次数
    t8 = user[user.date != 'null'][['user_id']]
    user_feature = add_count_new_feature(user_feature, t8, 'user_id', 'but_total')
    # 每个用户收到优惠券的总数
    t9 = user[user.coupon_id != 'null'][['user_id']]
    user_feature = add_count_new_feature(user_feature, t9, 'user_id', 'coupon_received')
    # 用户从收到优惠券到用券消费的时间间隔，统计其最大值、最小值、平均值、中位值
    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10 = get_day_gap(t10)
    t10 = t10[['user_id', 'day_gap']]
    user_feature = add_agg_feature(user_feature, t10, ['user_id'], 'day_gap', ['min', 'max', 'mean', 'median'], 'user')
    # 将数据中的NaN用0替换
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_user_coupon = user_feature.buy_user_coupon.replace(np.nan, 0)
    # 统计用户用券消费在总消费中的占比
    user_feature['buy_use_coupon_rate'] - user_feature.buy_user_coupon.astype('float') / user_feature.buy_total.astype(
        'float')
    # 统计用户收到消费券的使用率
    user_feature['buy_user_coupon_rate'] = user_feature.buy_user_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')
    # 将数据中的NaN用0来替换
    user_feature.but_total = user_feature.but_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    return user_feature


def get_user_merchant_feature(feature):
    """
    用户和商户关系特征群
    :param feature:
    :return:
    """
    t = feature[['user_id', 'merchant_id']].copy()
    # 用户和商户关系的去重
    t.drop_duplicates(inplace=True)

    # 一个用户在一个商户交易的次数
    t0 = feature[['user_id', 'merchant_id', 'date']].copy()
    t0 = t0[t0.coupon_id != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(t, t0, ['user_id', 'merchant_id'], 'user_merchant_but_total')
    # 一个用户在一个商家一共收到的优惠券数量
    t1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t1, ['user_id', 'merchant_id'], 'user_merchant_received')
    # 一个用户在一个商家使用优惠券消费的次数
    t2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    # group_cols -> 利用group_cols作为分组统计的基准，其他字段统计次数
    user_merchant = add_count_new_feature(user_merchant, t2, ['user_id', 'merchant_id'], 'user_merchant_buy_use_coupon')
    # 一个用户在一个商家的到店次数
    t3 = feature[['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t3, ['user_id', 'merchant_id'], 'user_merchant_any')
    # 一个用户在一个商家没有使用优惠券的次数
    t4 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id != 'null')][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t4, ['user_id', 'merchant_id'], 'user_merchant_buy_common')
    # 将数据中的NaN用0来替换
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    # 一个用户对一个商家的总消费次数中，有优惠券的消费次数占比
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_buy_common.astype('float')
    # 一个用户到店后消费的可能性统计
    user_merchant['user_merchant_comon_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_any.astype('float')
    return user_merchant


def get_leakage_feature(dataset):
    """
    Leakage特征群
    :param feature:
    :return:
    """
    t = dataset[['user_id']].copy()
    t['this_month_user_received_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    t1 = dataset[['user_id', 'coupon_id']].copy()
    t1['this_month_user_received_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()
    t2 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t2.date_received = t2.date_received.astype('str')
    # 如果出现相同的用户接收相同的优惠券，则在接收时间上用“：”连接上第n次优惠券的时间
    # user_id coupon_id date_received
    # 1       1         2011 2012 2013。。。这样的形式
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ":".join(x)).reset_index()
    # 将接收时间的一组按“：”分开，这样就可以计算所接收优惠券的数量
    # apply是合并
    t2['received_number'] = t2.date.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.received_number > 1]
    # 最大接收的日期
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    # 最小接收的日期
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset[['user_id', 'coupon_id', 'date_received']]
    # 将两个表融合只保留左表数据，相当于保留了最近接收时间和最远接收时间
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接收时间
    t3['this+_month_user_received_same_coupon_lastone'] = t3.max_date_received - t2.date_received.astype(int)
    # 这个优惠券最远接收时间
    t3['this_month_user_received_same_coupon_firstone'] = t3.date_received.astype(int) - t3.min_date_received
    t3.this_month_user_received_same_couon_lastone = t3.this_month_user_received_same_couon_lastone.apply(is_firstlastone)
    t3.this_month_user_received_same_couon_firstone = t3.this_month_user_received_same_couon_firstone.apply(is_firstlastone)


def f1(dataset, if_train):
    result = add_discount(dataset)
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


def f2(dataset, feature, if_train):
    result = add_discount(dataset)
    merchant_feature = get_merchant_feature(feature)
    result = result.merge(merchant_feature, on='merchant_id', how='left')
    user_feature = get_user_feature(feature)
    result = result.merge(user_feature, on='user_id', how='left')
    user_merchant = get_user_merchant_feature(feature)
    result = result.merge(user_merchant,
                          on=['user_id', 'merchant_id'],
                          how='left')
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


def f3(dataset, feature, if_train):
    result = add_discount(dataset)
    merchant_feature = get_merchant_feature(feature)
    result = result.merge(merchant_feature, on='merchant_id', how='left')
    user_feature = get_user_feature(feature)
    result = result.merge(user_feature, on='user_id', how='left')
    user_merchant = get_user_merchant_feature(feature)
    result = result.merge(user_feature, on=['user_id', 'merchant_id'], how='left')
    leakage_feature = get_leakage_feature(feature)
    result = result.merge(leakage_feature, on=['user_id', 'merchant_id', 'date_received'], how='left')
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


###########
# 特征输出
############
def normal_feature_generate(feature_function):
    off_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_train.csv'
    off_test_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_test_revised.csv'
    off_train = pd.read_csv(off_train_path, header=0, keep_default_na=False)
    off_test = pd.read_csv(off_test_path, header=0, keep_default_na=False)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    off_train = off_train[(off_train.coupon_id != 'null') & (off_train.date_received != 'null')
                          & (off_train.date_received >= '20160101')]
    dftrain = feature_function(off_train, True)
    dftest = feature_function(off_test, True)

    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)

    print('输出特征')
    dftrain.to_csv(feature + 'train_' + feature_function.__name__ + ".csv", index=False, sep=',')
    dftest.to_csv(feature + 'test_' + feature_function.__name__ + ".csv", index=False, sep=',')



def slide_feature_generate(feature_function):
    off_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_train.csv'
    off_test_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_test_revised.csv'
    off_train = pd.read_csv(off_train_path, header=0, keep_default_na=False)
    off_test = pd.read_csv(off_test_path, header=0, keep_default_na=False)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    # 交叉训练集一：收到券的日期大于4月14日且小于5月14日
    dataset1 = off_train[(off_train.date_received >= '201604014') & (off_train.date_revceived <= '20160514')]
    # 交叉训练集一特征：线下数据领券和用券时间大于1月1日且小于4月13日
    feature2 = off_train[(off_train.date >= '20160101') &
                         (off_train.date <= '20160413') |
                         ((off_train.date_received >= '20160101') &
                         (off_train.date_received <= '20160413') &
                         (off_train.date != 'null'))]
    # 交叉训练集二：收到券的日期大于5月15日且小于6月15日
    dataset1 = off_train[(off_train.date_received >= '201604014') & (off_train.date_revceived <= '20160514')]
    # 交叉训练集一特征：线下数据领券和用券时间大于1月1日且小于4月13日
    feature2 = off_train[(off_train.date >= '20160101') &
                         (off_train.date <= '20160413') |
                         ((off_train.date_received >= '20160101') &
                          (off_train.date_received <= '20160413') &
                          (off_train.date != 'null'))]
    # 测试集
    dataset3 = off_test
    # 测试集特征：线下数据中领券和用券的日期大于3月15日且小于6月30日
    feature3 = off_train[((off_train.date >= '20160315') &
                          (off_train.date <= '20160630'))]



    # off_train = off_train[(off_train.coupon_id != 'null') & (off_train.date_received != 'null')
    #                       & (off_train.date_received >= '20160101')]
    # dftrain = feature_function(off_train, True)
    # dftest = feature_function(off_test, True)
    #
    # dftrain.drop(['date'], axis=1, inplace=True)
    # dftrain.drop(['merchant_id'], axis=1, inplace=True)
    # dftest.drop(['merchant_id'], axis=1, inplace=True)
    #
    # print('输出特征')
    # dftrain.to_csv(feature + 'train_' + feature_function.__namne__ + ".csv", index=False, sep=',')
    # dftest.to_csv(feature + 'test_' + feature_function.__namne__ + ".csv", index=False, sep=',')






