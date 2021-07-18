# -*- coding:utf-8 -*-

"""
    数据预处理部分
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from utils import add_feature
from utils import add_label

sys.path.append(os.pardir)
import warnings

warnings.filterwarnings('ignore')

off_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_train.csv'
off_test_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_test_revised.csv'
on_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_online_stage1_train.csv'

off_train = pd.read_csv(off_train_path, keep_default_na=True)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

off_test = pd.read_csv(off_test_path, keep_default_na=True)
off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

on_train = pd.read_csv(on_train_path, keep_default_na=True)
on_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

# 拷贝数据，以免在调试时重读文件
dftrain = off_train.copy()
dftest = off_test.copy()

dftrain = add_feature(dftrain)
dftrain = add_label(dftrain)
dftest = add_feature(dftest)

print('Offline 训练集满减情况', dftrain.if_fd.value_counts() / dftrain.if_fd.count())
print('测试集满减情况', dftest.if_fd.value_counts() / dftest.if_fd.count())  # 满减分布情况相差较大

# 数据可视化
# 箱线图
fig = plt.figure(figsize=(4, 6))
sns.boxplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], orient="v", width=0.95)
plt.show()

# 直方图和Q-Q图
plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 2, 1)
sns.distplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], fit=stats.norm)
ax = plt.subplot(1, 2, 2)
res = stats.probplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], plot=plt)
plt.show()

# 分布对比
ax = sns.kdeplot(dftrain[(dftrain.label >= 0) & (dftrain.discount_rate >= 0)]['discount_rate'],
                 color='red',
                 shade=True)
ax = sns.kdeplot(dftrain[(dftrain.discount_rate >= 0)]['discount_rate'],
                 color='Blue',
                 shade=True)
ax.set_xlabel('discount_rate')
ax.set_ylabel('Frequency')
ax = ax.legend(['train', 'test'])
plt.show()

# 线性关系
fcols = 2
frows = 1
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
sns.regplot(x='distance',
            y='label',
            data=dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)][['distance', 'label']],
            ax=ax,
            scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
            line_kws={'color': 'k'})
plt.xlabel('distance')
plt.ylabel('label')
ax = plt.subplot(1, 2, 2)
sns.displot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'].dropna())
plt.xlabel('distance')
plt.show()
