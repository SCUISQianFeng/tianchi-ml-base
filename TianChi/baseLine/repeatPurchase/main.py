# -*- coding:utf-8 -*-
#  导入数据
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings('ignore')


def feature_selection(train, train_sel, target):
    """
    特征选择效果对比
    :param train: 未做特征选择的训练集
    :param train_sel: 已做特征选择的训练集
    :param target: 目标集
    :return:
    """
    clf = RandomForestClassifier(n_estimators=100,
                                 max_depth=2,
                                 random_state=0,
                                 n_jobs=-1)
    scores = cross_val_score(estimator=clf, X=train, y=target, cv=5)
    scores_sel = cross_val_score(estimator=clf, X=train_sel, y=target, cv=5)
    print('No Select Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    print('Select Accuracy: %0.2f (+/- %0.2f)' % (scores_sel.mean(), scores_sel.std() * 2))


if __name__ == "__main__":
    # 读取前1000行
    # train_data = pd.read_csv('train_all.csv', nrows=1000)
    # test_data = pd.read_csv('test_all.csv', nrows=1000)

    # 读取全部数据
    train_data = pd.read_csv('train_all.csv', nrows=None)
    test_data = pd.read_csv('test_all.csv', nrows=None)

    # 训练数据和测试数据处理
    feature_columns = [col for col in train_data.columns if col not in ['user_id', 'label']]
    train = train_data[feature_columns].values
    test = test_data[feature_columns].values
    target = train_data['label'].values

    imputer = SimpleImputer(strategy='median')
    imputer = imputer.fit(train)
    train_imputer = imputer.transform(train)
    test_imputer = imputer.transform(test)

    # 删除方差较小的特征
    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
    sel = sel.fit(train)
    train_sel = sel.transform(train)
    test_sel = sel.transform(test)
    print('训练数据未特征选择维度', train.shape)
    print('训练数据特征选择维度', train_sel.shape)

    feature_selection(train, train_sel, target)

    # 单变量特征选择
    sel = SelectKBest(score_func=mutual_info_classif, k=2)
    sel = sel.fit(train, target)
    train_sel = sel.transform(train)
    test_sel = sel.transform(test)
    print('训练数据未特征选择维度', train.shape)
    print('训练数据特征选择维度', train_sel.shape)

    feature_selection(train, train_sel, target)

    # 递归功能消除
    from sklearn.feature_selection import RFECV

    clf = RandomForestClassifier(n_estimators=10,
                                 max_depth=2,
                                 random_state=0,
                                 n_jobs=-1)

    selector = RFECV(estimator=clf, step=1, cv=2)
    selector = selector.fit(train, target)
    train_sel = train_sel.transform(train)
    test_sel = train_sel.transform(test)
    print('训练数据未特征选择维度', train.shape)
    print('训练数据特征选择维度', train_sel.shape)
    print(selector.support_)
    print(selector.ranking_)

    feature_selection(train, train_sel, target)

    # 使用模型选择特征
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer

    normalizer = Normalizer()
    normalizer = normalizer.fit(train)
    train_norm = normalizer.transform(train)
    test_norm = normalizer.transform(test)
    LR = LogisticRegression(penalty='l2', C=5)
    model = SelectFromModel(estimator=LR, prefit=True)
    train_sel = model.transform(train)
    test_sel = model.transform(test)
    print('训练数据未特征选择维度', train.shape)
    print('训练数据特征选择维度', train_sel.shape)

    feature_selection(train, train_sel, target)

    # 使用模型选择特征
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import Normalizer

    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(X=train, y=target)
    model = SelectFromModel(estimator=clf, prefit=True)
    train_sel = model.transform(train)
    test_sel = model.transform(test)
    print('训练数据未特征选择维度', train.shape)
    print('训练数据特征选择维度', train_sel.shape)

    feature_selection(train, train_sel, target)

