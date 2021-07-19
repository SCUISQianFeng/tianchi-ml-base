# -*- coding:utf-8 -*-

"""
    模型训练、验证、优化
"""
import os
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import get_predictors_df
from utils import get_target_df
from utils import myauc
from utils import get_id_df
from utils import read_data
from xgboost import XGBClassifier

sys.path.append(os.pardir)


def get_sklearn_model(model_name):
    if model_name == 'NB':
        # 朴素贝叶斯
        return MultinomialNB(alpha=0.01)
    elif model_name == 'LR':
        # 逻辑回归
        return LogisticRegression(penalty='l2')
    elif model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name == 'DT':
        return DecisionTreeClassifier()
    elif model_name == 'SVM':
        return SVC(kernel='rbf')
    elif model_name == 'GBDT':
        return GradientBoostingClassifier()
    elif model_name == 'RF':
        return RandomForestClassifier()
    elif model_name == 'XGB':
        return XGBClassifier()
    elif model_name == 'LGB':
        return LGBMClassifier()
    else:
        print('wrong model name')


def test_model(traindf, classifier):
    """
    按照日期分割
    :param traindf:
    :param classifier:
    :return:
    """
    train = traindf[traindf.date_received < 20160615].copy()
    test = traindf[traindf.date_received >= 20160615].copy()
    train_data = get_predictors_df(train).copy()
    test_data = get_predictors_df(test).copy()
    train_target = get_target_df(train_data).copy()
    test_target = get_target_df(test_data).copy()
    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]
    test['pred'] = result
    score = metrics.roc_auc_score(y_true=test_target, y_score=result)
    print(classifier + " 总体AUC： ", score)
    score_coupon = myauc(test)
    print(classifier + " Coupon AUC: ", score_coupon)


def test_model_split(traindf, classifier):
    target = get_target_df(traindf).copy()
    train_all, test_all, train_target, test_target = train_test_split(traindf, target, test_size=0.2, random_state=0)
    train_data = get_predictors_df(train_all).copy()
    test_data = get_target_df(test_all).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]
    test = test_all.copy()
    test['pred'] = result
    score = metrics.roc_auc_score(y_true=test_target, y_score=result)
    print(classifier + " 总体AUC： ", score)
    score_coupon = myauc(test)
    print(classifier + " Coupon AUC: ", score_coupon)


def classifier_df_simple(train_feat, test_feat, classifier):
    """
    预测函数
    :param train_feat:
    :param test_feat:
    :param classifier:
    :return:
    """
    model = get_sklearn_model(classifier)
    model.fit(get_predictors_df(train_feat), get_target_df(train_feat))
    predicted = pd.DataFrame(model.predict_proba(get_predictors_df(test_feat))[:, 1])
    return predicted


def output_predicted(predicted, resultfile, test_feat):
    predicted = round(predicted, 3)
    resultdf = get_id_df(test_feat).copy()
    resultdf['Probability'] = predicted
    return resultdf


if __name__ == "__main__":
    traindf, testdf = read_data('sf3')
    print('特征sf1朴素贝叶斯成绩')
    test_model(train_f1, 'NB')
    print('特征sf2朴素贝叶斯成绩')
    test_model(train_f2, 'NB')
    print('特征sf3朴素贝叶斯成绩')
    test_model(train_f3, 'NB')

    print('特征sf1逻辑回归成绩')
    test_model(train_f1, 'LR')
    print('特征sf2逻辑回归成绩')
    test_model(train_f2, 'LR')
    print('特征sf3逻辑回归成绩')
    test_model(train_f3, 'LR')

    print('特征sf1决策树成绩')
    test_model(train_f1, 'DT')
    print('特征sf2决策树成绩')
    test_model(train_f2, 'DT')
    print('特征sf3决策树成绩')
    test_model(train_f3, 'DT')

    print('特征sf1随机森林成绩')
    test_model(train_f1, 'RF')
    print('特征sf2随机森林成绩')
    test_model(train_f2, 'RF')
    print('特征sf3随机森林成绩')
    test_model(train_f3, 'RF')

    print('特征sf1 XGBoost 成绩')
    test_model(train_f1, 'XGB')
    print('特征sf2 XGBoost 成绩')
    test_model(train_f2, 'XGB')
    print('特征sf3 XGBoost 成绩')
    test_model(train_f3, 'XGB')

    print('特征sf1 LightGBM 成绩')
    test_model(train_f1, 'LGB')
    print('特征sf2 LightGBM 成绩')
    test_model(train_f2, 'LGB')
    print('特征sf3 LightGBM 成绩')
    test_model(train_f3, 'LGB')

    predicted = classifier_df_simple(train_f3, test_f3, "LGB")
    # 生成结果数据
    result = output_predicted(predicted, 'sf3_LGB.csv', test_f3)
    result.to_csv('sf3_lgb.csv', header=False, index=False, sep=',')
