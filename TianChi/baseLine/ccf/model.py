# -*- coding:utf-8 -*-

"""
    模型训练、验证、优化
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
from utils import f1
from utils import f2
from utils import f3
from utils import get_id_df
from utils import get_predictors_df
from utils import get_target_df
from utils import myauc
from utils import normal_feature_generate
from utils import read_data
from utils import slide_feature_generate
from utils import strandize_df
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

sys.path.append(os.pardir)

####################### 全局参数 ############
myeval = 'roc_auc'


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
    normal_feature_generate
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    """
    画学习曲线
    :param estimator:
    :param title:
    :param X:
    :param y:
    :param ylim:
    :param cv:
    :param n_jobs:
    :param train_sizes:
    :return:
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=myeval, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.mean(test_scores, axis=1)

    plt.fill_between(x=train_sizes, y1=train_scores_mean - train_scores_std, y2=train_scores_mean + train_scores_std,
                     alpha=0.1, color='r')
    plt.fill_between(x=train_sizes, y1=test_scores_mean - test_scores_std, y2=test_scores_mean + test_scores_std,
                     alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Train score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Train score')
    plt.legend('best')
    return plt


def plot_curve_single(traindf, classifier, cvnum, train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    X = get_predictors_df(traindf)
    y = get_target_df(traindf)
    estimator = get_sklearn_model(classifier)
    title = 'learning curve of ' + classifier + ", cv: " + str(cvnum)
    plot_learning_curve(estimator=estimator, title=title, X=X, y=y, ylim=(0, 1.01), cv=cvnum, train_sizes=train_sizes)


if __name__ == "__main__":
    # 生成特征文件
    normal_feature_generate(f1)
    slide_feature_generate(f2)
    slide_feature_generate(f3)

    train_f1, test_f1 = read_data('f1')
    train_f1, test_f1 = strandize_df(train_f1, test_f1)
    train_f2, test_f2 = read_data('f2')
    train_f2, test_f2 = strandize_df(train_f2, test_f2)
    train_f3, test_f3 = read_data('f3')
    train_f3, test_f3 = strandize_df(train_f3, test_f3)

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
