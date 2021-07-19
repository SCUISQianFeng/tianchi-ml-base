# -*- coding:utf-8 -*-

"""
    模型训练、验证、优化
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_sklearn_model(model_name):
    if model_name == 'NB':
        # 朴素贝叶斯
        return MultinomialNB(alpha=0.01)
    elif model_name == 'LR':
        # 逻辑回归
        return LogisticRegression(penalty='l2')
    elif model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name== 'DT':
        return DecisionTreeClassifier()
    elif model_name == 'SVM':
        return
