# -*- coding:utf-8 -*-

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def reformat(num, n):
    """
    小数格式化 类似于 0.2f
    :param num: 待格式化
    :param n: 保留小数位数
    :return: float
    """
    return float(format(num, '0.' + str(n) + 'f'))


def get_score(y_true, y_pred):
    """
    计算算法准确度
    :param y_true:
    :param y_pred:
    :return:
    """
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    return str((reformat(f1, 2), reformat(precision, 2), reformat(recall, 2))), reformat(recall, 2)


def batch_slice(data, batch_size):
    """
    批处理切片
    :param data: 源数据
    :param batch_size: 每批的数据大小
    :return:  迭代数据
    """
