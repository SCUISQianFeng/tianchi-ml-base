# -*- coding:utf-8 -*-

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd


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
    批处理数据切片
    :param data: 源数据
    :param batch_size: 每批的数据大小
    :return: 迭代数据
    """
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        # 最后一个批次的数据可能小于batch_size
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """
    batched_data = []
    if shuffle:
        np.random.shuffle(data)

        lengths = [example[1] for example in data]
        print(lengths)
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch


if __name__ == "__main__":
    data_path = r'E:\DataSet\Tianchi\nlpNews\train_set\train_set.csv'
    # data = pd.read_csv(data_path, sep='\t')
    data = pd.read_csv(data_path, sep='\t', nrows=100)
    # print(data['label'])
    batch_slice(data, 10)
