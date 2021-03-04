#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: data_process.py
@time: 2021/3/4 14:01
"""
from __future__ import print_function
import paddle
import numpy
import six.moves

def load_data_txt(filename):
    """
    如果想直接从txt文件中读取数据的话，可以参考以下方式。
    :param filename:
    :return:
    """
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'convert']

    feature_num = len(feature_names)

    data = numpy.fromfile(filename, sep=' ')  # 从文件中读取原始数据

    data = data.reshape(data.shape[0] // feature_num, feature_num)

    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0) / data.shape[0]

    for i in six.moves.range(feature_num - 1): data[:, i] = (data[:, i] - avgs[i]) / (
            maximums[i] - minimums[i])  # six.moves可以兼容python2和python3

    ratio = 0.8  # 训练集和验证集的划分比例

    offset = int(data.shape[0] * ratio)

    train_data = data[:offset]

    test_data = data[offset:]

    train_reader = paddle.batch(paddle.reader.shuffle(train_data, buf_size=500), batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.reader.shuffle(test_data, buf_size=500), batch_size=BATCH_SIZE)


if __name__ == '__main__':
    BATCH_SIZE = 20

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=BATCH_SIZE)
