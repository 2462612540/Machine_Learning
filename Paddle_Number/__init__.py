#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: train_model.py.py
@time: 2021/3/4 9:44
注: paddle.fluid.*, paddle.dataset.* 会在未来的版本中废弃，请您尽量不要使用这两个目录下的API。
深度学习步骤：
    １. 数据处理：读取数据 和 预处理操作
    ２. 模型设计：网络结构（假设）
    ３. 训练配置：优化器（寻解算法）
    ４. 训练过程：循环调用训练过程，包括前向计算 + 计算损失（优化目标) + 后向传播
    ５. 保存模型并测试：将训练好的模型保存
"""
# 导入需要的包
import numpy as np
import paddle as paddle


TRAIN_BUF_SIZE = 51200
TRAIN_BATCH_SIZE = 51200
TEST_BUF_SIZE = 12800
TEST_BATCH_SIZE = 12800

# **********************************************************************************************
# 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(),
                          buf_size=TRAIN_BUF_SIZE),
    batch_size=TRAIN_BATCH_SIZE)
# 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(),
                          buf_size=TEST_BUF_SIZE),
    batch_size=TEST_BATCH_SIZE)

train_iterator = next(train_reader())
test_iterator = next(test_reader())
# **********************************************************************************************

# 将数据转为numpy Array
train_data = np.array([data[0] for data in train_iterator]).astype("float32")
test_data = np.array([data[0] for data in test_iterator]).astype("float32")

train_label = np.array([data[1] for data in train_iterator]).astype("float32")
test_label = np.array([data[1] for data in test_iterator]).astype("float32")

# print(train_data.shape)
# 将白底黑字转化为黑底白字拼接在原矩阵下
train_data = np.vstack((train_data, -train_data))
test_data = np.vstack((test_data, -test_data))

# 构造新label训练集和测试集
train_label = np.vstack((train_label.reshape(-1, 1), train_label.reshape(-1, 1)))
test_label = np.vstack((test_label.reshape(-1, 1), test_label.reshape(-1, 1)))
# print(train_data.shape)

train_index = [i for i in range(len(train_data))]
# print(train_index)
np.random.shuffle(train_index)  # 打乱索引
train_data = train_data[train_index]
train_label = train_label[train_index]

test_index = [i for i in range(len(test_data))]
np.random.shuffle(test_index)  # 打乱索引
test_data = test_data[test_index]
test_label = test_label[test_index]


