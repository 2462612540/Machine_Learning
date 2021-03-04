#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: data_process.py
@time: 2021/3/4 10:21
"""
#导入需要的包
import paddle as paddle
BUF_SIZE = 512
BATCH_SIZE = 128

# 1、读取数据
def load_data():
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(),
                              buf_size=BUF_SIZE),
        batch_size=BATCH_SIZE)
    # 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
    test_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.test(),
                              buf_size=BUF_SIZE),
        batch_size=BATCH_SIZE)
    return train_reader, test_reader


