#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: split_date.py
@time: 2021/3/5 11:53
"""
# -*- coding: utf-8 -*-

import random
import os
"""
随机按比例拆分数据
"""
def split(all_list, shuffle=False, ratio=0.8):
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    if shuffle:
        random.shuffle(all_list)  # 列表随机排序
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test


def write_split(film, train, test):
    infilm = open(film, 'r', encoding='utf-8')
    tainfilm = open(train, 'w', encoding='utf-8')
    testfilm = open(test, 'w', encoding='utf-8')
    li = []
    for datas in infilm.readlines():
        datas = datas.replace('\n', '')
        li.append(datas)
    traindatas, testdatas = split(li, shuffle=True, ratio=0.8)
    for traindata in traindatas:
        tainfilm.write(traindata + '\n')
    for testdata in testdatas:
        testfilm.write(testdata + '\n')
    infilm.close()
    tainfilm.close()
    testfilm.close()


if __name__ == "__main__":
    path = os.path.abspath('.').replace('\\', '/')
    data_path=path+r"/train.txt"
    train_path=path+r"/train_list.txt"
    test_path=path+r"/val_list.txt"
    write_split(data_path, train_path,test_path)

