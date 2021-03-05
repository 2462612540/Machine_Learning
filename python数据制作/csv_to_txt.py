#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: csv_to_txt.py
@time: 2021/3/5 11:37
"""

import pandas as pd
import os


def csv_to_txt(csv_file, txt_file,abs_path):
    if not os.path.exists(csv_file):
        print('Not that files:%s' % csv_file)
    else:
        data = pd.read_csv(csv_file, encoding='utf-8')
        with open(txt_file, 'a+', encoding='utf-8') as f:
            for line in data.values:
                newdata=abs_path+str(line[0]) + ' ' + str(line[1]) + '\n'
                f.write(newdata)


if __name__ == '__main__':
    path=os.path.abspath('.').replace('\\','/')
    csv_file = path+r"/train.csv"
    txt_file =path+ r"/train.txt"
    abs_path=path+r"/train_images/"
    csv_to_txt(csv_file, txt_file,abs_path)
