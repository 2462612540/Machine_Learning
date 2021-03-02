#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: unzip.py
@time: 2021/2/28 14:27
@target :该文件的主要是用于的解压文件夹以及子文件下面的相关的zip 文件包并删除原来的文件zip
"""
import zipfile
import os
import shutil

def unzip(dirpath):

    """传入的是zip的文件路径 解压zip文件在当前目录中"""
    zip_file = zipfile.ZipFile(dirpath)  # 获取压缩文件
    newfilepath = dirpath.split(".", 1)[0].replace("【瑞客论坛 www", "") # 获取压缩文件的文件名
    if os.path.isdir(newfilepath):  # 根据获取的压缩文件的文件名建立相应的文件夹
        pass
    else:
        os.mkdir(newfilepath)
    for name in zip_file.namelist():  # 解压文件
        zip_file.extract(name, newfilepath)
    zip_file.close()

    # 判断解压或的文件中是否存在的zip文件
    file_name(newfilepath)

    # 如存在配置文件，则删除（需要删则删，不要的话不删）
    Conf = os.path.join(newfilepath, 'conf')
    if os.path.exists(Conf):
        shutil.rmtree(Conf)
    # 删除原先压缩包
    if os.path.exists(dirpath):
        os.remove(dirpath)
    print("解压{0}成功".format(dirpath))

def file_name(path):
    filenames = os.listdir(path)  # 获取目录下所有文件名
    for filename in filenames:
        print(filename)
        #判断是否是文件夹 如果是文件夹的继续获取下一个文件的目录
        if os.path.isdir(os.path.join(path,filename)):
            # 如果是文件的话 就继续获取的下一级文件
            newpath=os.path.join(path,filename);
            file_name(newpath)
        else:
            #解压zip文件
            if os.path.splitext(filename)[1] == '.zip':
                newpath = os.path.join(path, filename);
                unzip(newpath)


if __name__ == '__main__':
    zippath="填写你的文件的路径"
    file_name(zip)
