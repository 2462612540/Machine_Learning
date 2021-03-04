#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: 1.0
@author: xjl
@file: infer_model.py
@time: 2021/3/4 11:01
"""
import numpy as np
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

model_save_dir = r"D:\softwaresavfile\Github\machine_learning\手写数学的识别\mode_save\__model__"


# 图片预处理
def load_image(file):
    im = Image.open(file).convert('L')  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    im = im.resize((28, 28), Image.ANTIALIAS)  # resize image with high-quality 图像大小为28*28
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    # print(im)
    im = im / 255.0 * 2.0 - 1.0  # 归一化到【-1~1】之间
    return im


# 预测函数
def predict(infer_path):
    # 查询原来的图像
    # infer_path = '预测文件的路径'
    img = Image.open(infer_path)
    plt.imshow(img)  # 根据数组绘制图像
    plt.show()  # 显示图像

    # 创建Executer
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    # 加载数据并开始预测
    plt.imshow(img)  # 根据数组绘制图像
    plt.show()  # 显示图像

    with fluid.scope_guard(inference_scope):
        # 获取训练好的模型
        # 从指定目录中加载 推理model(inference model)
        [inference_program,  # 推理Program
         feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
         fetch_targets] = fluid.io.load_inference_model(model_save_dir,
                                                        # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                        infer_exe)  # infer_exe: 运行 inference model的 executor

        img = load_image(infer_path)

        results = infer_exe.run(program=inference_program,  # 运行推测程序
                                feed={feed_target_names[0]: img},  # 喂入要预测的img
                                fetch_list=fetch_targets)  # 得到推测结果,
        # 获取概率最大的label
        lab = np.argsort(results)  # argsort函数返回的是result数组值从小到大的索引值

        print("该图片的预测结果的label为: %d" % lab[0][0][-1])  # -1代表读取数组中倒数第一列


if __name__ == '__main__':
    predict()
