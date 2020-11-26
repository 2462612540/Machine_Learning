#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:xjl
# datetime:2019/9/24 20:52
# software: PyCharm
from aip import AipImageClassify

filepath="E:/Python/WordToAudio/result/audio_2.mp3"
app_id = "17334738"
api_key = "0q8ZjM8YU0sKPfh8bgIoM8z1"
secret_key = "ODOKp05YaEO7CKv604HrINA6FVR9POky"

client = AipImageClassify(app_id, api_key, secret_key)

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('../data/frist.jpg')

""" 调用食材识别 """
data=client.ingredient(image);
print(data)

""" 如果有可选参数 """
options = {}
options["top_num"] = 3

""" 带参数调用食材识别 """
client.ingredient(image, options)

if __name__ == '__main__':
    pass

