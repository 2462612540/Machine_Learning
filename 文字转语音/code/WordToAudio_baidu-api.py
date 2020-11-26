#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xjl
# datetime:2019/9/24 14:39
# software: PyCharm
from aip import AipSpeech

"""
    实现百度的API 的接口，来讲文字转化为语音。 其中参数：APPID AK SK  这几个参数需要在百度云中获取
"""
def wordtoaduo():
    filepath="E:/Python/WordToAudio/result/audio_man/audio_16.mp3"
    app_id = "17333417"
    api_key = "EOWp6OWQDsaDR17Zw9X8M2Ee"
    secret_key = "CeL3Yc7gV3q16IZ2ANFR94uihU3uwRYe "
    client = AipSpeech(app_id, api_key, secret_key)
    result = client.synthesis("垃圾不达标请分拣非厨余垃圾。", "zh", 1, {
        "vol": 10,
        "spd": 4,
        "pit": 5,
        "per": 1,
    })
    with open(filepath, "wb") as f:
        f.write(result)


if __name__ == '__main__':
    wordtoaduo()
