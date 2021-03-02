import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer

# 读入图片并转成numpy.ndarray
img = Image.open(r'D:\softwaresavfile\Github\machine_learning\计算视觉基础\test2.jpg').convert('L')
img = np.array(img)

# 换成灰度图
with fluid.dygraph.guard():
    # 创建初始化参数
    w = np.ones([1, 1, 5, 5], dtype = 'float32')/25
    conv = Conv2D(num_channels=1, num_filters=1, filter_size=[5, 5],
                  param_attr=fluid.ParamAttr(
                      initializer=NumpyArrayInitializer(value=w)))

    x = img.astype('float32')
    x = x.reshape(1,1,img.shape[0], img.shape[1])
    x = fluid.dygraph.to_variable(x)
    y = conv(x)
    out = y.numpy()

plt.figure(figsize=(20, 12))
f = plt.subplot(121)
f.set_title('input image')
plt.imshow(img, cmap='gray')

f = plt.subplot(122)
f.set_title('output feature map')
out = out.squeeze()
plt.imshow(out, cmap='gray')

plt.show()