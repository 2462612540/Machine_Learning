import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer

img = Image.open(r'D:\softwaresavfile\Github\machine_learning\计算视觉基础\test.jpg')
with fluid.dygraph.guard():
    # 设置卷积核参数
    w = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype='float32')/8
    w = w.reshape([1, 1, 3, 3])
    # 由于输入通道数是3，将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
    w = np.repeat(w, 3, axis=1)
    # 创建卷积算子，输出通道数为1，卷积核大小为3x3，
    # 并使用上面的设置好的数值作为卷积核权重的初始化参数
    conv = Conv2D(num_channels=3, num_filters=1, filter_size=[3, 3],
            param_attr=fluid.ParamAttr(
              initializer=NumpyArrayInitializer(value=w)))

    # 将读入的图片转化为float32类型的numpy.ndarray
    x = np.array(img).astype('float32')
    # 图片读入成ndarry时，形状是[H, W, 3]，
    # 将通道这一维度调整到最前面
    x = np.transpose(x, (2,0,1))

    print("图像的高度和宽度=",img.height, img.width)
    # 将数据形状调整为[N, C, H, W]格式
    x = x.reshape(1, 3, img.height, img.width)
    x = fluid.dygraph.to_variable(x)
    y = conv(x)
    out = y.numpy()

plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img)
f = plt.subplot(122)
f.set_title('output feature map', fontsize=15)
plt.imshow(out.squeeze(), cmap='gray')
plt.show()