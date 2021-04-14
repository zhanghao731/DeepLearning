import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as Data

mnist_train = torchvision.datasets.FashionMNIST(root = 'Datasets/FashionMNIST', train = True, download = True,
                                                transform = transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root = 'Datasets/FashionMNIST', train = False, download = True,
                                               transform = transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
feature, label = mnist_train[0]
print(feature.shape, label)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))
    for f, img, lbl in zip(figs, images, labels):
        # 转化为图像
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        # 隐藏坐标轴
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = Data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size = batch_size, shuffle = True, num_workers = num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
