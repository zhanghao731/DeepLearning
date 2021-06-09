import sys
import time

import torch
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader

import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('img/wallhaven-3z32j3_500x400.png')


# d2l.plt.imshow(img)
# d2l.plt.show()


def show_images(imgs, num_rows, num_cols, scale = 2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize = figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_rows + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    d2l.plt.show()
    return axes


def apply(img, aug, num_rows = 2, num_cols = 4, scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


# apply(img, torchvision.transforms.RandomHorizontalFlip())
# apply(img, torchvision.transforms.RandomVerticalFlip())
shape_aug = torchvision.transforms.RandomResizedCrop(200, (0.1, 1), ratio = (0.5, 2))
# apply(img, shape_aug)
# apply(img, torchvision.transforms.ColorJitter(brightness = 0.5))
# apply(img, torchvision.transforms.ColorJitter(hue = 0.5))
# apply(img, torchvision.transforms.ColorJitter(contrast = 0.5))
color_aug = torchvision.transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)
# apply(img, color_aug)
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)
all_images = torchvision.datasets.CIFAR10(train = True, root = 'Datasets/CIFAR', download = True)
# show_images([all_images[i][0] for i in range(32)], num_rows = 4, num_cols = 8, scale = 0.8)
flip_aug = torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
no_aug = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
num_workers = 0 if sys.platform.startswith('win32') else 4


def load_cifar10(is_train, augs, batch_size, root = 'Datasets/CIFAR'):
    dataset = torchvision.datasets.CIFAR10(root = root, train = is_train, transform = augs, download = True)
    return DataLoader(dataset, shuffle = is_train, batch_size = batch_size, num_workers = num_workers)


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (
            epoch, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_with_data_aug(train_augs, test_augs, lr = 0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = optim.Adam(net.parameters(), lr = lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs = 10)


train_with_data_aug(flip_aug, no_aug)

