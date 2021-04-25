import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding = 1)
X = torch.rand(8, 8)
# print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (5, 3), padding = (2, 1))
# print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size = 3, padding = 1, stride = 2)
# print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size = (3, 5), padding = (0, 1), stride = (3, 4))
print(comp_conv2d(conv2d, X).shape)
