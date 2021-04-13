import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root = 'Datasets/FashionMNIST', train = True, download = True,
                                                transform = transforms.ToTensor())

