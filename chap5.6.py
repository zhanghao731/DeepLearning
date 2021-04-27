import time
import torch
from torch import nn, optim
import torchvision
import d2lzh_pytorch as d2l
import torch.utils.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, 2),
                                  nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(), nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
                                  nn.MaxPool2d(3, 2))
        self.fc = nn.Sequential(nn.Linear(256 * 5 * 5, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096),
                                nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(feature.shape[0], -1))
        return output


# print(torch.__version__)
net = AlexNet()


# print(net)
def load_data_fashion_mnist(batch_size, resize = None, root = '~/Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = root, train = True, download = True, transform = transform)
    mnist_test = torchvision.datasets.FashionMNIST(root = root, train = False, download = True, transform = transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = 4)
    return train_iter, test_iter


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize = 224)
lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr = lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
