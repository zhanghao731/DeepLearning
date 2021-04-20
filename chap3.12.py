import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import d2lzh_pytorch as d2l
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
# 噪声
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype = torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def init_params():
    w = torch.randn((num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    return [w, b]


def l2_penalty(w):
    return (w ** 2).sum() / 2


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,
                 ['train', 'test'])
    d2l.plt.show()
    print('L2 form of w:', w.norm().item())


# 过拟合
# fit_and_plot(lambd = 0)
# 权重衰减
# fit_and_plot(lambd = 3)

# 简洁实现
def fit_and_plot_pytorch(wd):
    # 对权重参数衰减
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean = 0, std = 1)
    nn.init.normal_(net.bias, mean = 0, std = 0.1)
    optimizer_w = torch.optim.SGD(params = [net.weight], lr = lr, weight_decay = wd)
    optimizer_b = torch.optim.SGD(params = [net.bias], lr = lr)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,
                 ['train', 'test'])
    d2l.plt.show()
    print('L2 form of w:', net.weight.data.norm().item())


# fit_and_plot_pytorch(wd = 0)
fit_and_plot_pytorch(wd = 3)
