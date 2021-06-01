import numpy as np
import torch
import time
from torch import nn, optim
import d2lzh_pytorch as d2l
import torch.utils.data


def get_data_ch7():
    data = np.genfromtxt('Datasets/airfoil_self_noise.dat', delimiter = '\t')
    data = (data - data.mean(axis = 0)) / data.std(axis = 0)
    return torch.tensor(data[:1500, :-1], dtype = torch.float32), torch.tensor(data[:1500, -1], dtype = torch.float32)


features, labels = get_data_ch7()

# print(features.shape)
print(labels.shape)


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def train_ch7(optimizer_fn, states, hyperparams, features, labels, batch_size = 10, num_epochs = 2):
    net, loss = d2l.linreg, d2l.squared_loss
    w = torch.nn.Parameter(
        torch.tensor(np.random.normal(0, 0.01, size = (features.shape[1], 1)), dtype = torch.float32),
        requires_grad = True)
    b = torch.nn.Parameter(torch.zeros(1, dtype = torch.float32), requires_grad = True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size,
                                            shuffle = True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            optimizer_fn([w, b], states, hyperparams)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


def train_sgd(lr, batch_size, num_epochs = 2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


# train_sgd(1, 1500, 6)
# train_sgd(0.005, 1)
# train_sgd(0.05, 10)

def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels, batch_size = 10, num_epochs = 2):
    net = nn.Sequential(nn.Linear(features.shape[-1], 1))
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size,
                                            shuffle = True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y) / 2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


train_pytorch_ch7(optim.SGD, {'lr': 0.05}, features, labels, 10)
