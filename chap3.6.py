import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

# print(torch.__version__)
# print(torchvision.__version__)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float)
b = torch.zeros(num_outputs, dtype = torch.float)
W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)


# print(b)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / partition  # 广播机制


# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim = 1))

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])


# print(y_hat.gather(1, y.view(-1, 1)))


def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim = 1) == y).float().mean().item()


# print(accuracy(y_hat, y))


def evaluate_accuracy(data_iter, net):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        acc_num += (net(X).argmax(dim = 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_num / n


# print(evaluate_accuracy(test_iter, net))

num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr = None, optimizer = None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
