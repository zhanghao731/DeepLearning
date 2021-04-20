import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob


# X = torch.arange(16).view(2, 8)
# print(X)
# print(dropout(X, 0))
# print(dropout(X, 0.5))
# print(dropout(X, 1.0))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens1)), dtype = torch.float, requires_grad = True)
b1 = torch.zeros(num_hiddens1, requires_grad = True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_hiddens2)), dtype = torch.float, requires_grad = True)
b2 = torch.zeros(num_hiddens2, requires_grad = True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens2, num_outputs)), dtype = torch.float, requires_grad = True)
b3 = torch.zeros(num_outputs, requires_grad = True)

params = [W1, b1, W2, b2, W3, b3]

drop_prob1, drop_prob2 = 0.2, 0.5

# is_training表示是否处于训练模式，评估模式不使用丢弃法
# def net(X, is_training = True):
#     X = X.view(-1, num_inputs)
#     H1 = (torch.matmul(X, W1) + b1).relu()
#     if is_training:
#         H1 = dropout(H1, drop_prob1)
#     H2 = (torch.matmul(H1, W2) + b2).relu()
#     if is_training:
#         H2 = dropout(H2, drop_prob2)
#     return torch.matmul(H2, W3) + b3


num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)

# 简洁实现
net = nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(drop_prob1),
                    nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(drop_prob2),
                    nn.Linear(num_hiddens2, num_outputs))
for param in net.parameters():
    nn.init.normal_(param, mean = 0, std = 0.01)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

