import torch
from torch import nn, optim
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()


def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype = torch.float32)
    v_b = torch.zeros(1, dtype = torch.float32)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data


# d2l.train_ch7(sgd_momentum, init_momentum_states(),
#               {'lr': 0.02, 'momentum': 0.5}, features, labels)
# d2l.train_ch7(sgd_momentum, init_momentum_states(),
#               {'lr': 0.02, 'momentum': 0.9}, features, labels)

# d2l.train_ch7(sgd_momentum, init_momentum_states(),
#               {'lr': 0.004, 'momentum': 0.9}, features, labels)

d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                      features, labels)
