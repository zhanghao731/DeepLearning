import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


def one_hot(x, n_class, dtype = torch.float32):
    # X shape:(batch),output shape:(batch,n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype = dtype, device = x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


x = torch.tensor([0, 2])


# print(one_hot(x, vocab_size))
def to_onehot(X, n_class):
    # X shape:(batch,seq_len),output:seq_len elements of (batch,n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


X = torch.arange(10).view(2, 5)
# inputs = to_onehot(X, vocab_size)
# print(len(inputs), inputs[0].shape)
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size = shape), device = device, dtype = torch.float32)
        return torch.nn.Parameter(ts, requires_grad = True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device = device, requires_grad = True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device = device, requires_grad = True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device = device),)


def rnn(inputs, state, params):
    # input output shape:(num_steps,batch,vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)


# print(len(outputs), outputs[0].shape, state_new[0].shape)
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char,
                char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(len(prefix) + num_chars - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device = device), vocab_size)
        # 计算输出和更新隐藏状态
        Y, state = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim = 1).item()))
    return ''.join([idx_to_char[i] for i in output])


# print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
# 梯度裁剪
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device = device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pre_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:  # 如果采用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 随机采样，在每个小批量更新初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    s.detach_()
            inputs = to_onehot(X, vocab_size)
            outputs, state = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim = 0)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(outputs, y.long())
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            d2l.sgd(params, lr, 1)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        if (epoch + 1) % pred_period == 0:
            print('epoch %d,perplexity %f,time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn(prefix, pre_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, device,
                                       idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
