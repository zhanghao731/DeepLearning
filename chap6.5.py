import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256
rnn_layer = nn.RNN(input_size = vocab_size, hidden_size = num_hiddens)


# num_steps = 35
# batch_size = 2
# state = None
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y, state_new = rnn_layer(X, state)
# print(Y.shape, len(state_new), state_new[0].shape)
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs:(batch,seq_len)
        # 获取one-hot向量表示
        X = d2l.to_onehot(inputs, self.vocab_size)  # list
        Y, self.state = self.rnn(torch.stack(X), state)
        outout = self.dense(Y.view(-1, Y.shape[-1]))
        return outout, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # 记录prefix和输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device = device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(1).item()))
    return ''.join([idx_to_char[i] for i in output])


model = RNNModel(rnn_layer, vocab_size).to(device)


# print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                  prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样|
        for X, Y in data_iter:
            if state is not None:
                # 使用detach，防止梯度计算开销太大
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            # 困惑度
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))


num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                              num_epochs, num_epochs, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)

