import torch
import random
import zipfile

with zipfile.ZipFile('Datasets/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)


# print(vocab_size)
# corpus_indices = [char_to_idx[char] for char in corpus_chars]
# sample = corpus_indices[:20]
# print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
# print('indices:', sample)


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device = None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype = torch.float32, device = device), torch.tensor(Y, dtype = torch.float32,
                                                                                    device = device)


my_seq = list(range(30))


# for X, Y in data_iter_random(my_seq, num_steps = 6, batch_size = 2):
#     print('X: ', X, '\nY: ', Y, '\n')
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype = torch.float32, device = device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[:batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i + num_steps]
        Y = indices[:, i + 1:i + num_steps+1]
        yield X, Y
for X,Y in data_iter_consecutive(my_seq,batch_size = 2,num_steps = 6):
    print('X: ', X, '\nY: ', Y, '\n')
