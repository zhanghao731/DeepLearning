import time

import torch

assert torch.cuda.device_count() >= 2


class Benchmark:
    def __init__(self, prefix = None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("%stime: %.4f sec" % (self.prefix, time.time() - self.start))


def run(x):
    for _ in range(20000):
        y = torch.mm(x, x)


x_gpu1 = torch.rand(size = (100, 100), device = 'cuda:0')
x_gpu2 = torch.rand(size = (100, 100), device = 'cuda:1')

with Benchmark('Run on GPU1.'):
    run(x_gpu1)
    torch.cuda.synchronize()
with Benchmark('Then run on GPU2.'):
    run(x_gpu2)
    torch.cuda.synchronize()

with Benchmark('Run on both GPU1 and GPU2 in parallel.'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
