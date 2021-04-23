import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型向前计算，根据输入x返回模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)


# net = MLP()
# print(net)
# print(net(X))
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)  # 将module添加进self._modules
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个OrderedDict，保证顺序
        for module in self._modules.values():
            input = module(input)
        return input


# net = MySequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net)
# print(net(X))
# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10))
# print(net[-1])
# print(net)
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]


# net1 = Module_ModuleList()
# print(net1)
# for p in net1.parameters():
#     print(p)
# net2 = Module_List()
# # print(net2)
# for p in net2.parameters():
#     print(p)
# net = nn.ModuleDict({'linear': nn.Linear(784, 256), 'act': nn.ReLU()})
# net['output'] = nn.Linear(256, 10)
# print(net['linear'])
# print(net.output)
# print(net)
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad = False)  # 不可训练参数，常数参数
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# X = torch.rand(2, 20)
# net = FancyMLP()
# print(net)
# print(net(X))
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))
