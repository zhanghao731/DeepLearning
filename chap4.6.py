import torch
from torch import nn

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
x = torch.tensor([1, 2, 3])
# print(x)
x = x.cuda(0)
# print(x)
# print(x.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device = device)
# print(x)
y = x ** 2
# print(y)
# z = y + x.cpu()
net = nn.Linear(3, 1)
# print(list((net.parameters()))[0].device)
net.cuda()
print(list((net.parameters()))[0].device)
x = torch.rand(2, 3).cuda()
print(net(x))
