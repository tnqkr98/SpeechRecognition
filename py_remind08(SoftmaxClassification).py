import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

z = torch.FloatTensor([1,2,3])
h = F.softmax(z, dim=0)     # 1차원 벡터에 대해, dim=0
print(h)

z = torch.rand(3,5, requires_grad=True)     # 3x5 짜리 랜덤값 행렬
h = F.softmax(z, dim=1)     # 2차원 행렬에 대해, 행에 대한 softmax
print(h)
