"""
다중 분류 비용함수 관련, torch 리팩토링 과정
"""
import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1,2,3])
h = F.softmax(z, dim=0)     # 1차원 벡터에 대해, dim=0
print(h)

# ---- Muli Class classification Cross Entropy Function ----- #

z = torch.rand(3,5, requires_grad=True)     # 3x5 짜리 랜덤값 행렬
h = F.softmax(z, dim=1)     # 2차원 행렬에 대해, 행에 대한 softmax
print(h)

y = torch.randint(5, (3,)).long()         # randint([시작값=0], 끝값, (행렬형태))
print(y)
y_one_hot = torch.zeros_like(h)           # 0으로 채운 3x5 행렬
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # 위 y벡터를 행렬에 one-hot 인코딩.(참값)
print(y_one_hot)

cost = (y_one_hot * -torch.log(h)).sum(dim=1).mean()    # low-level
print(cost)

# F.log_softmax === torch.log(F.softmax(z,dim=1))
cost = F.nll_loss(F.log_softmax(z,dim=1),y)     # high-level
print(cost)

cost = F.cross_entropy(z, y)             # Super high-level (다중분류 비용함수)
print(cost)

