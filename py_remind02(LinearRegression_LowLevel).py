"""
Linear Regression 을 적당히 Low-Level 에서 구현.
Autograd(자동미분) 에 대한 적당한 이해
Tensor의 흐름 이해
"""
import torch
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)   # requires_grad: 텐서에 이루어진 연산추적.(변화도 및 연산 내역 저장)
b = torch.zeros(1, requires_grad=True)   # 일명 Autograd(자동미분)에 필요.

optimizer = optim.SGD([W, b], lr=0.01)  # 확률적경사하강법

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    hypothesis = x_train * W + b    # 예측값
    cost = torch.mean((hypothesis - y_train) ** 2)  # Low-level MSE

    optimizer.zero_grad()   # 텐서 내부적으로 누적 된 gradient를 0으로 초기화(역전파 이전에 반드시 수행해야함)
    cost.backward()         # 연산추적이 활성화 된 경우, 이 함수를 호출 시 모든 변화도(gradient=미분값)를 자동계산. dloss/dx. 이값은 텐서에 저장됨(.grad 속성에)
    optimizer.step()        # 텐서에 저장된 변화(gradient)를 기반으로 optimizer는 가중치를 update 함.

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))