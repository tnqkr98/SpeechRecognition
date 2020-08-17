import torch

x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
y = x**2
y_sum = y.sum()
y_sum.backward()    # 텐서가 하나의 스칼라 값을 가질때(즉 최종 연산이 끝났다고 볼때), backword() 수행 가능
                    # 연산 그래프의 리프노드(x)의 미분값을 구함. dy/dx (x=2), dy/dx (x=3)
print(x.grad)       #   4, 6


#z = 2*y + 3

#t = torch.tensor([3.0, 4.0])

#z_sum = z.sum()
#z_sum.backward()

