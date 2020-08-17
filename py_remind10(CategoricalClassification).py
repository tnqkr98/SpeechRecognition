import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

xy = np.loadtxt('./dataset/data-04-zoo.csv', delimiter=',', dtype=np.float32)
                                           # n은 데이터 수
x_train = torch.FloatTensor(xy[:, 0:-1])   # n x feature 종류수
y_train = torch.LongTensor(xy[:, -1])
#print(x_train)
print(y_train)

# 원핫인코딩 흐름
# 1. 1차원 텐서 준비(정수형)
# 2. (1차원 텐서의 길이 x 클래스 개수) 짜리 빈껍데기(zero) 텐서 만듬
# 3. 위에서 만든 텐서의 메서드 scatter로 원핫인코딩 완성 ( 1, unsqueeze(1), 1) 매개변수 국룰

nb_classes = 7  # 클래스의 개수
y_one_hot = torch.zeros((len(y_train), nb_classes))  # n x 7 의 빈껍데기 텐서
y_one_hot = y_one_hot.scatter(1,y_train.unsqueeze(1),1)  # 최종 one-hot 인코딩(이것이 정답인것)
print(y_one_hot)

print(x_train.shape)

W = torch.zeros((16,7),requires_grad=True)      #  1 layer 이므로  16(feature) x 7(class)
b = torch.zeros(1,requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # Cost 계산 (2)
    z = x_train.matmul(W) + b   # 일종의 예측
    cost = F.cross_entropy(z, y_train)      # 내부적으로 softmax 포함

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))