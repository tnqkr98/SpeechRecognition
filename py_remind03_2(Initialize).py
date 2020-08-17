import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 1200
# 현실성을 위해 노이즈 생성 (가우시안 노이즈) , 평균 0 (디폴트) ,분산 1
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)

# 데이터 로드
# 1000x1 텐서에 -10~10 사이의 수를 균등하게 분배
x = init.uniform_(torch.Tensor(num_data,1),-10, 10) #
y = 2*(x+noise) + 3

# 2x+3 을 근사하는 모델
model = nn.Linear(1, 1)
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(num_epoch):
    optimizer.zero_grad()
    h = model(x)

    loss = loss_func(y, h)
    loss.backward()
    optimizer.step()

    if i % 10 == 0 :
        param = list(model.parameters())
        w = param[0].item()
        b = param[1].item()
        print(loss.data, w, b)
