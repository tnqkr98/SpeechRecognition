import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = LinearRegressionModel()  # 가중치와 바이어스를 따로 설정해줄 필요가 없다
                                 # 그런데 언젠가 가중치 초기화 관련해서 어차피 따로 설정해야함
optimizer = optim.SGD(model.parameters(),lr=1e-5)
num_epoch = 25
for epoch in range(num_epoch+1):
    h = model(x_train)
    c = F.mse_loss(h,y_train)

    optimizer.zero_grad()
    c.backward()
    optimizer.step()

    print('Epoch : {:4d}/{} Cost : {:.6f}'.format(epoch,num_epoch,c.item()))