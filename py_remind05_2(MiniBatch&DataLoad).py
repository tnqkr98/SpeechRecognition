import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader     # 데이터로더


x_train = torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # 메모리 구조상 2의 멱수가 좋음(송수신간)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# 한번에 다돌리면 배치경사하강법, 미니배치로 돌리면 미니배치 경사하강법(쓰는이유 : 시스템 연산부담완화)
nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, data in enumerate(dataloader):
        mini_x, mini_y = data
        optimizer.zero_grad()
        h = model(mini_x)

        cost = F.mse_loss(h, mini_y)
        cost.backward()
        optimizer.step()

        print("Epoch : {:5d}/{}  Batch : {:d}/{} Cost : {:.6f}"
              .format(epoch,nb_epochs,batch_idx+1,len(dataloader),cost.item()))
