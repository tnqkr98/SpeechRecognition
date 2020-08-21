import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):   # 오버라이드 할 함수 3개
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  def __len__(self):
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 25
for epoch in range(nb_epochs+1):
    for batch_idx, data in enumerate(dataloader):
        mini_x, mini_y = data
        h = model(mini_x)

        optimizer.zero_grad()
        cost = F.mse_loss(h, mini_y)
        cost.backward()
        optimizer.step()

        print("epoch : {:3d}/{} batch : {:3d}/{} cost : {:.6f}"
              .format(epoch, nb_epochs, batch_idx, len(dataloader), cost.item()))