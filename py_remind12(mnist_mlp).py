import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version :', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32         # 보통 파이썬 내 하이퍼 파라미터는 영대문자로.
EPOCHS = 10

train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 미니배치 하나 확인  ( batch-size, channel, x, y ) -> channel : 1 이므로 그레이스케일
for (X_train, Y_train) in train_loader:
    print('X_train : ', X_train.size(), 'type:', X_train.type())
    print('Y_train : ', Y_train.size(), 'type:', Y_train.type())

    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap="gray_r")
        plt.title('Class: ' + str(Y_train[i].item()))
    break

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)   # fully connected layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forword(self, x):
        x = x.view(-1,28*28)    # 2차원 -> 1차원 (Flatten 펼친다)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)     # gradient 원활하게 계산
        return x

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):




model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()       # output ( one-hot e

print(model)






