import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)        # 일종의 멤버 변수 선언 후, 객체 할당.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))     # 포워드에서 멤버변수 사용.

torch.manual_seed(1)

xy = np.loadtxt('./dataset/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

W = torch.zeros([8,1],requires_grad=True)
b = torch.zeros(1,requires_grad=True)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(),lr=1)

nb_epochs = 120
for epoch in range(nb_epochs+1):
    h = model(x_train)
    cost = F.binary_cross_entropy(h,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%10 == 0:
        prediction = h >= torch.FloatTensor([0.5])
        correct = prediction.float() == y_train
        accuracy = correct.sum().item() / len(correct)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))