import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

norm_x_train = (x_train - x_train.mean(dim=0))/x_train.std(dim=0)
print(norm_x_train)
print(x_train)
model = MultivariateLinearRegressionModel()

#optimizer = optim.SGD(model.parameters(), lr=0.0000001)
#train(model, optimizer, x_train, y_train)
#print('------')

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)   # 노멀라이즈 한게 더 빠르게 수렴하는편.
train(model, optimizer, norm_x_train, y_train)