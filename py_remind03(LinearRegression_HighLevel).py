import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearRegrssionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)        # nn.Linear : 층(Layer)간 선형 결합. (Input , Output)
    def forward(self, x):
        return self.linear(x)

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

model = LinearRegrssionModel()
optimizer = optim.SGD(model.parameters(),lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))
