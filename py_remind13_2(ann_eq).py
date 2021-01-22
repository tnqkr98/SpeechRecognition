import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

num_data = 1000
num_epoch = 10000

noise = init.normal_(torch.FloatTensor(num_data,1),std=1)  # 1000x1 짜리 실수랜덤 텐서
x = init.uniform_(torch.Tensor(num_data,1),-15,15)
y = (x**2) + 3        # validation data
y_noise = y +noise    # training data

model = nn.Sequential(
    nn.Linear(1,6),
    nn.ReLU(),
    nn.Linear(6,10),
    nn.ReLU(),
    nn.Linear(10,6),
    nn.ReLU(),
    nn.Linear(6,1)
)

loss_fun = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.0002)

loss_array = []
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_fun(output,y)
    loss.backward()
    optimizer.step()

    loss_array.append(loss)

plt.plot(loss_array)
plt.show()