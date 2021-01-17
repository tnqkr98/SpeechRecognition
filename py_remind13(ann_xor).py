import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)

# 네트워크 레이어를 설정한다. Input = 2, Output = 1 , 이항 분류이므로 마지막에 Sigmoid !
linear = torch.nn.Linear(2,1,bias=True)
sigmoid = torch.nn.Sigmoid()

# 모델을 설정한다 (레이어를 이어붙여놓으면 그게 모델)
model = torch.nn.Sequential(linear, sigmoid).to(device)

# 손실함수와 최적화함수를 정한다
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1)

for step in range(10001):
    optimizer.zero_grad()
    h = model(X)

    cost = criterion(h,Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    h = model(X)
    predicted = (h>0.5).float()
    acc = (predicted == Y).float().mean()
    print('\nHypothesis: ', h.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(), '\nAccuracy: ', acc.item())