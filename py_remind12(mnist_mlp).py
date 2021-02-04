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

BATCH_SIZE = 32         # 보통 파이썬 내 하이퍼 파라미터는 영대문자로. <클린 코드>
EPOCHS = 10

train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential