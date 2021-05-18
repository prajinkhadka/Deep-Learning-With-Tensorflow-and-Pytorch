import torch 
import torchvision.datasets as datasets  
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader  

# LeNEt Architecture. 
# 1*32*32 Input -> (5*5), s=1,p=0 -> avg pool -> (5*5), s=1, p=0 -> avg pool 
# -> conv ( 5*5) to 120 channels -> linear 84 -> Linear 10 


class leNet(nn.Module):
    def __init__(self):
        super(leNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120)
        nn.linear1= nn.Linear(120, 84)
        nn.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x)) 
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x)) #  num_examples * 120 * 1 * 1 ->> num_exmaples*120 
        x = x.reshape(x.shape[0], -1)
        x=  self.relu(self.linear1(x))
        x = self.linear2(x)

        return x 


x = torch.randn(1, 64, 32, 32)
model = leNet()
print(model(x).shape)