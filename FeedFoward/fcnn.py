import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim 

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperaprams. 
input_size= 784 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64 
num_epochs = 1

# laod dada 
train_dataset = datasets.MNIST(root = "dataset/", train=True, transform = transform.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)

# laod dada 
test_dataset = datasets.MNIST(root = "dataset/", train=False, transform = transform.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the network 
model = NN(input_size=input_size, num_classes= num_classes)

# initialei hyperpparam.s
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),  lr = learning_rate)

# train dada
for epoch in range(rpochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        target = data.to(device=device)

        # orrct shape
        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



# acc dada 

def check_acc(loader, model):
    num_correct = 0 
    num_samples = 0 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x =x.reshape(x.shape[0], -1)
            scores = model(x)
            _ , pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)
            



