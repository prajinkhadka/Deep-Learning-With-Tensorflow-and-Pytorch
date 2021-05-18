import torch 
import torchvision.datasets as datasets  
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader  
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding= 1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2) )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001 
in_channels  =1 
num_classes = 10 
batch_size = 64 
epochs = 5 


# load dataset 
train_dataset = datasets.MNIST(root='dataset/', train=True, transform= transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size= batch_size)


model = CNN(in_channels=in_channels, num_classes=num_classes)
model.to(device)

# Loss and Optimier function 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
writer = SummaryWriter(f'runs/MNIST/traingout_tensorboard')

step = 0 
for epoch in epochs:
    losses = []
    accuracies = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.apppend(loss) 

        #backaward 
        optimizer.zero_grad()
        loss.backaward()
        optimizer.step() 

        # train accuracies
        _, predictions = scores.max(1) 
        num_corret = (predictions == targets).sum() 
        run_train_acc = float(num_corret/float(data.shape[0]))

        writer.add_scalar('Training Loss', loss, global_step=step)
        writer.add_scalar('rain Acc', run_train_acc, global_step=step)
        step +=1 
