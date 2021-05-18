import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shape -> N, 1, 28,28
# IN case of RNN, there are 28 sequeces and each sequence have 28 features.

input_size = 28  
sequence_length = 28 
num_layers = 2 
hidden_size 256 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64 
num_epochs = 2 


# RNN epxect the size to be N * 28*28 not, N*1*28*28, to remove 1, just remove do unsqueeze in that dimension.

class RNN(nn.Module):
    def __init(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size =hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_size=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
         
        
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out 


class GRU(nn.Module):
    def __init(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size =hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_size=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
         

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out 


class LSTM(nn.Module):
    def __init(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size =hidden_size
        self.num_layers = num_layers
        self.lstm = nn.lstm(input_size, hidden_size, num_layers, batch_size=True)


        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

        # Instad of using information from every hidden state, we  can use inforamtion form the last state. 
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        
        out, _ = self.rnn(x, (h0, c0))

        out = out.reshape(out.shape[0], -1)


        out = self.fc(out)
        
        # all minibatch, last hidden state, alll features 

        out = self.fc(out[]:, -1,:])

        return out