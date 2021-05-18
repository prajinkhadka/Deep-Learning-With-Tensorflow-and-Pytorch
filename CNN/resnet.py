import torch 
import torchvision.datasets as datasets  
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader  

class block(nn.Module):
    def __init__(self,in_channels, out_channels, identity_downsample = None, stride=1):
        super(block, self).__init__() 
        self.expansion = 4 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.RelU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity =x 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample not None:
            identity = self.identity_downsample(identiy)

        x += identity
        x = self.relu(x)

        return x 


class ResNet(nn.Module):
    def __init__(self, block,layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels= 64 
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu= nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # ResnetLatyers 
        

    def _make_layer(self, block, num_of_residual_blocks, out_channels, stride):
        identity_downsample = None 
        layers = []
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=1)
                nn.BatchNorm2d(out_channels*4)
            )
            

