import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from utils.options import args



device = torch.device(f"cuda:{args.gpus[0]}")

# Create CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0) #output_shape=(16,26,26)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # activation
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) #output_shape=(16,24,24)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,12,12)
        # Convolution 2
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) #output_shape=(32,10,10)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU() # activation
        # Convolution 3
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0) #output_shape=(32,8,8)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,3,3)

        

        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(256 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 3 
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        # Convolution 4 
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out