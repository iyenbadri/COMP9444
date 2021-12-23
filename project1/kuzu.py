# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.NL = nn.Linear(28*28, 10)      #Initialise the layer with specified i/p and o/p parameters

    def forward(self, x):
        x=x.view(x.shape[0],-1)             # Flatten the Image to apply the linear function
        x=self.NL(x)                        # Perform Linear Function
        x= F.log_softmax(x,dim=1)           # Apply log_softmax
        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.NF1=nn.Linear(28*28,64)        # Initialise the layer with specified i/p and o/p parameters
        self.NF2=nn.Linear(64,10)

    def forward(self, x):
        x=x.view(x.shape[0],-1)             # Flatten the Image to apply the linear function
        hid_sum1= self.NF1(x)               # Implement first layer
        hid1=torch.tanh(hid_sum1)           # Apply tanh activation
        hid_sum2=self.NF2(hid1)             # Implement second layer
        hid2=torch.tanh(hid_sum2)           # Apply tanh activation
        x= F.log_softmax(hid2,dim=1)        # Apply log softmax 
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1=nn.Conv2d(1,10,5)         
        self.conv2=nn.Conv2d(10,50,5)
        self.max_pool=nn.MaxPool2d(2)       
        self.linear1=nn.Linear(800,256)     
        self.linear2=nn.Linear(256,10)

    def forward(self, x):
        x= F.relu(self.conv1(x))            # Apply relu activation on first convolutional layer
        x= self.max_pool(x)                 # Apply max_pool to simplify the input data
        x=F.relu(self.conv2(x))             # Apply relu activation on second convolutional layer
        x=self.max_pool(x)                  # Apply max_pool to simplify the input data
        x=x.view(x.size(0),-1)              # Apply view transform for input to linear function
        x=F.relu(self.linear1(x))           # Apply relu activation on fully connected layer
        x=self.linear2(x)                   # Apply linear function
        x=F.log_softmax(x,dim=1)            # Apply log_softmax
        return x
