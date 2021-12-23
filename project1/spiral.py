# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
#import torch.nn.functional as F

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.NF1=nn.Linear(2,2)
        self.NF2=nn.Linear(2,1)
        # INSERT CODE HERE

    def forward(self, input):
        x= input[:,0]
        y= input[:,1]
        r= np.sqrt(x*x + y*y)
        a=torch.atan2(y,x)
        input= torch.stack([r,a], axis=-1)
        hid_sum1= self.NF1(input)
        hid1=torch.tanh(hid_sum1)
        hid_sum2=self.NF2(hid1)
        output=torch.sigmoid(hid_sum2)
        return output

class RawNet(torch.nn.Module):
    
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.NF1=nn.Linear(2,2)
        self.NF2=nn.Linear(2,2)
        self.NF3=nn.Linear(2,1)
        # INSERT CODE HERE

    def forward(self, input):
        hid_sum1= self.NF1(input)
        hid1=torch.tanh(hid_sum1)
        hid_sum2=self.NF2(hid1)
        hid2=torch.tanh(hid_sum2)
        hid_sum3=self.NF3(hid2)
        output = torch.sigmoid(hid_sum3)
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
