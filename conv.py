import torch
from torch import Tensor
import torch.nn as nn


class DynamiConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(DynamiConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.ones(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        self.dweight = nn.Parameter(torch.ones(1, self.in_channel, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.ones(self.out_channel))
        self.isbias = bias
        
        self.Unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), 
                                stride=self.stride, padding=self.padding) 
        
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dweight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
             
        batch_size, c, h, w = x.shape
        x = self.Unfold(x)
        x = self.weight.view(self.out_channel, -1) @ (x * (self.dweight.view(1, -1) @ torch.sigmoid(x)))
        if self.isbias :
            x = x + self.bias[None,:,None]   
        if self.stride == 2 :
            h //= 2
            w //= 2
        x = x.view(batch_size, self.out_channel, h, w)

        return x



class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(CustomConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.ones(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.ones(self.out_channel))
        self.isbias = bias
        
        self.Unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), 
                                stride=self.stride, padding=self.padding) 
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):     
        
        batch_size, c, h, w = x.shape
        if self.stride == 2 :
            h //= 2
            w //= 2

        if self.isbias :
            x = self.Unfold(x)
            x = self.weight.view(self.out_channel, -1) @ x + self.bias[None,:,None]      
        else :
            x = self.Unfold(x)
            x = self.weight.view(self.out_channel, -1) @ x
        x = x.view(batch_size, self.out_channel, h, w)

        return x