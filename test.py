import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.activation(self.bn(y))
        return y


if __name__ == '__main__':
    print("Hello World")
    
    in_channels = 8
    out_channels = 8
    kernel_size = 3

    convnet = Conv(in_channels, out_channels, kernel_size)
    x = torch.rand(2, 8, 5)
    out = convnet(x)
    print(out)
    
