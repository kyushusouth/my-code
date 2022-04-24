from random import random
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F_c
from psutil import CONN_NONE

import torch
import torch.nn as nn
import torch.nn.functional as F_t

from variable_to_tensor import variable_to_tensor


class CNN_c(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 6, 5)
            self.conv2 = L.Convolution2D(None, 16, 5)
            self.fc1 = L.Linear(None, 120)
            self.fc2 = L.Linear(None, 64)
            self.fc3 = L.Linear(None, 10)

    def forward(self,x):
        x = F_c.max_pooling_2d(F_c.relu(self.conv1(x)), 2)
        x = F_c.max_pooling_2d(F_c.relu(self.conv2(x)), 2)
        x = F_c.relu(self.fc1(x))
        x = F_c.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_t(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*22*22, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F_t.max_pool2d(F_t.relu(self.conv1(x)), 2)
        x = F_t.max_pool2d(F_t.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F_t.relu(self.fc1(x))
        x = F_t.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    data = np.random.rand(2, 1, 100, 100).astype(np.float32)
    # data_for_chainer_model
    data = data * 10
    # data_for_pytorch_model
    data2 = torch.from_numpy(data)
    net_c = CNN_c()
    net_t = CNN_t()
    out_c = net_c(data)
    out_t = net_t(data2)
    print(f"chainer出力\n{out_c}\n")
    print(f"chainer出力、variableなし\n{out_c.data}\n")
    print(f"pytorch出力\n{out_t}\n")

    out_c = variable_to_tensor(out_c)
    print(f"{out_c}\n")
    

if __name__ == '__main__':
    main()
    
