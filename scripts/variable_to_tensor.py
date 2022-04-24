import numpy as np
import torch


def variable_to_tensor(x):
    # print("variableをtensorに変換")
    data = x.data
    data = np.array(data)
    data = torch.from_numpy(data).float()
    return data