import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Model

model = Model.DQN(84, 4)

input = torch.randn([16, 84, 84, 4])

print(model(input))
