import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, imageDimensions, numActions):
        super(DQN, self).__init__()
        self.c1 = nn.Conv2d(imageDimensions, 16, 4)
        self.c2 = nn.Conv2d(16, 32, 4, 2)
        self.h1 = nn.Linear(32, 256)
        self.h2 = nn.Linear(256, numActions)
    def forward(self, input):
        input = F.relu(self.c1(input))
        input = F.relu(self.c2(input))
        input = F.relu(self.h1(input))
        actionQValues = self.h2(input)
        return actionQValues
