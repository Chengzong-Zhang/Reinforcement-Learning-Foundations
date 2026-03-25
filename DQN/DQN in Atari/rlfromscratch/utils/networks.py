import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], num_actions):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()   #The first line must be called to initialize the internal parameter dictionary.
        # TODO: initialize layers according to the paper
        self.conv1 = nn.Conv2d(4,16,8,stride=4)
        #The first hidden layer convolves 16(8*8) filters with stride 4 with the input image
        self.conv2 = nn.Conv2d(16,32,4,stride=2)
        # The second hidden layer convolves 32(4*4) filters with stride 2
        self.fc1 = nn.Linear(32*9*9,256)
        #The final hidden layer is fully-connected and consists of 256 rectifier units.
        self.fc2 = nn.Linear(256,num_actions)
        )
    def forward(self, x):
        # x: (batch, 4, 84, 84)
        # TODO: input is a batch of figure, do forward propogation to calculate the Q for each action
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)