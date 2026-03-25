import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# Notes on conv layer behavior:
# 1. Convolution type: This uses learned convolution (NOT mean/Gaussian filtering).
#    The kernel (filter) weights are learned via backpropagation, not fixed.
#    No padding (padding=0) is applied, so this is "valid" convolution —
#    output spatial size shrinks. (Opposite of "same" convolution which pads to keep size.)
# 2. Bias: nn.Conv2d has bias=True by default, so each filter has a learnable bias term.
#    The operation per output pixel is: output = sum(input * weight) + bias

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
        # Spatial dimension formula: out = floor((in - kernel + 2*padding) / stride) + 1
        # conv1: (84 - 8) / 4 + 1 = 20  →  feature map: (B, 16, 20, 20)
        # conv2: (20 - 4) / 2 + 1 = 9   →  feature map: (B, 32,  9,  9)
        # flatten: 32 * 9 * 9 = 2592    →  vector:      (B, 2592)
        # fc1: 2592 → 256               →  vector:      (B, 256)
        # fc2: 256  → num_actions        →  Q-values:    (B, num_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84) — input is a batch of stacked grayscale frames
        x = F.relu(self.conv1(x))   # 1st conv layer: extract low-level features (edges, textures), shape → (B, 16, 20, 20)
        x = F.relu(self.conv2(x))   # 2nd conv layer: extract high-level features, shape → (B, 32, 9, 9)

        x = x.view(x.size(0), -1)   # flatten: collapse spatial dims into a 1D vector per sample, shape → (B, 2592)

        x = F.relu(self.fc1(x))     # fully-connected layer: learn combinations of features, shape → (B, 256)

        return self.fc2(x)          # output layer: one Q-value per action, no activation (Q can be negative)