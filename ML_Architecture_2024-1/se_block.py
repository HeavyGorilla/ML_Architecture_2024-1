import torch
import torch.nn as nn
from config import config


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = x.view(batch_size, num_channels, -1).mean(dim=2)
        excitation = nn.ReLU(inplace=True)(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch_size, num_channels, 1, 1)
        return x * excitation
