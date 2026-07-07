import torch
import torch.nn as nn

class Encoder2D(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)