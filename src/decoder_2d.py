import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class Decoder2D(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Phase 1: Upsample to 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, 32, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            ResBlock(32), # Sharpens the blur natively without norms

            # Phase 2: Upsample to 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            ResBlock(16), # Sharpens the final edge

            # Phase 3: Output Projection
            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)