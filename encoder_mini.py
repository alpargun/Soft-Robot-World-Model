import torch
import torch.nn as nn

# ==========================================
# --- 1. THE CUSTOM MICRO-RESNET BLOCK ---
# ==========================================
# This MUST remain named MicroBasicBlock!
class MicroBasicBlock(nn.Module):
    """
    A lightweight Residual Block that uses GroupNorm instead of BatchNorm.
    This makes the network 100% immune to small batch sizes during inference.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        # bias=False is standard when a normalization layer immediately follows
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        # The Skip Connection (Shortcut)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)
        return out


# ==========================================
# --- 2. THE MAIN ENCODER CLASS ---
# ==========================================
# This matches your train.py import perfectly!
class MiniResNetTriPlaneEncoder(nn.Module):
    def __init__(self, feature_dim=32):
        super().__init__()
        
        # 1. Micro-ResNet Stem
        # Input 128x128 -> Conv1 (64x64) -> MaxPool (32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 2. Lightweight Residual Layers
        # Expands channels from 16 -> 32 -> 64 while maintaining 32x32 spatial resolution
        self.layer1 = MicroBasicBlock(16, 32, stride=1, groups=8)
        self.layer2 = MicroBasicBlock(32, 64, stride=1, groups=8)
        
        # 3. Compress the 64 ResNet channels down to your target Tri-Plane feature_dim
        self.feature_compressor = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 4. YZ Plane Combiner (Fuses Side1 and Side3)
        self.yz_combiner = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_view_frames):
        """
        Inputs:
            multi_view_frames: [Batch, Views=4, C=3, H=128, W=128]
        """
        B, Views, C, H, W = multi_view_frames.shape
        
        # Flatten the batch and views dimensions to process all images simultaneously
        # Shape becomes: [Batch * 4, 3, 128, 128]
        flat_frames = multi_view_frames.reshape(B * Views, C, H, W)
        
        # Forward pass through the custom Micro-ResNet -> Shape: [Batch * 4, 64, 32, 32]
        x = self.stem(flat_frames)
        x = self.layer1(x)
        resnet_features = self.layer2(x)
        
        # Compress channels -> Shape: [Batch * 4, feature_dim, 32, 32]
        compressed_features = self.feature_compressor(resnet_features)
        
        # Unflatten back to separated views -> Shape: [Batch, 4, feature_dim, 32, 32]
        _, F_dim, F_H, F_W = compressed_features.shape
        features = compressed_features.reshape(B, Views, F_dim, F_H, F_W)
        
        # Split into the specific camera views
        f_side1 = features[:, 0] # Looks at YZ
        f_side2 = features[:, 1] # Looks at XZ
        f_side3 = features[:, 2] # Looks at YZ
        f_top   = features[:, 3] # Looks at XY
        
        # Map to Tri-Planes
        plane_xy = f_top
        plane_xz = f_side2
        
        # Concatenate and fuse the opposing YZ views
        f_yz_cat = torch.cat([f_side1, f_side3], dim=1) 
        plane_yz = self.yz_combiner(f_yz_cat)           

        return {
            "xy": plane_xy,
            "xz": plane_xz,
            "yz": plane_yz
        }