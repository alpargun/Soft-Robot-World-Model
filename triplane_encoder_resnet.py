import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTriPlaneEncoder(nn.Module):
    def __init__(self, feature_dim=32):
        super().__init__()
        
        # 1. Load the pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Extract only the early layers to preserve a 32x32 spatial resolution
        # Input 128x128 -> Conv1 (64x64) -> MaxPool (32x32) -> Layer1 (32x32)
        # Output channels at this stage: 64
        self.backbone = nn.Sequential(*list(resnet.children())[:5])
        
        # 3. Compress the 64 ResNet channels down to your target Tri-Plane feature_dim
        self.feature_compressor = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 4. YZ Plane Combiner (Fuses Side1 and Side3)
        self.yz_combiner = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_view_frames):
        """
        Inputs:
            multi_view_frames: [Batch, Views=4, C=3, H=128, W=128]
        """
        B, Views, C, H, W = multi_view_frames.shape
        
        # Flatten the batch and views dimensions to process all images simultaneously
        # This prevents running the heavy ResNet 4 separate times in a loop!
        # Shape becomes: [Batch * 4, 3, 128, 128]
        flat_frames = multi_view_frames.view(B * Views, C, H, W)
        
        # Extract features -> Shape: [Batch * 4, 64, 32, 32]
        resnet_features = self.backbone(flat_frames)
        
        # Compress channels -> Shape: [Batch * 4, feature_dim, 32, 32]
        compressed_features = self.feature_compressor(resnet_features)
        
        # Unflatten back to separated views -> Shape: [Batch, 4, feature_dim, 32, 32]
        _, F_dim, F_H, F_W = compressed_features.shape
        features = compressed_features.view(B, Views, F_dim, F_H, F_W)
        
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