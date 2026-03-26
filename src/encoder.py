import torch
import torch.nn as nn
import torchvision.models as models

class TriPlaneEncoder(nn.Module):
    def __init__(self, feature_dim=64, image_mode="mask"):
        super().__init__()
        
        # We strictly want 1 channel for masks to save VRAM and parameter capacity
        in_channels = 1 if image_mode == "mask" else 3
        
        # 1. Load the standard ResNet backbone with GroupNorm
        resnet = models.resnet18(
            weights=None, 
            norm_layer=lambda channels: nn.GroupNorm(num_groups=8, num_channels=channels)
        )
        
        # 2. Modify the first convolutional layer for our specific input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Inherit the patched ResNet layers
        self.bn1 = resnet.bn1 # Mathematically a GroupNorm layer now
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 4. Spatial Upsampler (The "Minecraft" Fix)
        # layer4 outputs a tiny 4x4 grid. TriPlanes need 32x32 resolution for sharp geometry.
        self.spatial_upsampler = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 5. YZ Plane Combiner
        self.yz_combiner = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_view_frames):
        """
        Inputs:
            multi_view_frames: [Batch, Views=4, C, H=128, W=128]
        """
        B, Views, C, H, W = multi_view_frames.shape
        
        # THE FIX: If the dataloader hands us a 3-channel RGB mask from cv2.merge, 
        # but the network is strictly configured for 1-channel, slice it.
        if C == 3 and self.conv1.in_channels == 1:
            multi_view_frames = multi_view_frames[:, :, 0:1, :, :]
            C = 1 # Update C so the reshape math below doesn't break
        
        # Flatten the batch and views dimensions to process all images simultaneously
        # Shape becomes: [Batch * 4, 1, 128, 128]
        flat_frames = multi_view_frames.reshape(B * Views, C, H, W)
        
        # --- ResNet Backbone Forward Pass ---
        x = self.conv1(flat_frames)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        resnet_features = self.layer4(x) # Shape: [Batch * 4, 512, 4, 4]
        
        # --- TriPlane Spatial Reconstruction ---
        # Unfold the 4x4 logic core back up to crisp 32x32 spatial grids
        high_res_features = self.spatial_upsampler(resnet_features) # Shape: [Batch * 4, 64, 32, 32]
        
        # Unflatten back to separated views -> Shape: [Batch, 4, feature_dim, 32, 32]
        _, F_dim, F_H, F_W = high_res_features.shape
        features = high_res_features.reshape(B, Views, F_dim, F_H, F_W)
        
        # Split into the specific camera views
        f_side1 = features[:, 0] # Looks at YZ
        f_side2 = features[:, 1] # Looks at XZ
        f_side3 = features[:, 2] # Looks at YZ
        f_top   = features[:, 3] # Looks at XY
        
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