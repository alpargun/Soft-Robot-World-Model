import torch
import torchvision.models as models
import torch.nn as nn


class TriPlaneEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=32):
        super().__init__()
        
        # 1. Shared 2D Feature Extractor (CNN Backbone)
        # In a full model, this could be a ResNet or a Unet encoder.
        # We use a simple 3-layer CNN here to extract dense feature maps.
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 2. YZ Plane Combiner
        # Since Side1 and Side3 both look at the YZ plane, we concatenate 
        # their features and compress them back down to the target feature_dim.
        self.yz_combiner = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_view_frames):
        """
        Inputs:
            multi_view_frames: Tensor of shape [Batch, 4 (Views), C=3, H=128, W=128]
                               Expected view order: ["Side1", "Side2", "Side3", "Top"]
        Outputs:
            A dictionary containing the 3 orthogonal feature planes.
            Each plane has shape [Batch, feature_dim, H, W]
        """
        # Split the views
        side1 = multi_view_frames[:, 0] # Looks at YZ
        side2 = multi_view_frames[:, 1] # Looks at XZ
        side3 = multi_view_frames[:, 2] # Looks at YZ
        top   = multi_view_frames[:, 3] # Looks at XY

        # Extract 2D features for each view [Batch, feature_dim, H, W]
        f_side1 = self.backbone(side1)
        f_side2 = self.backbone(side2)
        f_side3 = self.backbone(side3)
        f_top   = self.backbone(top)

        # Map to the corresponding 3D spatial planes
        plane_xy = f_top
        plane_xz = f_side2
        
        # Merge the opposing X-axis views for a complete YZ plane
        f_yz_cat = torch.cat([f_side1, f_side3], dim=1) # Shape: [Batch, feature_dim*2, H, W]
        plane_yz = self.yz_combiner(f_yz_cat)           # Shape: [Batch, feature_dim, H, W]

        return {
            "xy": plane_xy,
            "xz": plane_xz,
            "yz": plane_yz
        }

# --- Quick Dimension Check ---
if __name__ == "__main__":
    # Dummy input representing a batch of 2 frames
    # [Batch=2, Views=4, Channels=3, H=128, W=128]
    dummy_input = torch.randn(2, 4, 3, 128, 128)
    
    encoder = TriPlaneEncoder(feature_dim=32)
    tri_planes = encoder(dummy_input)
    
    print("Tri-Plane Feature Dimensions:")
    print(f"XY Plane: {tri_planes['xy'].shape}") # Expected: [2, 32, 128, 128]
    print(f"XZ Plane: {tri_planes['xz'].shape}") # Expected: [2, 32, 128, 128]
    print(f"YZ Plane: {tri_planes['yz'].shape}") # Expected: [2, 32, 128, 128]