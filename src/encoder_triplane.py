import torch
import torch.nn as nn

class Encoder2D(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        # Shallow network to preserve 32x32 spatial resolution
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)

class TriplaneEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        # Siamese backbone shared across all side views to learn universal bending
        self.side_encoder = Encoder2D(feature_dim=feature_dim)
        
        # Independent backbone for the top view
        self.top_encoder = Encoder2D(feature_dim=feature_dim)
        
        # Fusion layer to merge opposing sides into a single unified YZ plane
        self.yz_fusion = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)

    def forward(self, top_img, side1_img, side2_img, side3_img):
        # XY Plane captured from Top View
        plane_xy = self.top_encoder(top_img)
        
        # XZ Plane captured from Side 2
        plane_xz = self.side_encoder(side2_img)
        
        # YZ Plane (Front/Back Fusion) captured from Side 1 and Side 3
        feat_s1 = self.side_encoder(side1_img)
        feat_s3 = self.side_encoder(side3_img)
        
        # Flip Side 3 so the geometry aligns physically with Side 1 
        feat_s3_aligned = torch.flip(feat_s3, dims=[-1])
        fused_yz = torch.cat([feat_s1, feat_s3_aligned], dim=1)
        plane_yz = self.yz_fusion(fused_yz)
        
        return {'xy': plane_xy, 'xz': plane_xz, 'yz': plane_yz}