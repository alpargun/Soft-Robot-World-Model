import torch
import torch.nn as nn
import torch.nn.functional as F

class NOFDecoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        
        hidden_dim = feature_dim * 3 # Increase hidden dimension for more capacity for high-res planes

        # Input is feature_dim * 3 because we concatenate the 3 planes
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.SiLU(), # Swapped ReLU for SiLU to ensure smooth, continuous 3D boundaries
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        #  We need density for binary masks (color is irrelevant)
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Ensures density is always positive
        )

        # Zero out the weights so random variance cannot explode the density
        torch.nn.init.zeros_(self.density_head[0].weight)
        # Set bias to -5.0 so Softplus outputs ~0.006 (transparent empty air)
        torch.nn.init.constant_(self.density_head[0].bias, -5.0)

    def sample_plane(self, plane_features, coordinates):
        """
        Extracts features from a 2D plane at the specified coordinates.
        coordinates: [Batch, N_points, 2] (Normalized between -1 and 1)
        """
        # grid_sample expects coordinates in shape [Batch, H, W, 2]
        B, N, _ = coordinates.shape
        grid = coordinates.view(B, 1, N, 2) # Expand N_points into a 1D spatial grid
        
        # Extract features using bilinear interpolation
        sampled_features = F.grid_sample(plane_features, grid, align_corners=False, padding_mode='zeros')
        
        return sampled_features.squeeze(2).permute(0, 2, 1) # Reshape back to [Batch, N_points, feature_dim]

    def forward(self, tri_planes, points_3d):
        """
        Inputs:
            tri_planes: Dict of 'xy', 'xz', 'yz' feature tensors [B, feature_dim, H, W]
            points_3d: 3D coordinates to query [B, N_points, 3] (normalized -1 to 1)
        Outputs:
            density: Predicted solidness [B, N_points, 1]
        """
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]
        
        # Maps dataset coordinates to each view
        coords_xy = torch.stack([-Y, -X], dim=-1) # Top View
        coords_xz = torch.stack([-X, -Z], dim=-1) # Side 2 View
        coords_yz = torch.stack([Y, -Z], dim=-1)  # Side 1 View
        
        # Extract the features from each plane
        feat_xy = self.sample_plane(tri_planes['xy'], coords_xy)
        feat_xz = self.sample_plane(tri_planes['xz'], coords_xz)
        feat_yz = self.sample_plane(tri_planes['yz'], coords_yz)
        
        # Aggregate features
        fused_features = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        
        density = self.density_head(self.mlp(fused_features))
        
        # Return a dummy 1-channel color since VolumetricRayMarcher expects RGB and Density
        dummy_color = torch.ones_like(density)

        return dummy_color, density