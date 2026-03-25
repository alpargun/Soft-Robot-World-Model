import torch
import torch.nn as nn
import torch.nn.functional as F

class TriPlaneDecoder(nn.Module):
    def __init__(self, feature_dim=64, image_mode="mask"):
        super().__init__()
        
        self.image_mode = image_mode
        # Toggle output channels based on the mode
        out_channels = 1 if image_mode == "mask" else 3
        
        hidden_dim = feature_dim * 3 # Increased hidden dimension for better expressiveness with high-res planes

        # Input is feature_dim * 3 because we concatenate the 3 planes
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.SiLU(), # Swapped ReLU for SiLU to ensure smooth, continuous 3D boundaries
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, out_channels),
            nn.Sigmoid() 
        )
        
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus() 
        )

    def sample_plane(self, plane_features, coordinates):
        """
        Extracts features from a 2D plane at the specified coordinates.
        coordinates: [Batch, N_points, 2] (Normalized between -1 and 1)
        """
        # grid_sample expects coordinates in shape [Batch, H, W, 2]
        # We dummy-expand our N_points into a 1D spatial grid
        B, N, _ = coordinates.shape
        grid = coordinates.view(B, 1, N, 2) 
        
        # Extract features using bilinear interpolation
        sampled_features = F.grid_sample(plane_features, grid, align_corners=True, padding_mode='zeros')
        
        # Reshape back to [Batch, N_points, feature_dim]
        return sampled_features.squeeze(2).permute(0, 2, 1)

    def forward(self, tri_planes, points_3d):
        """
        Inputs:
            tri_planes: Dict of 'xy', 'xz', 'yz' feature tensors [B, feature_dim, H, W]
            points_3d: 3D coordinates to query [B, N_points, 3] (normalized -1 to 1)
        Outputs:
            rgb: Predicted colors [B, N_points, 3]
            density: Predicted solidness [B, N_points, 1]
        """
        # 1. Project the 3D points onto the 3 orthogonal 2D planes
        coords_xy = points_3d[..., [0, 1]] # points_3d is (X, Y, Z)
        coords_xz = points_3d[..., [0, 2]]
        coords_yz = points_3d[..., [1, 2]]
        
        # 2. Extract the features from each plane
        feat_xy = self.sample_plane(tri_planes['xy'], coords_xy)
        feat_xz = self.sample_plane(tri_planes['xz'], coords_xz)
        feat_yz = self.sample_plane(tri_planes['yz'], coords_yz)
        
        # 3. Aggregate features (Concatenation)
        fused_features = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        
        # 4. Predict Color and Density using the MLP
        hidden = self.mlp(fused_features)
        
        color_out = self.color_head(hidden)
        density = self.density_head(hidden)

        return color_out, density