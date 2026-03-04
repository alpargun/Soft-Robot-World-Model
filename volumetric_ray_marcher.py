import torch
import torch.nn as nn

class VolumetricRayMarcher(nn.Module):
    def __init__(self, num_samples=64):
        super().__init__()
        self.num_samples = num_samples # Points sampled along each ray

    def render_rays(self, decoder, tri_planes, ray_origins, ray_directions, near=0.1, far=2.0):
        """
        Shoots rays through the volume and composites the final pixel colors.
        Inputs:
            decoder: Your TriPlaneDecoder module
            tri_planes: The predicted 'xy', 'xz', 'yz' feature dict [B, C, H, W]
            ray_origins: Camera positions [B, Num_Rays, 3]
            ray_directions: Ray vectors [B, Num_Rays, 3]
        Outputs:
            rendered_colors: The final 2D pixels [B, Num_Rays, 3]
        """
        B, num_rays, _ = ray_origins.shape
        device = ray_origins.device
        
        # 1. Generate sample points along the ray (from 'near' to 'far')
        t_vals = torch.linspace(near, far, self.num_samples, device=device)
        t_vals = t_vals.expand(B, num_rays, self.num_samples)
        
        # Add a tiny bit of random noise during training to prevent aliasing (stratified sampling)
        if self.training:
            noise = (torch.rand_like(t_vals) - 0.5) * ((far - near) / self.num_samples)
            t_vals = t_vals + noise

        # 2. Calculate the exact 3D coordinates for every point on every ray
        # ray_origins: [B, num_rays, 1, 3]
        # ray_directions: [B, num_rays, 1, 3]
        # t_vals: [B, num_rays, num_samples, 1]
        points_3d = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * t_vals.unsqueeze(3)
        points_flat = points_3d.reshape(B, num_rays * self.num_samples, 3)
        
        # 3. Query the Decoder
        rgb_flat, density_flat = decoder(tri_planes, points_flat)
        
        # DYNAMIC CHANNEL RESHAPING
        # Check if the decoder outputted 1 channel (mask) or 3 channels (rgb)
        out_channels = rgb_flat.shape[-1] 
        
        rgb = rgb_flat.reshape(B, num_rays, self.num_samples, out_channels)
        density = density_flat.reshape(B, num_rays, self.num_samples)
        
        # 4. Volumetric Compositing
        # Calculate distances between samples (delta)
        deltas = t_vals[:, :, 1:] - t_vals[:, :, :-1]
        # Append a large number for the last segment stretching to infinity
        last_delta = torch.full((B, num_rays, 1), 1e10, device=device)
        deltas = torch.cat([deltas, last_delta], dim=2)
        
        # Alpha: How opaque is this specific segment?
        alpha = 1.0 - torch.exp(-density * deltas)
        
        # Transmittance: How much light made it to this point without hitting something earlier?
        # T_i = exp(-sum(density * delta))
        transmittance = torch.cumprod(torch.cat([torch.ones(B, num_rays, 1, device=device), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :, :-1]
        
        # Weights: Contribution of each point to the final pixel color
        weights = alpha * transmittance
        
        # Sum the weighted colors (Result is [B, num_rays, out_channels])
        rendered_colors = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)

        # Duplicate 1-channel mask to 3 channels so TensorBoard and OpenCV don't crash
        if rendered_colors.shape[-1] == 1:
            rendered_colors = rendered_colors.repeat(1, 1, 3)
            
        return rendered_colors