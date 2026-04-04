import math
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
            rendered_colors: The final 2D pixels [B, Num_Rays, Channels]
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
            
        return rendered_colors
    

# Orthographic ray generator
def sample_orthographic_rays(target_frames, num_samples=1024, image_mode="mask"):
    """
    Maps 2D pixel coordinates to 3D continuous ray origins and directions.
    target_frames: Ground truth video tensor [Batch, Views=4, Channels=C, H=128, W=128]
    image_mode: Toggles between random scatter ("mask") and contiguous patches ("rgb")
    """
    B, Views, C, H, W = target_frames.shape
    device = target_frames.device

    # Evenly divide the random samples across all 4 camera views
    samples_per_view = num_samples // Views
    
    origins_list = []
    directions_list = []
    target_rgb_list = []

    for v in range(Views):
        orig = torch.zeros(B, samples_per_view, 3, device=device)
        dirs = torch.zeros(B, samples_per_view, 3, device=device)
        
        # 1. Generate continuous pixel coordinates between [-1, 1] based on mode
        if image_mode == "rgb":
            # PATCH SAMPLING: Grab a random contiguous 16x16 square for LPIPS
            patch_dim = int(math.sqrt(samples_per_view))
            start_x = torch.randint(0, W - patch_dim, (B,), device=device)
            start_y = torch.randint(0, H - patch_dim, (B,), device=device)
            
            y_grid, x_grid = torch.meshgrid(torch.arange(patch_dim, device=device), 
                                            torch.arange(patch_dim, device=device), indexing='ij')
            
            u_idx = start_x.unsqueeze(1) + x_grid.flatten().unsqueeze(0) # [B, 256]
            v_idx = start_y.unsqueeze(1) + y_grid.flatten().unsqueeze(0) # [B, 256]
            
            u = (u_idx.float() / (W - 1)) * 2.0 - 1.0
            v_coord = (v_idx.float() / (H - 1)) * 2.0 - 1.0
            
        else:
            # === THE FIX: FOREGROUND-BIASED SAMPLING ===
            num_fg = samples_per_view // 2  # 50% of rays MUST hit the robot
            num_bg = samples_per_view - num_fg
            
            u_idx = torch.zeros(B, samples_per_view, device=device, dtype=torch.long)
            v_idx = torch.zeros(B, samples_per_view, device=device, dtype=torch.long)
            
            # Get the view mask (just channel 0 to find the robot shape)
            view_mask = target_frames[:, v, 0, :, :] 
            
            for b in range(B):
                # Find coordinates where the mask is white (> 0.5)
                fg_coords = torch.nonzero(view_mask[b] > 0.5) 
                
                if len(fg_coords) > 0:
                    # Randomly pick from the foreground (the robot)
                    rand_fg_indices = torch.randint(0, len(fg_coords), (num_fg,))
                    chosen_fg = fg_coords[rand_fg_indices]
                    v_idx[b, :num_fg] = chosen_fg[:, 0]
                    u_idx[b, :num_fg] = chosen_fg[:, 1]
                else:
                    # Fallback if the frame is completely empty (rare)
                    v_idx[b, :num_fg] = torch.randint(0, H, (num_fg,), device=device)
                    u_idx[b, :num_fg] = torch.randint(0, W, (num_fg,), device=device)
                    
                # Randomly pick background/global points for the remaining 50%
                v_idx[b, num_fg:] = torch.randint(0, H, (num_bg,), device=device)
                u_idx[b, num_fg:] = torch.randint(0, W, (num_bg,), device=device)
                
            u = (u_idx.float() / (W - 1)) * 2.0 - 1.0
            v_coord = (v_idx.float() / (H - 1)) * 2.0 - 1.0
        
        # 2. Map coordinates based on strict ANSYS camera positions
        if v == 0: # Side 1 (Camera at +X, looking at -X)
            orig[..., 0] = 1.0
            orig[..., 1] = u
            orig[..., 2] = v_coord
            dirs[..., 0] = -1.0
        elif v == 1: # Side 2 (Camera at +Y, looking at -Y)
            orig[..., 0] = u
            orig[..., 1] = 1.0
            orig[..., 2] = v_coord
            dirs[..., 1] = -1.0
        elif v == 2: # Side 3 (Camera at -X, looking at +X)
            orig[..., 0] = -1.0
            orig[..., 1] = u
            orig[..., 2] = v_coord
            dirs[..., 0] = 1.0
        elif v == 3: # Top (Camera at +Z, looking at -Z)
            orig[..., 0] = u
            orig[..., 1] = v_coord
            orig[..., 2] = 1.0
            dirs[..., 2] = -1.0

        origins_list.append(orig)
        directions_list.append(dirs)
        
        # 3. Extract the RGB values safely
        view_frames = target_frames[:, v]
        view_frames = view_frames.permute(0, 2, 3, 1)
        
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, samples_per_view)
        rgb = view_frames[batch_indices, v_idx, u_idx, :] 
        target_rgb_list.append(rgb)

    # Combine all views into a single batch of rays
    origins = torch.cat(origins_list, dim=1)         # [B, num_samples, 3]
    directions = torch.cat(directions_list, dim=1)   # [B, num_samples, 3]
    target_rgb = torch.cat(target_rgb_list, dim=1)   # [B, num_samples, C]

    return origins, directions, target_rgb


# Generate rays for rendering the predicted frames
def get_full_image_rays(H, W, view_idx=0, device='cuda'):
    """
    Generates an orthographic ray grid for a full H x W image, strictly matching 
    the camera poses defined in the ANSYS training data.
    """
    # Create the 2D grid spanning [-1, 1]
    # v_coord corresponds to Height (rows), u corresponds to Width (columns)
    v_coord_grid, u_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    # Flatten into 1D coordinate lists
    v_coord = v_coord_grid.reshape(-1, 1)
    u = u_grid.reshape(-1, 1)
    
    # Initialize origins and directions
    ray_origins = torch.zeros((H * W, 3), device=device)
    ray_dirs = torch.zeros((H * W, 3), device=device)
    
    # Map the coordinates matching the training views
    if view_idx == 0:   # Side 1: Camera at +X, looking at -X
        ray_origins[..., 0] = 1.0
        ray_origins[..., 1] = u.squeeze()
        ray_origins[..., 2] = v_coord.squeeze()
        ray_dirs[..., 0] = -1.0
        
    elif view_idx == 1: # Side 2: Camera at +Y, looking at -Y
        ray_origins[..., 0] = u.squeeze()
        ray_origins[..., 1] = 1.0
        ray_origins[..., 2] = v_coord.squeeze()
        ray_dirs[..., 1] = -1.0
        
    elif view_idx == 2: # Side 3: Camera at -X, looking at +X
        ray_origins[..., 0] = -1.0
        ray_origins[..., 1] = u.squeeze()
        ray_origins[..., 2] = v_coord.squeeze()
        ray_dirs[..., 0] = 1.0
        
    elif view_idx == 3: # Top: Camera at +Z, looking at -Z
        ray_origins[..., 0] = u.squeeze()
        ray_origins[..., 1] = v_coord.squeeze()
        ray_origins[..., 2] = 1.0
        ray_dirs[..., 2] = -1.0

    return ray_origins, ray_dirs

def render_rays_chunked(ray_marcher, decoder, curr_planes, ray_origins, ray_dirs, chunk_size=4096):
    """
    Memory-safe wrapper for VolumetricRayMarcher.
    Splits a massive bundle of rays into smaller chunks to prevent VRAM OOM crashes.
    """
    B, num_rays, _ = ray_origins.shape
    device = ray_origins.device
    all_rgb = []
    
    with torch.no_grad():
        for i in range(0, num_rays, chunk_size):
            origins_chunk = ray_origins[:, i:i+chunk_size, :]
            dirs_chunk = ray_dirs[:, i:i+chunk_size, :]
            
            # Call your existing render function on just this small block
            rgb_chunk = ray_marcher.render_rays(decoder, curr_planes, origins_chunk, dirs_chunk)
            all_rgb.append(rgb_chunk)
            
    # Stitch the rendered blocks back together
    return torch.cat(all_rgb, dim=1)