import torch
import math

def sample_orthographic_rays(target_frames, num_samples=1024, image_mode="mask"):
    """
    Maps 2D pixel coordinates to 3D continuous ray origins and directions.
    target_frames: Ground truth video tensor [Batch, Views=4, Channels=3, H=128, W=128]
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
            # RANDOM SAMPLING: Your original scattered sampling for L1 masks
            u = (torch.rand(B, samples_per_view, device=device) * 2) - 1 # Width axis
            v_coord = (torch.rand(B, samples_per_view, device=device) * 2) - 1 # Height axis
            
            u_idx = torch.clamp(((u + 1) * 0.5 * W).long(), 0, W - 1)
            v_idx = torch.clamp(((v_coord + 1) * 0.5 * H).long(), 0, H - 1) 
        
        # 2. Map coordinates based on your strict ANSYS camera positions
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
        
        # 3. Extract the RGB values: [B, samples_per_view, 3]
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, samples_per_view)
        
        # PyTorch images are [Channels, Height, Width], so we index [..., C, v_idx, u_idx]
        rgb = target_frames[batch_indices, v, :, v_idx, u_idx] 
        target_rgb_list.append(rgb)

    # Combine all views into a single batch of rays
    origins = torch.cat(origins_list, dim=1)         # [B, num_samples, 3]
    directions = torch.cat(directions_list, dim=1)   # [B, num_samples, 3]
    target_rgb = torch.cat(target_rgb_list, dim=1)   # [B, num_samples, 3]

    return origins, directions, target_rgb