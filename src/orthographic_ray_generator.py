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
    target_rgb = torch.cat(target_rgb_list, dim=1)   # [B, num_samples, 3]

    return origins, directions, target_rgb