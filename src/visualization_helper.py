import os
import torch
import matplotlib.pyplot as plt
import numpy as np


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


def save_comparison_frame(real_image, pred_image, epoch, view_idx=0, save_dir="training_renders"):
    """Saves a side-by-side image of the real vs predicted frames for a specific view."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(np.clip(real_image, 0, 1))
    axes[0].set_title(f"Real ANSYS Frame (View {view_idx})", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(pred_image, 0, 1))
    axes[1].set_title(f"Neural Network Render (View {view_idx})", fontsize=14)
    axes[1].axis('off')
    
    # Include the view_idx in the filename
    filename = os.path.join(save_dir, f"render_epoch_{epoch:04d}_view_{view_idx}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)