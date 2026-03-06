import os
# CRITICAL: Keep this for Apple Silicon fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

import torch
import numpy as np
from skimage import measure
from torch.utils.data import DataLoader, Subset

# Import your custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder

def save_obj(filepath, verts, faces):
    """Saves vertices and faces to a standard .obj 3D mesh file."""
    with open(filepath, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # .obj faces are 1-indexed, so we add 1 to every index
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Successfully saved 3D mesh to: {filepath}")

def main():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "world_model_checkpoint_epoch_500.pth"
    
    FEATURE_DIM = 32
    IMAGE_MODE = "mask" # MUST match train.py!
    
    GRID_RES = 128  # Resolution of the 3D bounding box (128x128x128 = 2M points)
    
    # --- CRITICAL FIX FOR 1-CHANNEL MASK ---
    # Values range from 0.0 to 1.0. Set to 0.5 to find the exact boundary of the rubber.
    DENSITY_THRESHOLD = 0.5  
    
    CHUNK_SIZE = 100000  # Process points in chunks so your Mac's RAM doesn't crash
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    print(f"Extracting 3D Mesh Sequence on: {device}")

    # ==========================================
    # --- 2. LOAD THE TRAINED MODEL ---
    # ==========================================
    encoder = ResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        print(f"Error: Could not find {CHECKPOINT_PATH}.")
        return

    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # ==========================================
    # --- 3. GET THE TEST SEQUENCE ---
    # ==========================================
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    test_dataset = Subset(full_dataset, indices=[-1]) 
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device)       # [B=1, Time=60, Views=4, C=3, H=128, W=128]
    pressures = batch["pressures"].to(device)     # [B=1, Time=60, 3]
    
    _, Time, Views, C, H, W = real_videos.shape

    # ==========================================
    # --- 4. BUILD THE DENSE 3D QUERY GRID ---
    # ==========================================
    print(f"Building {GRID_RES}x{GRID_RES}x{GRID_RES} query grid...")
    x = torch.linspace(-1.0, 1.0, GRID_RES)
    y = torch.linspace(-1.0, 1.0, GRID_RES)
    z = torch.linspace(-1.0, 1.0, GRID_RES)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    flat_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)

    # Helper function to extract and save a frame
    def extract_and_save_mesh(planes, frame_idx):
        densities = []
        for i in range(0, flat_grid.shape[0], CHUNK_SIZE):
            chunk_points = flat_grid[i : i + CHUNK_SIZE].unsqueeze(0) # Add batch dim
            # Matching your specific decoder signature
            _, chunk_density = decoder(planes, chunk_points)
            densities.append(chunk_density.squeeze().cpu())
            
        full_density_flat = torch.cat(densities, dim=0)
        volume_density = full_density_flat.view(GRID_RES, GRID_RES, GRID_RES).numpy()
        
        try:
            verts, faces, normals, values = measure.marching_cubes(volume_density, level=DENSITY_THRESHOLD)
            verts = (verts / GRID_RES) * 2.0 - 1.0
            
            output_obj_path = f"AI_Predicted_Frame_{frame_idx:02d}.obj"
            save_obj(output_obj_path, verts, faces)
        except ValueError as e:
            print(f"Marching Cubes failed on frame {frame_idx}: {e}")
            print(f"Volume Min Density: {volume_density.min():.2f}, Max: {volume_density.max():.2f}")

    # ==========================================
    # --- 5. AUTOREGRESSIVE 3D ROLLOUT ---
    # ==========================================
    print("Starting Autoregressive 3D Generation...")
    hidden_state = None
    
    with torch.no_grad():
        # Frame 0: Extract the initial state directly from the first frame
        print("--- Processing Frame 00 (Initial State) ---")
        current_frames = real_videos[:, 0]
        tri_planes_t = encoder(current_frames)
        extract_and_save_mesh(tri_planes_t, 0)
        
        # Frames 1 to 59: AI predicts the future physics states
        for t in range(Time - 1):
            print(f"--- Predicting & Processing Frame {t+1:02d}/{Time-1} ---")
            action_t = pressures[:, t]
            
            # Predict the NEXT 3D state based on physics
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            # Extract the 3D mesh for the predicted state
            extract_and_save_mesh(planes_next_pred, t + 1)
            
            # Update the current state for the next loop iteration
            # We use the pure predicted 3D planes to avoid quantization noise!
            tri_planes_t = planes_next_pred 

    print("\nAll 60 predicted 3D frames saved successfully!")

if __name__ == "__main__":
    main()