import os
# CRITICAL: Keep this for Apple Silicon (M1/M2/M3) fallback!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from skimage.measure import marching_cubes

# Import your custom modules
from multiview_dataset import SoftRobotDataset
from encoder_mini import MiniResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder

def export_obj(vertices, faces, filename):
    """Writes raw vertices and faces to a standard .obj file."""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/miniresnet_100cases_MASK_2026-03-05_23-33-28/best_model.pth"
    
    # Create a dedicated folder for this specific extraction run
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = f"mesh_extraction_{TIMESTAMP}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    FEATURE_DIM = 32
    IMAGE_MODE = "mask" 
    GRID_RES = 128        # 128x128x128 bounding volume
    CHUNK_SIZE = 262144   # Chunking for Apple Silicon (128^3 / 8)
    DENSITY_THRESHOLD = 0.5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running 3D Inference on: {device}")

    # ==========================================
    # --- 2. LOAD THE TRAINED MODEL ---
    # ==========================================
    encoder = MiniResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])

    encoder.train()
    dynamics.train()
    decoder.eval()

    # ==========================================
    # --- 3. GENERATE DENSE 3D QUERY GRID ---
    # ==========================================
    # Bounding box [-1, 1] for X, Y, Z
    x = torch.linspace(-1.0, 1.0, GRID_RES)
    y = torch.linspace(-1.0, 1.0, GRID_RES)
    z = torch.linspace(-1.0, 1.0, GRID_RES)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    # Shape: [GRID_RES^3, 3]
    query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)

    # ==========================================
    # --- 4. GET A TEST SEQUENCE ---
    # ==========================================
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    test_dataset = Subset(full_dataset, indices=[-1, -2]) 
    dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device)    
    pressures = batch["pressures"].to(device)  

    _, Time, Views, C, H, W = real_videos.shape

    # ==========================================
    # --- 5. AUTOREGRESSIVE LATENT ROLLOUT ---
    # ==========================================
    print(f"Starting 3D Extraction. Files will be saved to: {OUTPUT_DIR}")
    hidden_state = None
    
    # Encode starting frames
    current_frames = real_videos[:, 0] 
    tri_planes_t = encoder(current_frames)

    with torch.no_grad():
        for t in range(Time - 1):
            action_t = pressures[:, t]
            
            # Predict the next latent state
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            # Extract only index 0 for the mesh generation
            single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
            
            print(f"Querying volume for Frame {t+1:03d} / {Time-1}...")
            
            # --- CHUNKED 3D DECODING FOR APPLE SILICON ---
            densities = []
            for i in range(0, query_points.shape[0], CHUNK_SIZE):
                chunk_pts = query_points[i:i+CHUNK_SIZE].unsqueeze(0) # Shape: [1, CHUNK_SIZE, 3]
                
                # Pass the planes and the 3D points directly to your decoder!
                color_out, _ = decoder(single_plane_pred, chunk_pts)
                
                # We extract the Sigmoid probabilities (color_out) since IMAGE_MODE="mask"
                # Drop the channel dimension so it matches the [1, CHUNK_SIZE] shape
                chunk_prob = color_out.squeeze(-1) 
                
                densities.append(chunk_prob.cpu())
                
            # Reconstruct the 128x128x128 volume
            full_volume = torch.cat(densities, dim=1).view(GRID_RES, GRID_RES, GRID_RES).numpy()
            
            # --- MARCHING CUBES MESH EXTRACTION ---
            try:
                vertices, faces, normals, values = marching_cubes(full_volume, level=DENSITY_THRESHOLD)
                
                # Normalize vertices back to physical scale [-1.0, 1.0] if necessary
                vertices = (vertices / (GRID_RES - 1)) * 2.0 - 1.0
                
                obj_filename = os.path.join(OUTPUT_DIR, f"frame_{t+1:03d}.obj")
                export_obj(vertices, faces, obj_filename)
                
            except ValueError:
                print(f"Warning: No geometry found in Frame {t+1} (Empty volume).")
            
            # Advance the temporal state
            tri_planes_t = planes_next_pred

    print("\n3D Extraction Complete!")

if __name__ == "__main__":
    main()