import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import your custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder

def extract_mesh_sequence():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/YOUR_LATEST_RUN_FOLDER/best_model.pth" # Update this later!
    OUTPUT_DIR = "Extracted_Meshes"
    
    FEATURE_DIM = 32
    IMAGE_MODE = "mask"
    GRID_RESOLUTION = 128 # 128x128x128 voxel grid
    DENSITY_THRESHOLD = 0.5 # Strict binary cutoff!
    CHUNK_SIZE = 100000 # Prevents Apple Silicon from running out of unified memory

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Extracting on: {device}")

    # ==========================================
    # --- 2. LOAD MODEL ---
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
        print(f"Waiting for checkpoint... (Could not find {CHECKPOINT_PATH})")
        return

    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # ==========================================
    # --- 3. PREPARE 3D QUERY GRID ---
    # ==========================================
    # Create a dense 3D grid in the bounds [-1, 1]
    linspace = torch.linspace(-1.0, 1.0, steps=GRID_RESOLUTION, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    
    # Flatten grid into a list of [X, Y, Z] points: Shape [128^3, 3] -> [2097152, 3]
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    num_points = query_points.shape[0]
    print(f"Generated 3D Query Grid with {num_points} points.")

    # ==========================================
    # --- 4. LOAD TARGET SEQUENCE ---
    # ==========================================
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    
    # Let's find the sequence with the highest pressure to extract extreme bending
    best_idx, highest_pressure = 0, 0.0
    for i in range(len(full_dataset)):
        current_max = full_dataset[i]["pressures"].max().item()
        if current_max > highest_pressure:
            highest_pressure = current_max
            best_idx = i
            
    print(f"Extracting Sequence {best_idx} (Peak Pressure: {highest_pressure:.2f} kPa)")
    test_dataset = Subset(full_dataset, indices=[best_idx]) 
    batch = next(iter(DataLoader(test_dataset, batch_size=1)))
    
    real_videos = batch["video"].to(device) 
    pressures = batch["pressures"].to(device)
    Time = real_videos.shape[1]

    # ==========================================
    # --- 5. AUTOREGRESSIVE EXTRACTION ---
    # ==========================================
    hidden_state = None
    tri_planes_t = encoder(real_videos[:, 0]) # t=0 starting state

    with torch.no_grad():
        for t in range(Time - 1):
            print(f"\nProcessing Frame {t+1}/{Time-1}...")
            action_t = pressures[:, t]
            
            # 1. Physics Engine Step
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            # 2. Query the entire 3D volume in safe chunks
            densities = torch.zeros(num_points, device=device)
            
            for i in tqdm(range(0, num_points, CHUNK_SIZE), desc="Querying Volume"):
                chunk_pts = query_points[i:i + CHUNK_SIZE].unsqueeze(0) # [1, CHUNK, 3]
                
                # Sample TriPlanes
                xy_feat = torch.nn.functional.grid_sample(planes_next_pred['xy'], chunk_pts[:, :, [0, 1]].unsqueeze(2), align_corners=True).squeeze(3)
                xz_feat = torch.nn.functional.grid_sample(planes_next_pred['xz'], chunk_pts[:, :, [0, 2]].unsqueeze(2), align_corners=True).squeeze(3)
                yz_feat = torch.nn.functional.grid_sample(planes_next_pred['yz'], chunk_pts[:, :, [1, 2]].unsqueeze(2), align_corners=True).squeeze(3)
                
                combined_feat = torch.cat([xy_feat, xz_feat, yz_feat], dim=1).permute(0, 2, 1) # [1, CHUNK, 96]
                
                # Pass through MLP Decoder
                chunk_density = decoder(combined_feat).squeeze(-1) # [1, CHUNK]
                densities[i:i + CHUNK_SIZE] = chunk_density[0]
                
            # 3. Reshape back to 3D Grid
            volume_3d = densities.view(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION).cpu().numpy()
            
            # 4. Marching Cubes Algorithm
            try:
                # Find the surface where probability crosses our strict 0.5 threshold
                verts, faces, normals, values = marching_cubes(volume_3d, level=DENSITY_THRESHOLD)
                
                # Export to .obj
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
                obj_filename = os.path.join(OUTPUT_DIR, f"frame_{t+1:03d}.obj")
                mesh.export(obj_filename)
                print(f"Saved: {obj_filename}")
                
            except ValueError as e:
                # Happens if the network predicts complete emptiness (no surface found)
                print(f"Skipping Frame {t+1}: No surface found at threshold {DENSITY_THRESHOLD}.")
                
            # Feed prediction back into the loop
            tri_planes_t = planes_next_pred

    print("\nExtraction Complete! Open the .obj files in Blender or MeshLab.")

if __name__ == "__main__":
    extract_mesh_sequence()