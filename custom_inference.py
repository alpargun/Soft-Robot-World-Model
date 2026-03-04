import os
# CRITICAL: Keep this for Apple Silicon (M1/M2/M3) fallback!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import your custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from visualization_helper import get_full_image_rays

# ==========================================
# --- 1. CUSTOM PRESSURE SCULPTOR ---
# ==========================================
def create_custom_pressure_profile(num_frames, device):
    """
    Design your custom 3-bellow pressure sequence here!
    Values must be normalized: 0.00001 (1 Pa) to 1.0 (100 kPa).
    Returns a tensor of shape [Batch=1, Time, Bellows=3]
    """
    # Initialize all 3 bellows to the MIN_PRESSURE resting state (1 Pa)
    pressures = torch.full((1, num_frames, 3), 0.00001).to(device)
    
    # EXAMPLE: A smooth ramp up and down on Bellow 1 (Index 0)
    # Ramps from 1 Pa to x kPa over the first 30 frames
    pressures[0, 0:30, 2] = torch.linspace(0.00001, 0.25, 30) # Mimic Case 1: P3 (Index 2) ramps to 25 kPa (0.25)
    
    # Ramps from x kPa back down to 1 Pa over the last 30 frames
    pressures[0, 30:60, 2] = torch.linspace(1.0, 0.00001, 30)
    
    # Optional: What if Bellow 2 suddenly inflates halfway through?
    # pressures[0, 30:60, 1] = torch.linspace(0.00001, 0.5, 30) # Ramps to 50 kPa

    return pressures

def main():
    # ==========================================
    # --- 2. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "world_model_checkpoint_epoch_500.pth"
    OUTPUT_VIDEO_PATH = "AI_Custom_Simulation.mp4"
    
    FEATURE_DIM = 32
    FPS = 30 
    TOTAL_FRAMES = 60
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Custom AI Simulation on: {device}")

    # ==========================================
    # --- 3. LOAD THE TRAINED MODEL ---
    # ==========================================
    encoder = ResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device) # High quality render
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        print(f"Error: Could not find {CHECKPOINT_PATH}. Wait for training to finish!")
        return

    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # ==========================================
    # --- 4. EXTRACT INITIAL RESTING FRAME ---
    # ==========================================
    # We still need the dataloader just to grab the resting frame (t=0) so the AI 
    # knows what the robot looks like before the custom pressures are applied.
    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600)
    dataloader = DataLoader(Subset(dataset, indices=[0]), batch_size=1)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device) 
    _, _, Views, C, H, W = real_videos.shape
    
    # Isolate the very first resting frame
    current_frames = real_videos[:, 0].clone() # Shape: [1, 4, 3, 128, 128]

    # Generate your custom pressures!
    custom_pressures = create_custom_pressure_profile(TOTAL_FRAMES, device)

    # ==========================================
    # --- 5. THE AI PHYSICS ENGINE LOOP ---
    # ==========================================
    print("\nStarting Open-Loop Autoregressive Simulation...")
    
    
    # We will stitch the 4 predicted views side-by-side
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (W * Views, H))
    
    # Save the initial resting frame to the video first
    initial_views_np = []
    for v in range(Views):
        init_img = current_frames[0, v].permute(1, 2, 0).cpu().numpy()
        init_img = (np.clip(init_img, 0, 1) * 255).astype(np.uint8)
        init_img = cv2.cvtColor(init_img, cv2.COLOR_RGB2BGR)
        initial_views_np.append(init_img)
    out_video.write(np.hstack(initial_views_np))
    
    hidden_state = None
    
    with torch.no_grad():
        # Loop from t=0 to t=58 to predict frames 1 through 59
        for t in range(TOTAL_FRAMES - 1):
            action_t = custom_pressures[:, t]
            
            # 1. Look at the current frame and build Tri-Planes
            tri_planes_t = encoder(current_frames)
            
            # 2. Push the physics forward by applying the pressure
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            # 3. Render the new 3D shape into 4 2D camera views
            pred_views_np = []
            pred_tensors = [] # We need to save these to feed back into the AI
            
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0)  # Add batch dimension: [1, num_rays, 3]
                ray_dirs = ray_dirs.unsqueeze(0)        # Add batch dimension: [1, num_rays, 3]
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins, ray_dirs)
                
                # Format for PyTorch (to feed back into the loop next step)
                pred_tensor_view = rgb_pred.view(C, H, W)
                pred_tensors.append(pred_tensor_view)
                
                # Format for OpenCV (to save to the video file)
                pred_img = rgb_pred.view(H, W, 3).cpu().numpy()
                pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                pred_views_np.append(pred_img)
                
            # Stitch the 4 views side-by-side and write to MP4
            combined_frame = np.hstack(pred_views_np)
            out_video.write(combined_frame)
            
            # 4. OVERRIDE REALITY: Make the AI's prediction the new "current_frame"
            # This is where the magic happens. We don't give it ANSYS data anymore.
            current_frames = torch.stack(pred_tensors).unsqueeze(0) # [1, 4, 3, 128, 128]
            
            # Console progress
            p1 = custom_pressures[0, t, 0].item() * 100
            print(f"Rendered Frame {t+1:02d} | Applied P1: {p1:05.1f} kPa")

    out_video.release()
    print(f"\nSimulation Complete! Custom video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()