import os
# CRITICAL: Keep this for Apple Silicon (M1/M2/M3) fallback!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Import your custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from visualization_helper import get_full_image_rays

def main():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "world_model_checkpoint_epoch_500.pth"
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_VIDEO_PATH = f"AI_Prediction_SideBySide_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 32
    FPS = 30
    IMAGE_MODE = "mask" # MUST match what you used in train.py!

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Inference on: {device}")

    # ==========================================
    # --- 2. LOAD THE TRAINED MODEL ---
    # ==========================================
    encoder = ResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
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
    # --- 3. GET A TEST SEQUENCE ---
    # ==========================================
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    test_dataset = Subset(full_dataset, indices=[-1]) 
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device) # [B=1, Time=60, Views=4, C=3, H=128, W=128]     
    pressures = batch["pressures"].to(device) # [B=1, Time=60, 3]
    
    _, Time, Views, C, H, W = real_videos.shape

    # ==========================================
    # --- 4. AUTOREGRESSIVE LATENT ROLLOUT ---
    # ==========================================
    print("Starting Latent Autoregressive Video Generation...")
    
    video_width = W * Views
    video_height = H * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (video_width, video_height))
    
    # Initialize ConvGRU memory for hysteresis
    hidden_state = None
    
    # Encode ONLY the very first frame to establish the initial 3D state
    current_frames = real_videos[:, 0] 
    tri_planes_t = encoder(current_frames)

    with torch.no_grad():
        for t in range(Time - 1):
            print(f"Rendering Frame {t+1}/{Time-1}...")
            action_t = pressures[:, t]
            
            # 1. Predict the NEXT 3D state based on the applied pressure & hidden memory
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            # 2. Render all 4 predicted views from the 3D planes (Visualization Only)
            pred_views_np = []
            real_views_np = []
            
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0) 
                ray_dirs = ray_dirs.unsqueeze(0)
                
                # Render the 1-channel geometry and duplicate to 3 for OpenCV
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins, ray_dirs)

                # Convert Prediction to Numpy Image (0-255 scale)
                pred_img = rgb_pred.view(H, W, 3).cpu().numpy()
                pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR) 
                pred_views_np.append(pred_img)
                
                # Extract Real Target Frame for side-by-side comparison
                real_target = real_videos[0, t+1, v].permute(1, 2, 0).cpu().numpy()
                real_target = (np.clip(real_target, 0, 1) * 255).astype(np.uint8)
                real_target = cv2.cvtColor(real_target, cv2.COLOR_RGB2BGR)
                real_views_np.append(real_target)
                
            # 3. Stitch Images Together
            top_row = np.hstack(real_views_np) # Real ANSYS views
            bottom_row = np.hstack(pred_views_np) # AI Predicted views
            combined_frame = np.vstack((top_row, bottom_row))
            out_video.write(combined_frame) 
            
            # 4. === THE LATENT FIX ===
            # Feed the pure 3D prediction directly back into the physics engine!
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()