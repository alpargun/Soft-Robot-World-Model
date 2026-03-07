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
from encoder_mini import MiniResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from visualization_helper import get_full_image_rays

def main():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/miniresnet_100cases_MASK_2026-03-05_23-33-28/best_model.pth"
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_VIDEO_PATH = f"miniResNet100_cases_SideBySide_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 32
    FPS = 30
    IMAGE_MODE = "mask" # MUST match what you used in train.py!
    
    # --- NEW: POST-PROCESSING TOGGLE ---
    APPLY_POSTPROCESSING = True # Set to False to see raw probability dust

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
    encoder = MiniResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])

    # CRITICAL: Keep these in train mode to bypass corrupted moving averages!
    encoder.train()
    dynamics.train()
    decoder.eval()

    # ==========================================
    # --- 3. GET A TEST SEQUENCE ---
    # ==========================================
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    
    # THE FIX: Load TWO DISTINCT runs to give BatchNorm actual variance!
    test_dataset = Subset(full_dataset, indices=[-1, -2]) 
    dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device)    # Shape will be [2, 60, 4, 3, 128, 128]
    pressures = batch["pressures"].to(device)  # Shape will be [2, 60, 3]

    _, Time, Views, C, H, W = real_videos.shape

    # ==========================================
    # --- 4. AUTOREGRESSIVE LATENT ROLLOUT ---
    # ==========================================
    print("Starting Latent Autoregressive Video Generation...")
    
    video_width = W * Views
    video_height = H * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (video_width, video_height))
    
    hidden_state = None
    
    # Encode both starting frames simultaneously
    current_frames = real_videos[:, 0] 
    tri_planes_t = encoder(current_frames)

    with torch.no_grad():
        for t in range(Time - 1):
            print(f"Rendering Frame {t+1}/{Time-1}...")
            action_t = pressures[:, t]
            
            # Predict the next state for BOTH sequences
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            pred_views_np = []
            real_views_np = []
            
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0) 
                ray_dirs = ray_dirs.unsqueeze(0)
                
                # EXTRACT ONLY INDEX 0 FOR RENDERING THE VIDEO
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins, ray_dirs)

                # Fetch raw prediction
                pred_img = rgb_pred.view(H, W, 3).cpu().numpy()
                
                # 1. Base Binary Thresholding
                pred_img = (pred_img > 0.5).astype(np.float32)
                pred_img = (pred_img * 255).astype(np.uint8)
                
                # 2. Optional CV Cleanup (Keep only largest contour)
                if APPLY_POSTPROCESSING:
                    gray = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
                    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        clean_mask = np.zeros_like(gray)
                        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        pred_img = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
                    else:
                        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                else:
                    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                
                pred_views_np.append(pred_img)
                
                # Extract real target from index 0
                real_target = real_videos[0, t+1, v].permute(1, 2, 0).cpu().numpy()
                real_target = (np.clip(real_target, 0, 1) * 255).astype(np.uint8)
                real_target = cv2.cvtColor(real_target, cv2.COLOR_RGB2BGR)
                real_views_np.append(real_target)
                
            top_row = np.hstack(real_views_np) 
            bottom_row = np.hstack(pred_views_np) 
            combined_frame = np.vstack((top_row, bottom_row))
            out_video.write(combined_frame) 
            
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()