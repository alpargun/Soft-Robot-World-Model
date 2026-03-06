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
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "world_model_checkpoint_epoch_210.pth" # Ensure this points to your latest!
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_VIDEO_PATH = f"Custom_Control_AI_Rollout_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 32
    FPS = 30
    IMAGE_MODE = "mask" 
    ROLLOUT_FRAMES = 60 # How long you want your custom video to be

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Custom Inference on: {device}")

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
    # --- 3. BOOTSTRAP INITIAL STATE ---
    # ==========================================
    # We load ONE sequence just to get the t=0 visual frame to encode the resting robot
    full_dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    test_dataset = Subset(full_dataset, indices=[0]) 
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    batch = next(iter(dataloader))
    real_videos = batch["video"].to(device) 
    
    _, _, Views, C, H, W = real_videos.shape

    # ==========================================
    # --- 4. DEFINE CUSTOM PRESSURE PROFILE ---
    # ==========================================
    # Shape: [Batch=1, Time=ROLLOUT_FRAMES, ActionDim=3]
    custom_pressures = torch.ones((1, ROLLOUT_FRAMES, 3), device=device) # Initializes at 1 kPa
    
    print("Generating custom pressure profile...")
    for t in range(ROLLOUT_FRAMES):
        # Example Profile: Ramp Chamber 0 up to 120 kPa, then hold.
        # Ensure MIN_PRESSURE is strictly locked at 1 to prevent latent collapse
        p0 = min(1.0 + (t * 4.0), 120.0) 
        p1 = 1.0  
        p2 = 1.0  
        
        # --- Feel free to change this to sine waves, step functions, etc! ---
        # p1 = 1.0 + 50.0 * np.sin(t / 10.0) 
        
        custom_pressures[0, t, 0] = max(p0, 1.0)
        custom_pressures[0, t, 1] = max(p1, 1.0)
        custom_pressures[0, t, 2] = max(p2, 1.0)

    # ==========================================
    # --- 5. AUTOREGRESSIVE LATENT ROLLOUT ---
    # ==========================================
    print("Starting Custom Latent Autoregressive Video Generation...")
    
    video_width = W * Views
    video_height = H # Only one row now!
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (video_width, video_height))
    
    hidden_state = None
    
    # Encode ONLY the very first frame to establish the initial 3D resting state
    current_frames = real_videos[:, 0] 
    tri_planes_t = encoder(current_frames)

    with torch.no_grad():
        for t in range(ROLLOUT_FRAMES - 1):
            print(f"Rendering Frame {t+1}/{ROLLOUT_FRAMES-1} | Pressure: {custom_pressures[0, t].cpu().numpy()}")
            action_t = custom_pressures[:, t]
            
            # Predict the NEXT 3D state based on custom pressure & hidden memory
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            pred_views_np = []
            
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0) 
                ray_dirs = ray_dirs.unsqueeze(0)
                
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins, ray_dirs)

                pred_img = rgb_pred.view(H, W, 3).cpu().numpy()
                pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR) 
                pred_views_np.append(pred_img)
                
            # Stitch Images Together (AI Prediction Only)
            combined_frame = np.hstack(pred_views_np) 
            out_video.write(combined_frame) 
            
            # Feed the pure 3D prediction back into the physics engine
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Custom Control Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()