import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Import custom modules
from multiview_dataset import SoftRobotDataset
from encoder_resnet_gn import ResNetGNTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from visualization_helper import get_full_image_rays

def create_custom_pressure_sequence(num_frames, device):
    """
    Creates a custom 3D pressure tensor of shape [1, num_frames, 3].
    Action Dim 0, 1, 2 correspond to the 3 bellows.
    """
    # Initialize all pressures to our strict physical floor of 1.0
    pressures = torch.ones((1, num_frames, 3), dtype=torch.float32, device=device)
    
    # Custom Ramp Logic (Example: Bellow 0 ramps up to 80 kPa and back down)
    half_frames = num_frames // 2
    peak_pressure = 60000.0
    ramp_up = torch.linspace(1.0, peak_pressure, half_frames)
    ramp_down = torch.linspace(peak_pressure, 1.0, num_frames - half_frames)
    
    # Assign the ramp to the first bellow, keeping others at 1.0
    pressures[0, :, 0] = torch.cat((ramp_up, ramp_down))
    
    # You can customize Bellow 1 and 2 here for more combinations
    # pressures[0, :, 1] = ... 
    
    # Intercept any exact 0.0 inputs and enforce the strict physics floor of 1 Pa
    pressures[pressures == 0.0] = 1.0
    
    # CRITICAL FIX: Normalize by 100,000 to match the [0.00001, 1.0] AI space
    pressures = pressures / 100000.0
    
    return pressures

def main():
    # --- 1. CONFIGURATION ---
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/resnetGN_decoderConcat_125cases_clampfix_MASK_2026-03-12_01-10-53/last_checkpoint.pth"
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set output path
    VIDEOS_DIR = "generated_vids"
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = f"{VIDEOS_DIR}/Custom_Simulation_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 64
    FPS = 30
    IMAGE_MODE = "mask" 
    NUM_FRAMES = 60 # 2 seconds of video
    APPLY_POSTPROCESSING = True 

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Custom Inference on: {device}")

    # --- 2. LOAD THE TRAINED MODEL ---
    encoder = ResNetGNTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # --- 3. GET INITIAL RESTING STATE ---
    # We load just ONE frame from the dataset to tell the model what the robot looks like at rest
    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    dataloader = DataLoader(Subset(dataset, indices=[0]), batch_size=1, shuffle=False)
    
    batch = next(iter(dataloader))
    initial_videos = batch["video"].to(device) 
    _, _, Views, C, H, W = initial_videos.shape

    # --- 4. GENERATE CUSTOM ACTIONS ---
    custom_pressures = create_custom_pressure_sequence(NUM_FRAMES, device)

    # --- 5. AUTOREGRESSIVE LATENT ROLLOUT ---
    print("Starting Custom Latent Autoregressive Generation...")
    
    # Video will be a 1x4 horizontal strip showing all 4 views
    video_width = W * Views
    video_height = H 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (video_width, video_height))
    
    hidden_state = None
    
    # Encode the starting frame (time 0)
    current_frames = initial_videos[:, 0] 
    tri_planes_t = encoder(current_frames)

    with torch.no_grad():
        for t in range(NUM_FRAMES):
            print(f"Rendering Frame {t+1}/{NUM_FRAMES}...")
            
            # Clamp the custom pressures just to enforce absolute safety in normalized space
            action_t = torch.clamp(custom_pressures[:, t], min=0.00001, max=1.0)
            
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            pred_views_np = []
            
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0) 
                ray_dirs = ray_dirs.unsqueeze(0)
                
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                pred_img = rgb_pred.view(H, W, 3).cpu().numpy()
                
                # Base Binary Thresholding
                pred_img = (pred_img > 0.5).astype(np.float32)
                pred_img = (pred_img * 255).astype(np.uint8)
                
                # Optional CV Cleanup
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
                
            # Stitch the 4 predicted views together side-by-side
            combined_frame = np.hstack(pred_views_np) 
            out_video.write(combined_frame) 
            
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Custom World Model Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()