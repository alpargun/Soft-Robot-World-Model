import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from src.multiview_dataset import SoftRobotDataset
from src.encoder import TriPlaneEncoder
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.volumetric_ray_marcher import VolumetricRayMarcher
from src.visualization_helper import get_full_image_rays

def generate_synthetic_pressures(time_steps, device):
    """
    Define your own custom pressure sequences here! 
    Values must be normalized between [0.00001, 1.0] to respect the 1.0 Pa MIN_PRESSURE rule.
    """
    pressures = torch.ones((time_steps, 3), device=device) * 0.00001 # Start at vacuum baseline
    
    # EXAMPLE: Smoothly inflate Bellow 1 to max over the sequence
    for t in range(time_steps):
        progress = t / (time_steps - 1)
        pressures[t, 0] = 0.00001 + (progress * 0.99999) # Bellow 1 ramps up
        # Bellow 2 and 3 stay at minimum
        
    return pressures

def main():
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/125cases_64features_actionWeightedLoss_MASK_2026-03-24_17-08-49/last_checkpoint.pth"
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    VIDEOS_DIR = "generated_vids/true_custom"
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = f"{VIDEOS_DIR}/Synthetic_Rollout_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 64
    FPS = 30
    IMAGE_MODE = "mask" 
    NUM_FRAMES = 60 # Let's simulate a 2-second custom move
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")

    # Load Models
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])

    for model in [encoder, dynamics, decoder]: model.eval()

    # Get resting anchor from Case 0
    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, seq_len=None)
    loader_img = DataLoader(Subset(dataset, indices=[0]), batch_size=1, shuffle=False)
    initial_videos = next(iter(loader_img))["video"].to(device)
    
    _, _, Views, C, H, W = initial_videos.shape
    
    # Generate custom math pressures
    synthetic_pressures = generate_synthetic_pressures(NUM_FRAMES, device)

    print(f"Starting {NUM_FRAMES}-frame Synthetic Latent Generation...")
    
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W * Views, H))
    
    hidden_state = None
    tri_planes_t = encoder(initial_videos[:, 0])

    with torch.no_grad():
        for t in range(NUM_FRAMES): 
            print(f"Rendering Frame {t+1}/{NUM_FRAMES}...", end="\r")
            
            action_t = synthetic_pressures[t].unsqueeze(0)
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            pred_views_np = []
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins.unsqueeze(0), ray_dirs.unsqueeze(0))
                
                pred_img = (rgb_pred.view(H, W, 3).cpu().numpy() > 0.5).astype(np.float32) * 255
                pred_img = cv2.cvtColor(pred_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                pred_views_np.append(pred_img)
                
            out_video.write(np.hstack(pred_views_np)) 
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()