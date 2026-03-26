import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from src.multiview_dataset import SoftRobotDataset
from src.encoder_resnet_gn import ResNetGNTriPlaneEncoder
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.volumetric_ray_marcher import VolumetricRayMarcher
from src.visualization_helper import get_full_image_rays

def main():
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    CHECKPOINT_PATH = "runs/actionWeightedLoss_mixedDataset_MASK_2026-03-22_13-50-49/best_model.pth"
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    VIDEOS_DIR = "generated_vids/cross_pollination"
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = f"{VIDEOS_DIR}/CrossPollination_Test_{TIMESTAMP}.mp4"
    
    FEATURE_DIM = 64
    FPS = 30
    IMAGE_MODE = "mask" 

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")

    encoder = ResNetGNTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    for model in [encoder, dynamics, decoder]: model.eval()

    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, seq_len=None)

    # --- NEW: SMART FOLDER SEARCH ---
    def get_index_by_folder_name(dataset, target_name):
        for idx, folder_path in enumerate(dataset.case_folders):
            if target_name in os.path.basename(folder_path):
                return idx
        raise ValueError(f"Could not find any folder containing '{target_name}'")

    # Now you explicitly ask for the folder names you want!
    anchor_idx = get_index_by_folder_name(dataset, "Case_1") # Or Case_1, depending on your naming
    action_idx = get_index_by_folder_name(dataset, "Case_5") # Case_76 (1,1,100k) (extreme bending)
    
    print(f"Found Visual Anchor at Dataset Index: {anchor_idx}")
    print(f"Found Physical Pressures at Dataset Index: {action_idx}")

    # 1. Grab the resting visual frame
    loader_img = DataLoader(Subset(dataset, indices=[anchor_idx]), batch_size=1, shuffle=False)
    anchor_videos = next(iter(loader_img))["video"].to(device)
    
    # 2. Grab the physical action sequence
    loader_press = DataLoader(Subset(dataset, indices=[action_idx]), batch_size=1, shuffle=False)
    action_batch = next(iter(loader_press))
    real_pressures = action_batch["pressures"].to(device)
    
    _, ACTUAL_TIME, Views, C, H, W = anchor_videos.shape
    
    print(f"Starting Cross-Pollination Rollout for {ACTUAL_TIME - 1} frames...")
    
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W * Views, H))
    
    hidden_state = None
    # Encode the INITIAL frame of Case 0
    tri_planes_t = encoder(anchor_videos[:, 0])

    with torch.no_grad():
        for t in range(ACTUAL_TIME - 1): 
            print(f"Rendering Frame {t+1}/{ACTUAL_TIME - 1}...", end="\r")
            
            # Apply the pressures from Case 75!
            action_t = torch.clamp(real_pressures[:, t], min=0.00001, max=1.0)
            
            planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
            
            pred_views_np = []
            for v in range(Views):
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins.unsqueeze(0), ray_dirs.unsqueeze(0))
                
                pred_img = (rgb_pred.view(H, W, 3).cpu().numpy() > 0.5).astype(np.float32) * 255
                pred_views_np.append(cv2.cvtColor(pred_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
            out_video.write(np.hstack(pred_views_np)) 
            tri_planes_t = planes_next_pred

    out_video.release()
    print(f"\nDone! Saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()