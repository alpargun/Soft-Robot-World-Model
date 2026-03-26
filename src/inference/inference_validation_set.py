import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
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
    VIDEOS_DIR = "generated_vids/bulk_validation"
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    FEATURE_DIM = 64
    FPS = 30
    IMAGE_MODE = "mask" 

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")

    encoder = ResNetGNTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)
    
    # --- SMART CHECKPOINT LOADER ---
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    for model in [encoder, dynamics, decoder]: model.eval()

    # Find validation indices from the checkpoint.
    VAL_INDICES = checkpoint['val_indices']
    print(f"SUCCESS: Automatically loaded {len(VAL_INDICES)} validation indices from checkpoint!")

    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, seq_len=None)

    for case_idx in VAL_INDICES:
        print(f"\n--- Processing Validation Case {case_idx} ---")
        loader = DataLoader(Subset(dataset, indices=[case_idx]), batch_size=1, shuffle=False)
        batch = next(iter(loader))
        
        real_videos = batch["video"].to(device) 
        real_pressures = batch["pressures"].to(device)
        _, ACTUAL_TIME, Views, C, H, W = real_videos.shape
        
        output_path = f"{VIDEOS_DIR}/Val_Case_{case_idx}.mp4"
        out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W * Views, H * 2)) 
        
        hidden_state = None
        tri_planes_t = encoder(real_videos[:, 0])

        with torch.no_grad():
            for t in range(ACTUAL_TIME - 1): 
                print(f"Frame {t+1}/{ACTUAL_TIME - 1}", end="\r")
                
                # Top Row: Get the real ground truth frame
                real_frame_views = []
                for v in range(Views):
                    real_img = real_videos[0, t, v].permute(1, 2, 0).cpu().numpy() * 255
                    real_frame_views.append(cv2.cvtColor(real_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                top_row = np.hstack(real_frame_views)

                # Bottom Row: Generate the AI prediction
                action_t = torch.clamp(real_pressures[:, t], min=0.00001, max=1.0)
                planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
                
                pred_views_np = []
                for v in range(Views):
                    ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                    single_plane_pred = {k: val[0:1] for k, val in planes_next_pred.items()}
                    rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, ray_origins.unsqueeze(0), ray_dirs.unsqueeze(0))
                    
                    pred_img = (rgb_pred.view(H, W, 3).cpu().numpy() > 0.5).astype(np.float32) * 255
                    pred_views_np.append(cv2.cvtColor(pred_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                bottom_row = np.hstack(pred_views_np)
                
                combined_frame = np.vstack((top_row, bottom_row))
                out_video.write(combined_frame) 
                
                tri_planes_t = planes_next_pred

        out_video.release()
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()