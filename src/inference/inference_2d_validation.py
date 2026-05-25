import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Subset

# Import custom modules
from src.encoder_2d import Encoder2D
from src.decoder_2d import Decoder2D
from src.temporal_dynamics_2d import Dynamics2D
from src.multiview_dataset import SoftRobotDataset

def render_side_by_side_video(frames_gt, frames_pred, pressures, burn_in_len, output_path, fps=15.0):
    """
    Renders side-by-side video with colored telemetry overlay (Pressures and Network State).
    Upscales the frames so the telemetry text fits perfectly and looks professional.
    """
    base_height, base_width = frames_gt[0].shape[-2:]
    
    SCALE = 3 # Multiply resolution by 3 so text can fit
    height = base_height * SCALE
    width = base_width * SCALE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Total width is width * 2 (for side-by-side)
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height), isColor=True)

    for idx, (gt, pred, p) in enumerate(zip(frames_gt, frames_pred, pressures)):
        # Convert tensors [1, H, W] to numpy [H, W] (0-255)
        gt_img = (gt.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pred_img = (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert Grayscale to BGR color space so we can draw colored text
        gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
        
        # UPSCALE: INTER_NEAREST preserves the sharp model pixels without faking HD blur
        gt_bgr = cv2.resize(gt_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        pred_bgr = cv2.resize(pred_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Concatenate horizontally
        combined_frame = np.concatenate((gt_bgr, pred_bgr), axis=1)
        
        # Write text over the video
        # Column Labels
        cv2.putText(combined_frame, "Ground Truth", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, "Prediction", (width + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Pressure Vectors (Bottom Center of the Pred frame)
        p_text = f"P: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
        cv2.putText(combined_frame, p_text, (width + 15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Burn-in vs Autoregressive State (Top Center of the Pred frame)
        if idx < burn_in_len - 1:
            state_text = "STATE: BURN-IN"
            color = (0, 165, 255) # Orange
        else:
            state_text = "STATE: AUTOREGRESSIVE"
            color = (0, 0, 255) # Red
            
        cv2.putText(combined_frame, state_text, (width + 15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        out_video.write(combined_frame)
        
    out_video.release()

def main():
    CHECKPOINT_PATH = "runs/singleView12_MASK_2026-05-24_13-30-07/best_model.pth" 
    MASTER_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis"
    OUTPUT_DIR = "validation_videos"
    
    FEATURE_DIM = 64
    BURN_IN_LENGTH = 5
    VAL_PERCENTAGE = 0.15
    FRAME_STRIDE = 2
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Annotated Validation Inference on: {device}")

    # Reconstruct the Validation Dataset
    DATA_DIRS = [os.path.join(MASTER_DIR, d) for d in os.listdir(MASTER_DIR) if os.path.isdir(os.path.join(MASTER_DIR, d)) and d != "old"]
    val_base = SoftRobotDataset(run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode="mask", seq_len=None, frame_stride=FRAME_STRIDE)

    all_bending_indices = []
    for idx, folder_path in enumerate(val_base.case_folders):
        if os.path.basename(folder_path).startswith("Case_"):
            all_bending_indices.append(idx)
            
    random.seed(42) # Strictly ensures we pull the same validation cases as training
    num_val_cases = int(len(all_bending_indices) * VAL_PERCENTAGE)
    val_indices = random.sample(all_bending_indices, num_val_cases)
    val_dataset = Subset(val_base, val_indices)
    
    print(f"Loaded {len(val_dataset)} Validation Sequences.")

    # Initialize Baseline Architecture
    encoder = Encoder2D(feature_dim=FEATURE_DIM).to(device)
    dynamics = Dynamics2D(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=64).to(device)
    decoder = Decoder2D(feature_dim=FEATURE_DIM).to(device)

    # Load Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # Rollout Engine
    def autoregressive_rollout(video_seq, pressure_seq, start_t, max_steps):
        end_t = min(start_t + max_steps, video_seq.shape[0])
        if end_t - start_t <= BURN_IN_LENGTH:
            return None, None, None
            
        gt_frames = []
        pred_frames = []
        used_pressures = []
        
        with torch.no_grad():
            curr_feat = encoder(video_seq[start_t:start_t+1])
            h_val = None
            
            # PHASE 1: Burn-In
            for t in range(start_t, start_t + BURN_IN_LENGTH - 1):
                action = torch.clamp(pressure_seq[t:t+1], min=0.00001, max=1.0)
                _, h_val = dynamics(curr_feat, action, h_val)
                curr_feat = encoder(video_seq[t+1:t+2])
                
                gt_frames.append(video_seq[t:t+1])
                pred_frames.append(video_seq[t:t+1])
                used_pressures.append(action.squeeze().cpu().numpy())
                
            # PHASE 2: Strict Autoregression
            for t in range(start_t + BURN_IN_LENGTH - 1, end_t - 1):
                action = torch.clamp(pressure_seq[t:t+1], min=0.00001, max=1.0)
                pred_feat, h_val = dynamics(curr_feat, action, h_val)
                rgb_p = decoder(pred_feat)
                
                gt_frames.append(video_seq[t+1:t+2])
                pred_frames.append(rgb_p)
                used_pressures.append(action.squeeze().cpu().numpy())
                
                curr_feat = pred_feat
                
        return gt_frames, pred_frames, used_pressures

    # Process validation cases
    num_cases_to_render = len(val_dataset)
    
    for i in range(num_cases_to_render):
        sample = val_dataset[i]
        vid = sample["video"][:, 0].to(device)
        press = sample["pressures"].to(device)
        total_time = vid.shape[0]
        
        print(f"\nProcessing Validation Case {i+1}/{num_cases_to_render} (Length: {total_time} frames)")
        
        print("  -> Rolling out from Standstill (t=0)...")
        gt_0, pred_0, press_0 = autoregressive_rollout(vid, press, start_t=0, max_steps=450)
        if gt_0:
            render_side_by_side_video(gt_0, pred_0, press_0, BURN_IN_LENGTH, os.path.join(OUTPUT_DIR, f"Case_{i}_Start0.mp4"))
            
        mid_start = total_time // 3
        print(f"  -> Rolling out from Mid-Sequence (t={mid_start})...")
        gt_mid, pred_mid, press_mid = autoregressive_rollout(vid, press, start_t=mid_start, max_steps=450)
        if gt_mid:
            render_side_by_side_video(gt_mid, pred_mid, press_mid, BURN_IN_LENGTH, os.path.join(OUTPUT_DIR, f"Case_{i}_StartMid.mp4"))

    print(f"\nDone! Annotated videos saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()