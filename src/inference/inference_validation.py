import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.multiview_dataset import SoftRobotDataset
from src.encoder import TriPlaneEncoder
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.renderer import VolumetricRayMarcher, get_full_image_rays, render_rays_chunked # Added chunked import

def create_video_writer(filename, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def generate_validation_video(model_components, real_video, real_pressures, save_path, device, H=128, W=128):
    encoder, dynamics, decoder, ray_marcher = model_components
    Time, Views, C, _, _ = real_video.shape
    
    # We double the height (H * 2) to stack the Real and Predicted rows vertically
    writer = create_video_writer(save_path, fps=15, frame_size=(W * Views, H * 2))
    
    # 1. Initialize with the exact first frame of the validation sequence
    curr_frame = real_video[0].unsqueeze(0).to(device)
    curr_planes = encoder(curr_frame)
    h_state = None 
    
    print(f"Generating: {os.path.basename(save_path)}...")
    with torch.no_grad():
        for t in tqdm(range(Time - 1), leave=False):
            # 2. Feed the exact ground-truth pressure for this timestep
            action = real_pressures[t].unsqueeze(0).to(device)
            action = torch.clamp(action, min=0.00001, max=1.0)
            
            # 3. Predict the next state entirely autoregressively
            curr_planes, h_state = dynamics(curr_planes, action, h_state)
            
            pred_row_frames = []
            real_row_frames = []
            
            for v in range(Views):
                # --- A. Render the Prediction (Using Memory-Safe Chunking) ---
                ray_origins, ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                ray_origins = ray_origins.unsqueeze(0)
                ray_dirs = ray_dirs.unsqueeze(0)
                
                rgb_pred = render_rays_chunked(ray_marcher, decoder, curr_planes, ray_origins, ray_dirs, chunk_size=4096)
                
                # Dynamically get the number of output channels (1 for mask, 3 for rgb)
                C_out = rgb_pred.shape[-1]
                pred_frame = rgb_pred.view(H, W, C_out).cpu().numpy()
                
                # If it is a 1-channel mask, duplicate it to 3 channels so OpenCV can stack it
                if C_out == 1:
                    pred_frame = np.repeat(pred_frame, 3, axis=2)
                
                pred_frame = (np.clip(pred_frame, 0, 1) * 255).astype(np.uint8)
                pred_row_frames.append(pred_frame)
                
                # --- B. Format the Ground Truth ---
                real_v_frame = real_video[t+1, v].cpu().numpy() # Extract the actual next frame
                real_v_frame = np.transpose(real_v_frame, (1, 2, 0)) # Change to [H, W, C]
                
                # If it is a 1-channel mask, duplicate it to 3 channels so OpenCV can write it
                if real_v_frame.shape[2] == 1:
                    real_v_frame = np.repeat(real_v_frame, 3, axis=2)
                    
                real_v_frame = (np.clip(real_v_frame, 0, 1) * 255).astype(np.uint8)
                real_row_frames.append(real_v_frame)
                
            # Concatenate horizontally across views
            pred_row = np.concatenate(pred_row_frames, axis=1)
            real_row = np.concatenate(real_row_frames, axis=1)
            
            # Stack vertically: Ground Truth on Top, Prediction on Bottom
            final_frame = np.concatenate([real_row, pred_row], axis=0)
            
            # 4. Overlay Labels and Pressures
            p_text = f"P1: {action[0,0]:.2f} | P2: {action[0,1]:.2f} | P3: {action[0,2]:.2f}"
            
            # Top Left (Green Text)
            cv2.putText(final_frame, f"GROUND TRUTH | {p_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Bottom Left (Red Text) - Notice the Y-coordinate is pushed down by 'H'
            cv2.putText(final_frame, f"PREDICTION | {p_text}", (10, H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            writer.write(final_frame)
            
    writer.release()

def main():
    DATA_DIRS = [
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/125_cases",
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/216_cases",
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/Staircase_creep"
    ]
    
    # --- VERIFY THIS TIMESTAMP MATCHES YOUR LOCAL REVERT_1 FOLDER ---
    CHECKPOINT_PATH = "runs/revert_3_onlyCurriculumLearning_MASK_2026-04-11_12-37-03/best_model.pth" 
    
    # Automatically name the output folder after the run so things stay organized
    run_name = CHECKPOINT_PATH.split("/")[-2]
    OUTPUT_DIR = os.path.join("val_reconstruction_videos", run_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    IMAGE_MODE = "mask"
    FEATURE_DIM = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n--- Initializing Validation Inference on {device} ---")

    # Encoder doesn't take image_mode
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM).to(device).eval()
    
    # Re-added action_embed_dim=32 to match revert_1 training architecture
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=32).to(device).eval()
    
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device).eval()
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device).eval()
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    model_components = (encoder, dynamics, decoder, ray_marcher)
    
    # Load the Validation Subset exactly as it was during training
    base_dataset = SoftRobotDataset(run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, seq_len=None, frame_stride=2)
    val_dataset = torch.utils.data.Subset(base_dataset, checkpoint['val_indices'])
    
    num_to_test = min(10, len(val_dataset)) # Run the first 10 validation cases
    print(f"Generating {num_to_test} validation reconstructions...")
    
    for i in range(num_to_test):
        real_video = val_dataset[i]["video"]
        real_pressures = val_dataset[i]["pressures"]
        
        save_path = os.path.join(OUTPUT_DIR, f"val_case_{i}_reconstruction.mp4")
        generate_validation_video(model_components, real_video, real_pressures, save_path, device)
        
    print(f"\nDone! Please review the videos in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()