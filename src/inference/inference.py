import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Import custom modules
from src.multiview_dataset import SoftRobotDataset
from src.encoder_unet import TriPlaneEncoder # UPDATED to the UNet
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.renderer import VolumetricRayMarcher, get_full_image_rays, render_rays_chunked

def main():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    # UPDATE THIS to your Epoch 620 run folder
    CHECKPOINT_PATH = "runs/8_uNet_decayingFeatureConsistencyLoss_sharedGRUmultipleHead_MASK_2026-04-08_17-18-35/best_model.pth" 
    OUTPUT_VIDEO_PATH = "simulation_output.mp4"
    
    # Needs to match your dataset path to grab the t=0 resting frame
    DATA_DIRS = [
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/125_cases",
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/216_cases",
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/Staircase_creep"
    ]
    IMAGE_MODE = "mask"
    FEATURE_DIM = 64
    IMAGE_SIZE = 128
    
    # PHYSICS TIME SCALING (FRAME_STRIDE = 2)
    FPS = 15 
    TOTAL_FRAMES = 30 # 2 physical seconds
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")
        
    print(f"Starting Inference on: {device}")

    # ==========================================
    # --- LOAD MODEL WEIGHTS ---
    # ==========================================
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}. Please update the path!")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # ==========================================
    # --- GENERATE SYNTHETIC ACTION CURVE ---
    # ==========================================
    # Phase 1 (0-10): Ramp up Pressure 1 to max
    # Phase 2 (10-20): Hold Pressure 1 at max
    # Phase 3 (20-30): Release Pressure 1 back to 0
    synthetic_pressures = torch.zeros((1, TOTAL_FRAMES, 3), device=device)
    
    for t in range(TOTAL_FRAMES):
        if t < 10:
            synthetic_pressures[0, t, 0] = t / 10.0 # Ramp up P1
        elif t < 20:
            synthetic_pressures[0, t, 0] = 1.0      # Hold P1
        else:
            synthetic_pressures[0, t, 0] = 1.0 - ((t - 20) / 10.0) # Ramp down P1
            
        synthetic_pressures = torch.clamp(synthetic_pressures, min=0.00001, max=1.0)

    # ==========================================
    # --- GET INITIAL RESTING STATE ---
    # ==========================================
    print("Fetching initial resting frame from dataset...")
    dataset = SoftRobotDataset(run_folders=DATA_DIRS, img_size=(IMAGE_SIZE, IMAGE_SIZE), 
                               crop_size=600, image_mode=IMAGE_MODE, seq_len=None, frame_stride=2)
    
    # Grab the very first sequence, and the very first frame (t=0)
    sample = dataset[0]
    initial_video = sample["video"].unsqueeze(0).to(device) # Shape: [1, Time, Views, C, H, W]
    initial_frame = initial_video[:, 0] # Shape: [1, Views, C, H, W]
    
    # ==========================================
    # --- VIDEO WRITER SETUP ---
    # ==========================================
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (IMAGE_SIZE * 2, IMAGE_SIZE * 2), isColor=True)

    # ==========================================
    # --- THE AUTOREGRESSIVE ROLLOUT ---
    # ==========================================
    print("Starting 3D Volumetric Physics Simulation...")
    
    with torch.no_grad():
        # Encode the true resting state
        current_tri_planes = encoder(initial_frame)
        hidden_state = None
        
        for t in tqdm(range(TOTAL_FRAMES), desc="Simulating Frames"):
            # 1. Step the Physics Engine forward
            current_action = synthetic_pressures[:, t]
            current_tri_planes, hidden_state = dynamics(current_tri_planes, current_action, hidden_state)
            
            # 2. Render all 4 views for the current predicted step
            view_frames = []
            for v in range(4):
                full_ray_o, full_ray_d = get_full_image_rays(IMAGE_SIZE, IMAGE_SIZE, view_idx=v, device=device)
                full_ray_o = full_ray_o.unsqueeze(0)
                full_ray_d = full_ray_d.unsqueeze(0)
                
                # Render using the memory-safe chunked method
                rgb_pred = render_rays_chunked(
                    ray_marcher, decoder, current_tri_planes, full_ray_o, full_ray_d, chunk_size=4096)
                
                frame_2d = rgb_pred.view(IMAGE_SIZE, IMAGE_SIZE, 1).cpu().numpy()
                frame_8bit = (frame_2d * 255.0).clip(0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)
                
                cv2.putText(frame_bgr, f"Side {v+1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                view_frames.append(frame_bgr)
                
            # 3. Stitch the 4 views into a 2x2 grid
            top_row = np.hstack((view_frames[0], view_frames[1]))
            bottom_row = np.hstack((view_frames[2], view_frames[3]))
            grid_frame = np.vstack((top_row, bottom_row))
            
            # Add pressure gauge
            pressure_val = current_action[0, 0].item() * 100 
            cv2.putText(grid_frame, f"Input P1: {pressure_val:.1f} kPa", (IMAGE_SIZE - 40, IMAGE_SIZE), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out_video.write(grid_frame)

    out_video.release()
    print(f"\nSimulation Complete! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()