import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Import custom modules
from src.encoder_2d import Encoder2D
from src.decoder_2d import Decoder2D
from src.temporal_dynamics_2d import Dynamics2D
from src.multiview_dataset import SoftRobotDataset

def create_synthetic_pressures(time_steps, device):
    """
    Creates a completely custom pressure sequence that the model has never seen.
    Currently set to a smooth Sine Wave oscillation on Chamber 1.
    """
    pressures = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    
    # Create a smooth sine wave from 0.0 to 1.0
    t = np.linspace(0, 4 * np.pi, time_steps) # 2 full oscillation cycles
    sine_wave = (np.sin(t) + 1.0) / 2.0 
    
    # Apply to Chamber 0
    pressures[0, :, 0] = torch.tensor(sine_wave, dtype=torch.float32).to(device)
    
    # Keep the other two chambers slightly pressurized to maintain rigidity
    pressures[0, :, 1] = 0.1
    pressures[0, :, 2] = 0.1
    
    return pressures

def render_annotated_video(frames_pred, pressures, output_path, fps=15.0):
    base_height, base_width = frames_pred[0].shape[-2:]
    
    SCALE = 3 # Multiply resolution by 3 so text can fit
    height = base_height * SCALE
    width = base_width * SCALE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    for pred, p in zip(frames_pred, pressures):
        # Convert tensor [1, H, W] to numpy [H, W] (0-255)
        pred_img = (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert Grayscale to BGR color space so we can draw colored text
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
        
        # Upscale using Nearest Neighbor to preserve sharp model pixels
        pred_bgr = cv2.resize(pred_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Write text over the video
        # 1. Label
        cv2.putText(pred_bgr, "Synthetic Prediction", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 2. Pressure Vectors (Bottom Center)
        p_text = f"P: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
        cv2.putText(pred_bgr, p_text, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 3. Network State Tracker
        cv2.putText(pred_bgr, "STATE: AUTOREGRESSIVE (Blind)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out_video.write(pred_bgr)
        
    out_video.release()

def main():
    CHECKPOINT_PATH = "runs/singleView12_MASK_2026-05-24_13-30-07/best_model.pth" # UPDATE
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis"
    OUTPUT_VIDEO = "synthetic_physics_sine_wave.mp4"
    
    FEATURE_DIM = 64
    TIME_STEPS = 450
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Physics Sandbox on: {device}")

    # Initialize the architecture
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
    print("Baseline Model Loaded Successfully.")

    # Get a single starting frame to initialize the geometry
    DATA_DIRS = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "old"]
    dataset = SoftRobotDataset(run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode="mask", seq_len=2, frame_stride=2)
    
    # Grab the very first frame of the very first video as the initial state for our simulation
    sample = dataset[0]
    first_frame = sample["video"][0:1, 0].to(device) 
    
    # Generate Custom Physics Inputs
    custom_pressures = create_synthetic_pressures(TIME_STEPS, device)

    print("Rolling out custom physics simulation...")
    predicted_frames = []
    used_pressures = []

    with torch.no_grad():
        # Encode the initial real-world state
        current_features = encoder(first_frame)
        hidden_state = None
        
        for t in tqdm(range(TIME_STEPS)):
            action_t = custom_pressures[:, t]
            
            # Step the physics engine
            current_features, hidden_state = dynamics(current_features, action_t, hidden_state)
            
            # Decode to mask
            rgb_pred = decoder(current_features)
            
            predicted_frames.append(rgb_pred)
            used_pressures.append(action_t.squeeze().cpu().numpy())
            
    print("Rendering annotated video...")
    render_annotated_video(predicted_frames, used_pressures, OUTPUT_VIDEO)
    print(f"Simulation Complete! Saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()