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

def generate_10_publishable_scenarios(time_steps, device):
    """
    Every scenario uses Linear Ramps like the dataset, but in unique combinations to test different dynamic behaviors.
    """
    scenarios = {}
    
    # Helper function to generate clean linear tensor ramps
    def l_ramp(start, end, steps):
        return torch.tensor(np.linspace(start, end, steps), dtype=torch.float32).to(device)

    # --- 01. Sequential Handoff (Ch0 then Ch2) ---
    p1 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    q_t = time_steps // 4
    p1[0, :q_t, 0] = l_ramp(0, 1.0, q_t)
    p1[0, q_t:q_t*2, 0] = l_ramp(1.0, 0, q_t)
    p1[0, q_t*2:q_t*3, 2] = l_ramp(0, 1.0, q_t)
    p1[0, q_t*3:, 2] = l_ramp(1.0, 0, time_steps - q_t*3)
    scenarios["01_Sequential_Handoff"] = p1

    # --- 02. Dual Pinch (Ch0 & Ch2 together) ---
    p2 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    half_t = time_steps // 2
    p2[0, :half_t, 0] = l_ramp(0, 1.0, half_t)
    p2[0, half_t:, 0] = l_ramp(1.0, 0, time_steps - half_t)
    p2[0, :half_t, 2] = l_ramp(0, 1.0, half_t)
    p2[0, half_t:, 2] = l_ramp(1.0, 0, time_steps - half_t)
    scenarios["02_Dual_Pinch"] = p2

    # --- 03. The Ripple (Ch0 -> Ch1 -> Ch2) ---
    p3 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    t_50 = time_steps // 3
    t_25 = t_50 // 2
    p3[0, :t_25, 0] = l_ramp(0, 1.0, t_25)
    p3[0, t_25:t_50, 0] = l_ramp(1.0, 0, t_50 - t_25)
    p3[0, t_50:t_50+t_25, 1] = l_ramp(0, 1.0, t_25)
    p3[0, t_50+t_25:t_50*2, 1] = l_ramp(1.0, 0, t_50 - t_25)
    p3[0, t_50*2:t_50*2+t_25, 2] = l_ramp(0, 1.0, t_25)
    p3[0, t_50*2+t_25:, 2] = l_ramp(1.0, 0, time_steps - (t_50*2+t_25))
    scenarios["03_The_Ripple"] = p3

    # --- 04. Opposing Force (Ch0 vs Ch2) ---
    # Ch0 ramps up. As it ramps down, Ch2 pushes against it.
    p4 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    p4[0, :half_t, 0] = l_ramp(0, 1.0, half_t)
    p4[0, half_t:, 0] = l_ramp(1.0, 0, time_steps - half_t)
    p4[0, half_t:, 2] = l_ramp(0, 1.0, time_steps - half_t)
    scenarios["04_Opposing_Force"] = p4

    # --- 05. The Hold & Pulse (Independent Articulation) ---
    # Ch1 inflates and acts as a rigid spine, while Ch0 pulses rapidly
    p5 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    p5[0, :30, 1] = l_ramp(0, 0.6, 30)
    p5[0, 30:120, 1] = 0.6
    p5[0, 120:, 1] = l_ramp(0.6, 0, time_steps - 120)
    p5[0, 45:75, 0] = l_ramp(0, 0.8, 30)
    p5[0, 75:105, 0] = l_ramp(0.8, 0, 30)
    scenarios["05_Hold_and_Pulse"] = p5

    # --- 06. Full Volume Swell ---
    # All 3 chambers inflate to 0.8 to test maximum spatial limits safely
    p6 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    for i in range(3):
        p6[0, :half_t, i] = l_ramp(0, 0.8, half_t)
        p6[0, half_t:, i] = l_ramp(0.8, 0, time_steps - half_t)
    scenarios["06_Full_Volume_Swell"] = p6

    # --- 07. The See-Saw ---
    # Continuous, rapid back-and-forth momentum crossing
    p7 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    t_cycle = time_steps // 3
    t_up = t_cycle // 2
    # Cycle 1: Ch0
    p7[0, :t_up, 0] = l_ramp(0, 1.0, t_up)
    p7[0, t_up:t_cycle, 0] = l_ramp(1.0, 0, t_cycle - t_up)
    # Cycle 2: Ch2
    p7[0, t_cycle:t_cycle+t_up, 2] = l_ramp(0, 1.0, t_up)
    p7[0, t_cycle+t_up:t_cycle*2, 2] = l_ramp(1.0, 0, t_cycle - t_up)
    # Cycle 3: Ch0
    p7[0, t_cycle*2:t_cycle*2+t_up, 0] = l_ramp(0, 1.0, t_up)
    p7[0, t_cycle*2+t_up:, 0] = l_ramp(1.0, 0, time_steps - (t_cycle*2+t_up))
    scenarios["07_The_See_Saw"] = p7

    # --- 08. Asymmetric Dual ---
    # Ch0 inflates slowly/deflates fast. Ch2 inflates fast/deflates slowly.
    p8 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    p8[0, :100, 0] = l_ramp(0, 1.0, 100)
    p8[0, 100:, 0] = l_ramp(1.0, 0, time_steps - 100)
    p8[0, :50, 2] = l_ramp(0, 1.0, 50)
    p8[0, 50:, 2] = l_ramp(1.0, 0, time_steps - 50)
    scenarios["08_Asymmetric_Dual"] = p8

    # --- 09. The Staircase Descent ---
    # Tests stable hysteresis resting at partial pressures
    p9 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    p9[0, :30, 2] = l_ramp(0, 1.0, 30)
    p9[0, 30:60, 2] = 1.0
    p9[0, 60:90, 2] = l_ramp(1.0, 0.5, 30)
    p9[0, 90:120, 2] = 0.5
    p9[0, 120:, 2] = l_ramp(0.5, 0, time_steps - 120)
    scenarios["09_Staircase_Descent"] = p9

    # --- 10. Smooth Random Walk ---
    # Mirrors the 15 dataset random cases for direct comparison
    p10 = torch.zeros((1, time_steps, 3), dtype=torch.float32).to(device)
    keyframes = np.random.rand(4, 3) 
    keyframes[0] = [0.0, 0.0, 0.0]
    keyframes[3] = [0.0, 0.0, 0.0]
    
    seg = time_steps // 3
    for i in range(3):
        start = i * seg
        end = (i + 1) * seg if i < 2 else time_steps
        for c in range(3):
            p10[0, start:end, c] = l_ramp(keyframes[i, c], keyframes[i+1, c], end - start)
    scenarios["10_Smooth_Random_Walk"] = p10

    return scenarios

def render_annotated_video(frames_pred, pressures, title, output_path, fps=15.0):
    base_height, base_width = frames_pred[0].shape[-2:]
    
    SCALE = 3 
    height = base_height * SCALE
    width = base_width * SCALE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    for pred, p in zip(frames_pred, pressures):
        pred_img = (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.resize(pred_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        
        cv2.putText(pred_bgr, f"Sim: {title.replace('_', ' ')}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        p_text = f"P: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
        cv2.putText(pred_bgr, p_text, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(pred_bgr, "STATE: AUTOREGRESSIVE (Blind)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out_video.write(pred_bgr)
        
    out_video.release()

def main():
    CHECKPOINT_PATH = "runs/singleView12_MASK_2026-05-24_13-30-07/best_model.pth" # UPDATE THIS
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis"
    OUTPUT_DIR = "synthetic_simulations"
    
    FEATURE_DIM = 64
    TIME_STEPS = 150 # 10 seconds at 15 FPS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Physics Sandbox on: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    encoder = Encoder2D(feature_dim=FEATURE_DIM).to(device)
    dynamics = Dynamics2D(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=64).to(device)
    decoder = Decoder2D(feature_dim=FEATURE_DIM).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    dynamics.eval()
    decoder.eval()
    print("Baseline Model Loaded Successfully.")

    DATA_DIRS = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "old"]
    DATA_DIRS.sort() # Sorting ensures Case_1 is grabbed, giving us a true 0.0 Pa starting frame
    
    dataset = SoftRobotDataset(run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode="mask", seq_len=2, frame_stride=2)
    
    sample = dataset[0]
    first_frame = sample["video"][0:1, 0].to(device) 
    
    scenarios = generate_10_publishable_scenarios(TIME_STEPS, device)

    print(f"Executing {len(scenarios)} custom simulation protocols...")

    for scenario_name, custom_pressures in scenarios.items():
        print(f"\n-> Simulating: {scenario_name}")
        predicted_frames = []
        used_pressures = []

        with torch.no_grad():
            current_features = encoder(first_frame)
            hidden_state = None
            
            for t in tqdm(range(TIME_STEPS), leave=False):
                # Clamp the pressures to prevent latent shock
                action_t = torch.clamp(custom_pressures[:, t], min=0.00001, max=1.0)
                
                # Step the physics engine
                current_features, hidden_state = dynamics(current_features, action_t, hidden_state)
                rgb_pred = decoder(current_features)
                
                predicted_frames.append(rgb_pred)
                used_pressures.append(action_t.squeeze().cpu().numpy())
                
        video_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.mp4")
        render_annotated_video(predicted_frames, used_pressures, scenario_name, video_path)
        print(f"   Saved to: {video_path}")

    print("\nAll Custom Simulations Complete!")

if __name__ == "__main__":
    main()