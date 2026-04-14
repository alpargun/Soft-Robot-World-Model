import os
import cv2
import torch
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
import gc

# Import your updated modules
from src.multiview_dataset import SoftRobotDataset
from src.encoder import TriPlaneEncoder
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.renderer import VolumetricRayMarcher, get_full_image_rays, render_rays_chunked

# ==========================================
# --- DIAGNOSTIC HELPER FUNCTIONS ---
# ==========================================
def _mean_abs_plane_diff(planes_a, planes_b):
    """Returns mean absolute difference across all three tri-planes."""
    diffs = []
    for key in ("xy", "xz", "yz"):
        diffs.append(torch.mean(torch.abs(planes_a[key] - planes_b[key])).item())
    return float(np.mean(diffs))

def print_action_sensitivity_report(encoder, dynamics, resting_frames, device):
    """
    Measures whether one-step dynamics output changes when action channels change.
    If these numbers are near zero, the model is effectively action-blind.
    """
    print("\n[DIAGNOSTIC] One-step action sensitivity from identical initial state")
    with torch.no_grad():
        base_planes = encoder(resting_frames)

        action_set = {
            "A0_zero": torch.tensor([[0.00001, 0.00001, 0.00001]], device=device),
            "A1_bellow1": torch.tensor([[0.80, 0.00001, 0.00001]], device=device),
            "A2_bellow2": torch.tensor([[0.00001, 0.80, 0.00001]], device=device),
            "A3_bellow3": torch.tensor([[0.00001, 0.00001, 0.80]], device=device),
            "A4_uniform": torch.tensor([[0.80, 0.80, 0.80]], device=device),
        }

        next_planes = {}
        for name, action in action_set.items():
            pred, _ = dynamics(base_planes, action, hidden_states_prev=None)
            next_planes[name] = pred

        report_pairs = [
            ("A1_bellow1", "A2_bellow2"),
            ("A1_bellow1", "A3_bellow3"),
            ("A2_bellow2", "A3_bellow3"),
            ("A0_zero", "A4_uniform"),
        ]
        for left, right in report_pairs:
            value = _mean_abs_plane_diff(next_planes[left], next_planes[right])
            print(f"  | {left} vs {right} -> mean |delta tri-plane| = {value:.8f}")

def print_action_gradient_report(encoder, dynamics, resting_frames, device):
    """
    Uses autograd to measure d(next_state)/d(action) per pressure channel.
    Very small gradients indicate weak action conditioning.
    """
    print("\n[DIAGNOSTIC] Action-gradient sensitivity at selected operating points")
    dynamics.eval()
    encoder.eval()

    with torch.no_grad():
        base_planes = encoder(resting_frames)

    probe_actions = {
        "low": torch.tensor([[0.05, 0.05, 0.05]], device=device),
        "mid": torch.tensor([[0.40, 0.40, 0.40]], device=device),
        "mixed": torch.tensor([[0.80, 0.20, 0.60]], device=device),
    }

    for label, action_seed in probe_actions.items():
        action = action_seed.clone().detach().requires_grad_(True)
        next_planes, _ = dynamics(base_planes, action, hidden_states_prev=None)

        # Scalar proxy for "how much state moved"; backprop to action channels.
        proxy = 0.0
        for key in ("xy", "xz", "yz"):
            proxy = proxy + next_planes[key].abs().mean()

        if action.grad is not None:
            action.grad.zero_()
        proxy.backward()

        grad = action.grad.detach().abs().squeeze(0)
        grad_norm = torch.linalg.vector_norm(grad, ord=2).item()
        print(
            "  | "
            f"{label}: |dstate/dP1|={grad[0].item():.8e}, "
            f"|dstate/dP2|={grad[1].item():.8e}, "
            f"|dstate/dP3|={grad[2].item():.8e}, "
            f"L2={grad_norm:.8e}"
        )

def print_scenario_distance_report(test_suite):
    """
    Prints pairwise mean absolute action distance between scenario sequences.
    Confirms whether synthetic scenarios are actually different at the input level.
    """
    print("\n[DIAGNOSTIC] Pairwise action-sequence distances")
    names = list(test_suite.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            dist = torch.mean(torch.abs(test_suite[ni] - test_suite[nj])).item()
            print(f"  | {ni} vs {nj} -> mean |delta pressure| = {dist:.6f}")

# ==========================================
# --- SCENARIO GENERATOR ---
# ==========================================
def generate_test_scenarios(seq_length, device):
    """
    Creates a dictionary of smoothly ramped pressure profiles.
    Includes both orthogonal diagnostic tests and complex visual tests.
    """
    scenarios = {}
    
    # 1. Absolute Zero (Dead state to test static inertia)
    scenarios["1_Absolute_Zero"] = torch.tensor(np.ones((seq_length, 3), dtype=np.float32) * 0.00001, device=device).unsqueeze(0)

    # 2. Maximum Uniform Inflation
    p_max = np.ones((seq_length, 3), dtype=np.float32) * 0.00001
    for t in range(seq_length):
        p_max[t, :] = min(0.25 * (t / 15.0), 0.75)
    scenarios["2_Max_Uniform"] = torch.tensor(p_max, device=device).unsqueeze(0)

    # 3. Pure Bellow 1
    p_B1 = np.ones((seq_length, 3), dtype=np.float32) * 0.00001
    for t in range(seq_length):
        p_B1[t, 0] = min(0.25 * (t / 15.0), 0.75)
    scenarios["3_Pure_Bellow_1"] = torch.tensor(p_B1, device=device).unsqueeze(0)

    # 4. Pure Bellow 2
    p_B2 = np.ones((seq_length, 3), dtype=np.float32) * 0.00001
    for t in range(seq_length):
        p_B2[t, 1] = min(0.25 * (t / 15.0), 0.75)
    scenarios["4_Pure_Bellow_2"] = torch.tensor(p_B2, device=device).unsqueeze(0)

    # 5. Complex Twist: Bellow 2 into Bellow 3 (With fixed plateau math)
    p_B = np.ones((seq_length, 3), dtype=np.float32) * 1.0
    for t in range(min(25, seq_length)):
        p_B[t, 1] = 1.0 + (60000.0 * (t / 25.0))
    if seq_length > 25:
        p_B[25:, 1] = 1.0 + 60000.0 # Math fix applied here
    for t in range(20, min(45, seq_length)):
        p_B[t, 2] = 1.0 + (50000.0 * ((t - 20) / 25.0))
    if seq_length > 45:
        p_B[45:, 2] = 1.0 + 50000.0 # Math fix applied here
    scenarios["5_Complex_Twist"] = torch.tensor(np.clip(p_B / 100000.0, 0.00001, 1.0), device=device).unsqueeze(0)

    return scenarios

# ==========================================
# --- MAIN LOOP ---
# ==========================================
def main():
    # --- Configuration ---
    DATA_DIRS = [
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/125_cases",  
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/216_cases",  
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/Staircase_creep"     
    ]
    
    # UPDATE THIS TO YOUR NEW RESNET RUN ONCE IT FINISHES TRAINING
    CHECKPOINT_PATH = "runs/revert_2_curriculum_latentConsistency_globalActionAugment_MASK_2026-04-11_01-29-23/best_model.pth"
    
    run_folder_name = CHECKPOINT_PATH.split("/")[-2]
    OUTPUT_DIR = os.path.join("simulation_results", "generalization_tests", run_folder_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    IMAGE_MODE = "mask"
    FEATURE_DIM = 64
    SIMULATION_LENGTH = 30 
    FRAME_STRIDE = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing Simulator on {device}...")

    # --- Initialize Architecture ---
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=128).to(device)
    
    # --- Load Weights ---
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Cannot find checkpoint at {CHECKPOINT_PATH}")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics.load_state_dict(checkpoint['dynamics'])
    decoder.load_state_dict(checkpoint['decoder'])
    print(f"Weights loaded successfully from Epoch {checkpoint['epoch']}.")
    
    encoder.eval()
    dynamics.eval()
    decoder.eval()

    # --- Setup Base Dataset ---
    print("Initializing dataset to extract resting frames...")
    base_dataset = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, 
        image_mode=IMAGE_MODE, seq_len=None, frame_stride=FRAME_STRIDE
    )
    
    if 'val_indices' in checkpoint:
        val_dataset = Subset(base_dataset, checkpoint['val_indices'])
    else:
        val_dataset = base_dataset
    
    # Extract the absolute zero-state
    idle_frame = val_dataset[0]["video"][0].clone()
    resting_frames = idle_frame.unsqueeze(0).to(device) 
    
    H, W = resting_frames.shape[-2:]
    view_rays = {}
    for v in range(4):
        origins, dirs = get_full_image_rays(H, W, view_idx=v, device=device)
        view_rays[v] = {"origins": origins.unsqueeze(0), "dirs": dirs.unsqueeze(0)}

    test_suite = generate_test_scenarios(SIMULATION_LENGTH, device)
    
    # ==========================================
    # --- PRINT DIAGNOSTICS ---
    # ==========================================
    print_scenario_distance_report(test_suite)
    print_action_sensitivity_report(encoder, dynamics, resting_frames, device)
    print_action_gradient_report(encoder, dynamics, resting_frames, device)

    # ==========================================
    # --- RENDER THE TEST BATTERY ---
    # ==========================================
    for scenario_name, custom_pressures in test_suite.items():
        print(f"\n========================================")
        print(f"Running Scenario: {scenario_name}")
        print(f"========================================")
        
        # Save as MP4. Width is W * 4 because we stack 4 views horizontally
        video_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (W * 4, H))

        with torch.no_grad():
            current_tri_planes = encoder(resting_frames)
            hidden_state = None
            scenario_latent_delta = []
            
            for t in tqdm(range(SIMULATION_LENGTH), desc=f"Rendering {scenario_name}"):
                action_t = custom_pressures[:, t]
                
                planes_next_pred, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)
                scenario_latent_delta.append(_mean_abs_plane_diff(current_tri_planes, planes_next_pred))
                
                rendered_views = []
                for v in range(4):
                    rays_o = view_rays[v]["origins"]
                    rays_d = view_rays[v]["dirs"]
                    
                    rgb_pred = render_rays_chunked(
                        ray_marcher, decoder, planes_next_pred, rays_o, rays_d, chunk_size=4096
                    )
                    
                    frame_np = (rgb_pred.view(H, W).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
                    rendered_views.append(frame_bgr)
                
                # Stack all 4 views horizontally side-by-side
                final_frame = np.concatenate(rendered_views, axis=1)
                
                # Extract specific pressure values for the overlay
                p1, p2, p3 = action_t[0, 0].item(), action_t[0, 1].item(), action_t[0, 2].item()
                p_text = f"P1: {p1:.2f} | P2: {p2:.2f} | P3: {p3:.2f}"
                
                # Overlay SIMULATION label and live pressures in Red
                cv2.putText(final_frame, f"SIMULATION | {p_text}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                video_writer.write(final_frame)
                current_tri_planes = planes_next_pred

            if len(scenario_latent_delta) > 0:
                print(
                    f"[DIAGNOSTIC] {scenario_name}: "
                    f"avg mean |delta tri-plane| per step = {np.mean(scenario_latent_delta):.8f}"
                )

        video_writer.release()
        
        del current_tri_planes, hidden_state, planes_next_pred
        if str(device) == "mps":
            torch.mps.empty_cache()
        elif str(device) == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nAll tests complete! Check the MP4 videos in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()