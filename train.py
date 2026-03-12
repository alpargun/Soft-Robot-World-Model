import os
# Tell PyTorch to use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from encoder_resnet_gn import ResNetGNTriPlaneEncoder
from encoder_mini import MiniResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from orthographic_ray_generator import sample_orthographic_rays
from visualization_helper import get_full_image_rays


def dice_loss(pred, target, smooth=1e-5):
    # Flatten tensors to calculate spatial overlap
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    # Dice formula: 2 * Intersection / (Pred_Area + Target_Area)
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice_coeff

def main():
    # 1. Configuration
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    IMAGE_MODE = "mask" # Change to "rgb" to automatically enable LPIPS perceptual loss!
    
    BATCH_SIZE = 2 # or 4 if GPU memory allows
    FEATURE_DIM = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000
    RAYS_PER_STEP = 1600 # Number of rays to sample per time step for loss calculation
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Training on GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Training on Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Training on CPU.")

    print(f"Initializing World Model Training on: {device} | Mode: {IMAGE_MODE.upper()}")

    # Initialize TensorBoard Writer and Log Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/resnetGN_decoderConcat_125cases_{IMAGE_MODE.upper()}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")
    print(f"Checkpoints will be saved to: {log_dir}")

    # 2. Initialize Dataset
    dataset = SoftRobotDataset(DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    
    # ===========================================================================================================
    # FULL DATASET ENABLED: We use the full dataset to properly learn the hysteresis curve across all cases
    
    # --- AUTOMATIC 10% VALIDATION SPLIT ---
    val_size = int(len(dataset) * 0.10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch 1 for clean, sequential validation evaluation
    
    # OVERFIT TEST: Commented out for production run.
    # overfit_dataset = Subset(dataset, indices=[-1, -2]) 
    # dataloader = DataLoader(overfit_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # ===========================================================================================================

    # 3. Initialize Model Components
    #encoder = ResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    #encoder = MiniResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    encoder = ResNetGNTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    # 4. Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    
    # Added weight decay to prevent FiLM Shift (beta) parameters from growing too large and ignoring inputs
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Cosine Annealing with Warm Restarts: T_0 is the first cycle length, T_mult multiplies the cycle length after each restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    # - Epochs 0-49: LR decays from 1e-4 to 1e-6: Gently learn basic 3D geometry and dynamics with Teacher Forcing still active
    # - Epoch 50: LR jumps back to 1e-4: This "warm restart" jolts the ConvGRU out of any lazy memorization habits right as the Teacher Forcing ratio drops below 0.50.
    # - Epochs 50-149: LR decays from 1e-4 to 1e-6
    # - Epochs 150-349: LR decays from 1e-4 to 1e-6
    # - Epochs 350-749: LR decays from 1e-4 to 1e-6
    # This cycle allows the model to escape local minima and encourages better convergence, especially in the later stages of training
    
    # Define Loss Functions
    # L1 is kept ONLY for validation tracking (it is a good human-readable error metric)
    l1_loss_fn = nn.L1Loss() # L1 is highly forgiving of the fuzzy smearing so we see robot expanding like a balloon. Switched to BCE + Dice for sharper edges and better spatial overlap.
    # # Use BCE for hard edges, Dice for spatial overlap
    bce_loss_fn = nn.BCELoss()
    
    if IMAGE_MODE == "rgb":
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) 

    # Track the best validation loss
    best_val_loss = float('inf')

    # 5. The Training Loop
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        # --- FASTER TF DECAY ---
        # Calculate Teacher Forcing Ratio: Decays to 0.0 by epoch 200 to force autoregressive learning sooner
        tf_ratio = max(0.0, 1.0 - (epoch / 200.0))
        
        # Wraps the dataloader to show a progress bar for the current epoch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            videos = batch["video"].to(device)       
            pressures = batch["pressures"].to(device) # Pressures are pre-normalized by Dataset!
            
            B, Time, Views, C, H, W = videos.shape 
            
            optimizer.zero_grad()
            batch_sequence_loss = 0.0
            
            hidden_state = None 
            
            # Establish the initial state
            current_tri_planes = encoder(videos[:, 0])
            
            for t in range(Time - 1):
                # Clamp the pressure to ensure a strict floor of 1.0 and a physical ceiling of 100000.0
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                frames_next_true = videos[:, t+1] 
                
                # Predict the next 3D state
                planes_next_pred, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)
                
                # Render the current frame
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                
                # 1. Combined BCE and Dice Loss for expanding geometric masks
                loss_bce = bce_loss_fn(rgb_pred, target_rgb)
                loss_dice = dice_loss(rgb_pred, target_rgb)
                
                # 2. 3D Sparsity Loss (Entropy-based for crisp boundaries)
                # Sample 1024 random 3D coordinates in the bounding box [-1, 1]^3
                random_points_3d = (torch.rand(B, 1024, 3, device=device) * 2.0) - 1.0
                random_probs, _ = decoder(planes_next_pred, random_points_3d)
                
                # Calculate binary entropy. Epsilon (1e-6) prevents log(0) NaN explosions.
                eps = 1e-6
                entropy = -random_probs * torch.log(random_probs + eps) - (1.0 - random_probs) * torch.log(1.0 - random_probs + eps)
                sparsity_loss = torch.mean(entropy)
                
                # Slightly reduced lambda since maximum entropy is ~0.69 (ln 2), which scales differently than uniform mean
                lambda_sparse = 0.005
                
                # Weight them equally, adding the sparsity penalty
                step_loss = loss_bce + loss_dice + (lambda_sparse * sparsity_loss)
                
                # 3. Perceptual LPIPS Loss (Requires 2D image patches)
                if IMAGE_MODE == "rgb":
                    # Calculate the dimension of the square patch (e.g., 400 rays = 20x20 patch)
                    rays_per_view = RAYS_PER_STEP // Views
                    patch_dim = int(rays_per_view ** 0.5)
                    
                    # Reshape from [Batch, 1600, 3] -> [Batch * 4 Views, 3 Channels, 20 Height, 20 Width]
                    # This creates 4 separate mini-images for LPIPS to analyze
                    rgb_pred_patch = rgb_pred.view(B * Views, patch_dim, patch_dim, 3).permute(0, 3, 1, 2)
                    target_rgb_patch = target_rgb.view(B * Views, patch_dim, patch_dim, 3).permute(0, 3, 1, 2)
                    
                    # LPIPS expects images in range [-1, 1], our RGB is currently [0, 1]
                    pred_lpips = rgb_pred_patch * 2.0 - 1.0
                    target_lpips = target_rgb_patch * 2.0 - 1.0
                    
                    # Add the perceptual loss to the L1 loss (usually weighted slightly less)
                    lpips_val = lpips_loss_fn(pred_lpips, target_lpips).mean()
                    step_loss += (0.1 * lpips_val)

                batch_sequence_loss += step_loss
                
                # === SCHEDULED SAMPLING LOGIC ===
                # Roll a virtual die to decide if we help the AI or force it to rely on itself
                if random.random() < tf_ratio:
                    # Teacher Forcing: Feed it the perfect ground truth for the next step
                    current_tri_planes = encoder(frames_next_true)
                else:
                    # Autoregressive: Force it to use its own noisy prediction
                    current_tri_planes = {k: v.detach() for k, v in planes_next_pred.items()}
                
                # ---------------------------------------------------------
                # VISUALIZATION BLOCK (All 4 Views -> TensorBoard)
                # ---------------------------------------------------------
                # Log both the middle frame (peak motion) and the last frame (max compounding error/hallucination)
                if (epoch + 1) % 10 == 0 and batch_idx == 0 and (t == (Time // 2) or t == (Time - 2)):
                    
                    # Group them cleanly in TensorBoard
                    stage_name = "Middle" if t == (Time // 2) else "Last"
                    
                    with torch.no_grad():
                        for v in range(Views):
                            real_frame = frames_next_true[0, v].detach().cpu()
                            full_ray_origins, full_ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                            full_ray_origins = full_ray_origins.unsqueeze(0)
                            full_ray_dirs = full_ray_dirs.unsqueeze(0)
                            
                            single_plane_pred = {key: planes_next_pred[key][0:1] for key in planes_next_pred} 
                            full_rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, full_ray_origins, full_ray_dirs)
                            pred_frame = full_rgb_pred.view(H, W, 3).permute(2, 0, 1).detach().cpu()
                            
                            comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                            writer.add_image(f'Comparison_{stage_name}/Side_{v+1}', comparison_grid, epoch + 1)

            batch_sequence_loss = batch_sequence_loss / (Time - 1)
            batch_sequence_loss.backward()
            
            # CRITICAL: Gradient clipping prevents FiLM from blowing up during pure autoregression
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_sequence_loss.item()
            
        # Step the learning rate down appropriately per epoch
        scheduler.step()

        # --- DECAYING PEAK LR LOGIC ---
        # The T_0=50, T_mult=2 schedule restarts at epochs 50, 150, and 350.
        # We cut the maximum learning rate in half at each restart.
        restart_epochs = [50, 150, 350]
        if (epoch + 1) in restart_epochs:
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] *= 0.5
            scheduler.base_lrs = [base_lr * 0.5 for base_lr in scheduler.base_lrs]
            print(f">>> LR Warm Restart: Peak decayed to {scheduler.base_lrs[0]:.6f}")
            
        avg_loss = epoch_loss / len(dataloader)
        
        # ==========================================
        # --- VALIDATION PHASE ---
        # ==========================================
        encoder.eval()
        dynamics.eval()
        decoder.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                vids_val = batch["video"].to(device)
                press_val = batch["pressures"].to(device)
                _, V_Time, _, _, _, _ = vids_val.shape
                
                # Validation is ALWAYS 100% Autoregressive (No Teacher Forcing) to test true physics learning
                curr_planes = encoder(vids_val[:, 0])
                h_val = None
                
                for t in range(V_Time - 1):
                    # ADD THE EXACT SAME CLAMP TO VALIDATION
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    # Feed the clamped action into the dynamics engine
                    pred_planes, h_val = dynamics(curr_planes, action_val_clamped, h_val)
                    
                    # Compute fast L1 loss on rays to track generalization
                    ray_o, ray_d, target = sample_orthographic_rays(vids_val[:, t+1], num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                    rgb_p = ray_marcher.render_rays(decoder, pred_planes, ray_o, ray_d)
                    
                    # === USE THE HYBRID BCE+DICE LOSS FOR VALIDATION ===
                    loss_bce_val = bce_loss_fn(rgb_p, target)
                    loss_dice_val = dice_loss(rgb_p, target)
                    val_loss += (loss_bce_val + loss_dice_val).item()
                    
                    # Strictly feed prediction back into the engine
                    curr_planes = pred_planes
                    
        avg_val_loss = val_loss / (len(val_loader) * (V_Time - 1))
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | TF Ratio: {tf_ratio:.2f} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/Validation_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Training/TF_Ratio', tf_ratio, epoch + 1)
        # Optional: track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # ==========================================
        # --- CHECKPOINT SAVING ---
        # ==========================================
        # Save the best overall model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'dynamics': dynamics.state_dict(),
                'decoder': decoder.state_dict(),
            }, os.path.join(log_dir, "best_model.pth"))
            print(f"*** New Best Model Saved (Val Loss: {best_val_loss:.6f}) ***")

        # Save milestone checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'dynamics': dynamics.state_dict(),
                'decoder': decoder.state_dict(),
            }, os.path.join(log_dir, f"world_model_checkpoint_epoch_{epoch+1}.pth"))
            
        # ALWAYS save the latest state so progress is never lost during sudden stops
        torch.save({
            'encoder': encoder.state_dict(),
            'dynamics': dynamics.state_dict(),
            'decoder': decoder.state_dict(),
        }, os.path.join(log_dir, "last_checkpoint.pth"))

    writer.close()

if __name__ == "__main__":
    main()