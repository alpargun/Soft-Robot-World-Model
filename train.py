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
import gc # NEW: Added for aggressive memory cleanup

# Import custom modules
from multiview_dataset import SoftRobotDataset
from encoder_resnet_gn import ResNetGNTriPlaneEncoder
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
    
    # --- NEW: RESUME CAPABILITY ---
    # Put the exact path to the folder that just crashed so we can pick up where we left off
    RESUME_CHECKPOINT_PATH = ''#"runs/resnetGN_decoderConcat_125cases_clampfix_MASK_2026-03-12_01-10-53/last_checkpoint.pth"
    START_EPOCH = 0
    
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
    log_dir = f"runs/mixedDataset_{IMAGE_MODE.upper()}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")
    print(f"Checkpoints will be saved to: {log_dir}")

    # 2. Initialize Dataset
    dataset = SoftRobotDataset(DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    
    # Explicitly define Validation vs Training to protect the long videos
    total_cases = len(dataset)
    
    # Manually pick 12 specific indices from the original 2-second videos for validation set.
    val_indices = [
        79, 11, 124, 89, 7, 34, 61, 16, 24, 97, 109, 114, # original val set
        5, 14, 23, 37, 48, 52, 66, 78, 81, 95, 101, 118, 122 # 13 additional 2-second cases
    ]
    
    # Everything else including the 10s and 60s cases goes into Training
    train_indices = [i for i in range(total_cases) if i not in val_indices]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Data Split -> Training Cases: {len(train_dataset)} | Validation Cases: {len(val_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model Components
    encoder = ResNetGNTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    # 4. Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    
    # Added weight decay to prevent FiLM Shift (beta) parameters from growing too large and ignoring inputs
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Cosine Annealing with Warm Restarts: T_0 is first cycle length, T_mult multiplies the cycle length after restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # --- NEW: RESUME LOGIC ---
    best_val_loss = float('inf')
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"=================================================")
        print(f"RESUMING TRAINING FROM: {RESUME_CHECKPOINT_PATH}")
        print(f"=================================================")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # If your checkpoint has the optimizer/epoch state, load it. If not, we just start at Epoch 203 manually.
        if 'epoch' in checkpoint:
            START_EPOCH = checkpoint['epoch']
        else:
            # We know it crashed right after finishing Epoch 203, so we start at 203
            START_EPOCH = 203 
            
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        else:
            best_val_loss = 0.061356 # Injecting the known best score
            
        # Fast-forward the learning rate scheduler to the correct epoch
        for _ in range(START_EPOCH):
            scheduler.step()

    # Define Loss Functions
    bce_loss_fn = nn.BCELoss() # Use BCE instead of L1 for sharper edges, Dice for better spatial overlap

    if IMAGE_MODE == "rgb":
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) 

    # 5. The Training Loop
    for epoch in range(START_EPOCH, NUM_EPOCHS):
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
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(
                    frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
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
                entropy = -random_probs * torch.log(random_probs + eps) -\
                    (1.0 - random_probs) * torch.log(1.0 - random_probs + eps)
                sparsity_loss = torch.mean(entropy)
                
                # Slightly reduced lambda as max entropy is ~0.69 (ln 2), which scales differently than uniform mean
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
                            full_rgb_pred = ray_marcher.render_rays(
                                decoder, single_plane_pred, full_ray_origins, full_ray_dirs)
                            pred_frame = full_rgb_pred.view(H, W, 3).permute(2, 0, 1).detach().cpu()
                            
                            comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                            writer.add_image(f'Comparison_{stage_name}/Side_{v+1}', comparison_grid, epoch + 1)

            batch_sequence_loss = batch_sequence_loss / (Time - 1)
            batch_sequence_loss.backward()
            
            # CRITICAL: Gradient clipping prevents FiLM from blowing up during pure autoregression
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_sequence_loss.item()
            
            # --- AGGRESSIVE MEMORY CLEANUP FOR MACS ---
            # Deleting these heavy tensors and explicitly clearing the cache prevents fragmentation crashes
            del videos, pressures, ray_origins, ray_dirs, target_rgb, rgb_pred, planes_next_pred, current_tri_planes, batch_sequence_loss, step_loss
            if str(device) == "mps":
                torch.mps.empty_cache()
            elif str(device) == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        # Step the learning rate down appropriately per epoch
        scheduler.step()

        # --- DECAYING PEAK LR LOGIC ---
        restart_epochs = [50, 150, 350] # The T_0=50, T_mult=2 schedule restarts at epochs 50, 150, and 350.
        if (epoch + 1) in restart_epochs:
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] *= 0.5 # Cut the maximum learning rate in half at each restart.
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
                    ray_o, ray_d, target = sample_orthographic_rays(
                        vids_val[:, t+1], num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                    rgb_p = ray_marcher.render_rays(decoder, pred_planes, ray_o, ray_d)
                    
                    # === USE THE HYBRID BCE+DICE LOSS FOR VALIDATION ===
                    loss_bce_val = bce_loss_fn(rgb_p, target)
                    loss_dice_val = dice_loss(rgb_p, target)
                    val_loss += (loss_bce_val + loss_dice_val).item()
                    
                    # Strictly feed prediction back into the engine
                    curr_planes = pred_planes
                    
                # Clean up Validation Memory too
                del vids_val, press_val, curr_planes, h_val
                if str(device) == "mps":
                    torch.mps.empty_cache()
                elif str(device) == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                    
        avg_val_loss = val_loss / (len(val_loader) * (V_Time - 1))
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | TF Ratio: {tf_ratio:.2f} |\
            Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/Validation_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Training/TF_Ratio', tf_ratio, epoch + 1)
        # Optional: track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # ==========================================
        # --- CHECKPOINT SAVING ---
        # ==========================================
        checkpoint_dict = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_indices': train_dataset.indices, # <--- SAVES EXACT TRAINING CASES
            'val_indices': val_dataset.indices,     # <--- SAVES EXACT VALIDATION CASES
            'encoder': encoder.state_dict(),
            'dynamics': dynamics.state_dict(),
            'decoder': decoder.state_dict(),
        }
        
        # Save the best overall model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_dict, os.path.join(log_dir, "best_model.pth"))
            print(f"*** New Best Model Saved (Val Loss: {best_val_loss:.6f}) ***")

        # Save milestone checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_dict, os.path.join(log_dir, f"world_model_checkpoint_epoch_{epoch+1}.pth"))
            
        # ALWAYS save the latest state so progress is never lost during sudden stops
        torch.save(checkpoint_dict, os.path.join(log_dir, "last_checkpoint.pth"))

    writer.close()

if __name__ == "__main__":
    main()