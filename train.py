import os
# Use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc

# Import custom modules
from src.multiview_dataset import SoftRobotDataset
from src.encoder import TriPlaneEncoder
from src.temporal_dynamics import TriPlaneDynamics
from src.decoder import TriPlaneDecoder
from src.renderer import VolumetricRayMarcher, sample_orthographic_rays, get_full_image_rays


def dice_loss_per_batch(pred, target, smooth=1e-5):
    # Preserve the batch dimension [B], flatten the spatial/channel dimensions [-1]
    B = pred.shape[0]
    pred_flat = pred.contiguous().view(B, -1)
    target_flat = target.contiguous().view(B, -1)
    
    # Sum across the spatial dimensions (dim=1), maintaining shape [B]
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # Dice formula calculates a distinct coefficient for each item in the batch
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    
    return 1.0 - dice_coeff # Returns a tensor of shape [B]

def main():
    
    # 1. Configuration

    # Pass all your separate data directories as a list
    DATA_DIRS = [
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/125_cases",  # 25k step size
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/216_cases",  # 20k step size
        r"/Users/alp/SoftRobot_Dataset_Hysteresis/Staircase_creep"      # The 60-second creep data
    ]
    IMAGE_MODE = "mask"

    # Initialize TensorBoard Writer and Log Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/actionConcat_visualDropout_{IMAGE_MODE.upper()}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")
    print(f"Checkpoints will be saved to: {log_dir}")
      
    RESUME_CHECKPOINT_PATH = '' # If left empty, training starts from scratch.
    
    BATCH_SIZE = 2 # or 4 if GPU memory allows
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000

    FRAME_STRIDE = 2 # Skip every other frame to force learning of dynamics, not just memorization.
    SEQUENCE_LENGTH = 24
    FEATURE_DIM = 128
    RAYS_PER_STEP = 512 # Number of rays to sample per time step for loss calculation
    TF_UNTIL = 200 # Epoch until which teacher forcing is used

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

    # 2. Initialize Dataset
    train_base = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, 
        seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE
    )
    
    # Validation Base: seq_len=None. Returns the full original sequences
    val_base = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, 
        seq_len=None, frame_stride=FRAME_STRIDE
    )
    
    # ==========================================
    # --- Validation Split ---
    # ==========================================
    # Randomly select 15% of the pure bending cases for validation
    all_bending_indices = []
    special_indices = []
    
    for idx, folder_path in enumerate(train_base.case_folders):
        folder_name = os.path.basename(folder_path)
        if folder_name.startswith("Case_"):
            all_bending_indices.append(idx)
        else:
            # Captures Staircase, PE, or any other non-standard folders
            special_indices.append(idx)
            
    VAL_PERCENTAGE = 0.15
    
    # Set seed for reproducible splits across runs.
    random.seed(42)
    
    num_val_cases = int(len(all_bending_indices) * VAL_PERCENTAGE)
    
    # Randomly pick indices from ONLY the pure bending cases
    val_indices = random.sample(all_bending_indices, num_val_cases)
    print(f"Validation Cases: {len(val_indices)} | Validation Indices: {sorted(val_indices)}")
    
    # Training indices: All un-selected bending cases PLUS all special cases (Creep/PE)
    train_indices = [i for i in all_bending_indices if i not in val_indices] + special_indices
    
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)
    print(f"Data Split -> Training Cases: {len(train_dataset)} | Validation Cases: {len(val_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model Components
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=16).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    # 4. Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=1e-6) # AdamW decouples weight decay from grad update
    
    # Cosine Annealing with Warm Restarts: T_0 is first cycle length, T_mult multiplies the cycle length after restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # --- RESUME CHECKPOINT ---
    best_val_loss = float('inf')
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"=================================================")
        print(f"RESUMING TRAINING FROM: {RESUME_CHECKPOINT_PATH}")
        print(f"=================================================")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
        
        # 1. Load Model Weights
        encoder.load_state_dict(checkpoint['encoder'])
        dynamics.load_state_dict(checkpoint['dynamics'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # 2. Load Optimizer and Scheduler States (Industry Standard)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 3. Re-inject the manually decayed base learning rates!
        if 'scheduler_base_lrs' in checkpoint:
            scheduler.base_lrs = checkpoint['scheduler_base_lrs']
            
        start_epoch = checkpoint['epoch'] 
        best_val_loss = checkpoint['best_val_loss']

    # Define Loss Functions
    # Initialize with reduction='none' so it outputs per-pixel losses
    bce_loss_fn = nn.BCELoss(reduction='none') # Use BCE instead of L1 for sharper edges

    # 5. The Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        # Calculate Teacher Forcing Ratio: Decays to 0.0 by epoch TF_UNTIL, forcing pure autoregression
        tf_ratio = max(0.0, 1.0 - (epoch / TF_UNTIL))
        
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
                # Action jittering: Adds small random noise to the pressures to prevent overfitting to exact values 
                # and encourage learning of smooth dynamics. 
                base_action = pressures[:, t]
                # Add random noise between -0.025 and +0.025 (2.5% pressure fluctuation)
                action_noise = (torch.rand_like(base_action) - 0.5) * 0.05
                # Clamp the jittered action so we never feed the model impossible physics (like negative pressure)
                action_t = torch.clamp(base_action + action_noise, min=0.00001, max=1.0)
                                
                frames_next_true = videos[:, t+1] 
                
                # Predict the next 3D state
                # ==========================================
                # --- VISUAL DROPOUT ---
                # ==========================================
                # 15% of the time, we zero out the visual features.
                # This breaks the "video retrieval" cheat code and forces the physics 
                # engine to rely solely on the pressure inputs to calculate the next state.
                if random.random() < 0.15:
                    dropped_planes = {k: torch.zeros_like(v) for k, v in current_tri_planes.items()}
                    planes_next_pred, hidden_state = dynamics(dropped_planes, action_t, hidden_state)
                else:
                    planes_next_pred, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)
                
                # Render the current frame
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(
                    frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                
                # 1. Combined BCE and Dice Loss (Shape: [B])
                raw_bce = bce_loss_fn(rgb_pred, target_rgb)
                # Average the BCE loss across spatial dimensions, keeping the batch dimension
                loss_bce = raw_bce.view(B, -1).mean(dim=1) 
                
                loss_dice = dice_loss_per_batch(rgb_pred, target_rgb)
                
                # 2. 3D Sparsity Loss (Shape: [B])
                random_points_3d = (torch.rand(B, 1024, 3, device=device) * 2.0) - 1.0
                random_probs, _ = decoder(planes_next_pred, random_points_3d)
                
                eps = 1e-6
                entropy = -random_probs * torch.log(random_probs + eps) -\
                    (1.0 - random_probs) * torch.log(1.0 - random_probs + eps)
                
                # Average entropy per batch item
                sparsity_loss = entropy.view(B, -1).mean(dim=1) 
                lambda_sparse = 0.005
                
                # ==========================================
                # --- PER-SAMPLE ACTION WEIGHTING ---
                # ==========================================
                # Calculate the max pressure for each specific video in the batch. Shape: [B]
                pressure_magnitude = torch.max(action_t, dim=1)[0]
                
                # Each batch item gets its own exact multiplier. Shape: [B]
                loss_multiplier = 1.0 + (3.0 * pressure_magnitude)
                
                # Sum the isolated losses, apply the specific multiplier, then reduce to a scalar
                base_step_loss = loss_bce + loss_dice + (lambda_sparse * sparsity_loss)
                weighted_step_loss = base_step_loss * loss_multiplier 
                
                # Collapse to a single scalar for .backward()
                step_loss = weighted_step_loss.mean()
                
                batch_sequence_loss += step_loss
                
                # === SCHEDULED SAMPLING ===
                # Roll a virtual die to decide if we help the AI or force it to rely on itself
                if random.random() < tf_ratio:
                    # Teacher Forcing: Feed it the perfect ground truth for the next step
                    current_tri_planes = encoder(frames_next_true)
                else:
                    # Autoregressive: Force it to use its own noisy prediction
                    current_tri_planes = {k: v for k, v in planes_next_pred.items()}

            batch_sequence_loss = batch_sequence_loss / (Time - 1)
            batch_sequence_loss.backward()
            
            # Gradient clipping prevents the network from blowing up during pure autoregression
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
            for val_batch_idx, batch in enumerate(val_loader):
                vids_val = batch["video"].to(device)
                press_val = batch["pressures"].to(device)
                B_val, V_Time, Views, C, H, W = vids_val.shape
                
                # Validation is ALWAYS 100% Autoregressive (No Teacher Forcing)
                curr_planes = encoder(vids_val[:, 0])
                h_val = None
                
                for t in range(V_Time - 1):
                    # Apply the same clamping to validation pressures
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    # Feed the clamped action into the dynamics engine
                    pred_planes, h_val = dynamics(curr_planes, action_val_clamped, h_val)
                    
                    # Compute fast L1 loss on rays to track generalization
                    ray_o, ray_d, target = sample_orthographic_rays(
                        vids_val[:, t+1], num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                    rgb_p = ray_marcher.render_rays(decoder, pred_planes, ray_o, ray_d)
                    
                    # === Hybrid BCE+Dice Loss for Validation ===
                    # 1. Calculate per-batch BCE (matching training logic)
                    raw_bce_val = bce_loss_fn(rgb_p, target)
                    loss_bce_val = raw_bce_val.view(B_val, -1).mean(dim=1)
                    # 2. Calculate per-batch Dice
                    loss_dice_val = dice_loss_per_batch(rgb_p, target)
                    
                    val_step_loss = (loss_bce_val + loss_dice_val).mean()
                    val_loss += val_step_loss.item()
                    
                    # ---------------------------------------------------------
                    # VALIDATION VISUALIZATION BLOCK (100% Autoregressive)
                    # ---------------------------------------------------------
                    # Log every 10 epochs, ONLY on the first validation batch
                    if (epoch + 1) % 10 == 0 and val_batch_idx == 0 and (t == (V_Time // 2) or t == (V_Time - 2)):
                        
                        stage_name = "Val_Middle" if t == (V_Time // 2) else "Val_Last"
                        
                        for v in range(Views):
                            real_frame = vids_val[0, t+1, v].detach().cpu()
                            full_ray_origins, full_ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                            full_ray_origins = full_ray_origins.unsqueeze(0)
                            full_ray_dirs = full_ray_dirs.unsqueeze(0)
                            
                            # Render the full image for the first video in the batch [0:1]
                            single_plane_pred = {key: pred_planes[key][0:1] for key in pred_planes} 
                            full_rgb_pred = ray_marcher.render_rays(
                                decoder, single_plane_pred, full_ray_origins, full_ray_dirs)
                            
                            pred_frame = full_rgb_pred.view(H, W, 3).permute(2, 0, 1).detach().cpu()
                            
                            comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                            # Tag clearly as Validation in TensorBoard
                            writer.add_image(f'Validation_Autoregressive_{stage_name}/Side_{v+1}', comparison_grid, epoch + 1)
                    
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
        # Track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # ==========================================
        # --- CHECKPOINT SAVING ---
        # ==========================================
        checkpoint_dict = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_indices': train_dataset.indices, 
            'val_indices': val_dataset.indices,     
            'encoder': encoder.state_dict(),
            'dynamics': dynamics.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scheduler_base_lrs': scheduler.base_lrs # Preserves LR decay
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