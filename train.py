import os
# Use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
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
from src.renderer import VolumetricRayMarcher, sample_orthographic_rays, get_full_image_rays, render_rays_chunked

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

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) for binary masks.
    Ignores the background and strictly grades the silhouette shape.
    """
    # Convert soft predictions and targets to hard binary masks
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    
    # Calculate intersection and union over the spatial/channel dimensions
    intersection = (pred_bin * target_bin).sum(dim=[1, 2]) 
    union = pred_bin.sum(dim=[1, 2]) + target_bin.sum(dim=[1, 2]) - intersection
    
    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean() # Return the batch average

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
    log_dir = f"runs/5_lossCoefs_renderingFix_splitRNNs_latentConsist_spatialFiLM_residualPred_{IMAGE_MODE.upper()}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")
    print(f"Checkpoints will be saved to: {log_dir}")
      
    RESUME_CHECKPOINT_PATH = '' # If left empty, training starts from scratch.
    
    BATCH_SIZE = 4 # or 4 if GPU memory allows
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000

    FRAME_STRIDE = 2 # Skip every other frame to force learning of dynamics, not just memorization.
    SEQUENCE_LENGTH = 24
    FEATURE_DIM = 64
    RAYS_PER_STEP = 256 # Number of rays to sample per time step for loss calculation
    
    BURN_IN_LENGTH = 5 
    TF_UNTIL = 100.0 # Number of epochs to apply teacher forcing. After this, it decays to 0.
    VAL_PERCENTAGE = 0.15 # Percentage of pure bending cases to hold out for validation

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
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty! Check your DATA_DIRS and split logic.")

    # 3. Initialize Model Components
    encoder = TriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    # 4. Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=1e-4) # AdamW decouples weight decay from grad update
    # Cosine Annealing with Warm Restarts: T_0 is first cycle length, T_mult multiplies the cycle length after restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1, eta_min=1e-6)
    
    # --- RESUME CHECKPOINT ---
    best_val_loss = float('inf')
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print("=================================================")
        print(f"RESUMING TRAINING FROM: {RESUME_CHECKPOINT_PATH}")
        print("=================================================")
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
    l1_loss_fn = nn.L1Loss() # for Latent Consistency

    # 5. The Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):

        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0

        # Start with 100% Teacher Forcing, decaying smoothly to 0% by Epoch 150
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / TF_UNTIL))
        writer.add_scalar('Training/Teacher_Forcing_Ratio', teacher_forcing_ratio, epoch)
        
        # Wraps the dataloader to show a progress bar for the current epoch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            videos = batch["video"].to(device)       
            pressures = batch["pressures"].to(device) # Pressures are pre-normalized by Dataset!
            B, Time, Views, C, H, W = videos.shape 
            
            optimizer.zero_grad()
            hidden_state = None 
            current_tri_planes = encoder(videos[:, 0]) # Establish the initial visual state

            # Trackers for the Autoregressive Phase
            batch_sequence_loss = 0.0
            autoregressive_steps = 0
            
            # Trajectory-Level Action Augmentation
            # Generates a single, smooth offset applied to the entire pressure sequence to simulate 
            # realistic sensor miscalibration/hysteresis drift, replacing the unrealistic frame-by-frame jitter.
            sequence_action_noise = (torch.rand(B, 3, device=device) - 0.5) * 0.05
            
            # ==========================================
            # --- CONDITIONAL BURN-IN SELECTION ---
            # ==========================================
            # 1. Check if the sequence starts from a true physical resting state.
            is_resting_start = torch.max(pressures[:, 0]).item() < 0.01
            
            # 2. Only allow Cold Starts if the robot is actually at rest.
            if is_resting_start and random.random() < 0.50:
                current_burn_in = 1
            else:
                current_burn_in = BURN_IN_LENGTH
            
            # ==========================================
            # --- PHASE 1: BURN-IN ---
            # ==========================================
            for t in range(current_burn_in - 1):
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                
                # Step the physics engine to build memory
                _, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)
                
                # Force the visual state to reality (Teacher Forcing) for the next step
                current_tri_planes = encoder(videos[:, t+1])
                
            # ==========================================
            # --- PHASE 2: AUTOREGRESSION ---
            # ==========================================
            for t in range(current_burn_in - 1, Time - 1):
                
                # Apply the trajectory-level augmentation globally to the step
                base_action = pressures[:, t]
                action_t = torch.clamp(base_action + sequence_action_noise, min=0.00001, max=1.0)
                frames_next_true = videos[:, t+1] 
                
                # Predict the next latent state
                planes_next_pred, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)

                # --- LATENT CONSISTENCY ---
                # Encode the actual next frame to see where the dynamics "should" have landed
                with torch.no_grad(): # we don't want to accidentally train the encoder to cheat
                    planes_next_real = encoder(frames_next_true)

                latent_loss = 0.0
                for pk in ['xy', 'xz', 'yz']:
                    latent_loss += l1_loss_fn(planes_next_pred[pk], planes_next_real[pk])
                
                # Render the current frame
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(
                    frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                
                # 1. Combined BCE and Dice Loss (Shape: [B])
                raw_bce = bce_loss_fn(rgb_pred, target_rgb)
                loss_bce = raw_bce.view(B, -1).mean(dim=1) 

                loss_dice = dice_loss_per_batch(rgb_pred, target_rgb)
                
                # 2. 3D Sparsity Loss (Shape: [B])
                random_points_3d = (torch.rand(B, 1024, 3, device=device) * 2.0) - 1.0
                random_probs, _ = decoder(planes_next_pred, random_points_3d)
                eps = 1e-6
                entropy = -random_probs * torch.log(random_probs + eps) - (1.0 - random_probs) * torch.log(1.0 - random_probs + eps)
                sparsity_loss = entropy.view(B, -1).mean(dim=1) # Average entropy per batch item
                
                # ==========================================
                # --- CALCULATE LOSS ---
                # ==========================================
                lambda_latent = 5.0 # Weight for latent consistency
                lambda_sparsity = 0.05 # Weight for sparsity regularization (tuned to prevent over-sparsification)
                lambda_dice = 2.0 # Weight for Dice loss to ensure shape accuracy
                step_loss = (loss_bce + (lambda_dice * loss_dice) + (lambda_sparsity * sparsity_loss) + (lambda_latent * latent_loss)).mean()
                
                batch_sequence_loss += step_loss
                autoregressive_steps += 1
                
                # --- SCHEDULED SAMPLING ---
                if random.random() < teacher_forcing_ratio:
                    # Provide the real previous frame to prevent early gradient panic
                    current_tri_planes = encoder(frames_next_true)
                else:
                    # Strict Autoregression: Feed our own prediction back
                    current_tri_planes = {k: v for k, v in planes_next_pred.items()}

            # We only average the loss over the steps we actually predicted
            if autoregressive_steps > 0:
                batch_sequence_loss = batch_sequence_loss / autoregressive_steps
                batch_sequence_loss.backward()
                
                # Gradient clipping prevents the network from blowing up during pure autoregression
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_sequence_loss.item()

                # delete autoregressive variables only if they were created
                del ray_origins, ray_dirs, target_rgb, rgb_pred, planes_next_pred, step_loss
            
            # --- MEMORY CLEANUP ---
            # These variables are guaranteed to exist regardless of sequence length
            del videos, pressures, current_tri_planes, batch_sequence_loss
            
            if str(device) == "mps":
                torch.mps.empty_cache()
            elif str(device) == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        # Step the learning rate down appropriately per epoch
        scheduler.step()

        # --- DECAYING PEAK LR LOGIC ---
        if (epoch + 1) % 150 == 0: # Every 150 epochs, apply a warm restart with a decayed peak learning rate
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] *= 0.5 # Cut the maximum learning rate in half at each restart.
            scheduler.base_lrs = [base_lr * 0.5 for base_lr in scheduler.base_lrs]
            print(f">>> LR Warm Restart: Peak decayed to {scheduler.base_lrs[0]:.6f}")
            
        avg_loss = epoch_loss / max(1, len(dataloader)) 
        
        # ==========================================
        # --- VALIDATION PHASE ---
        # ==========================================
        encoder.eval()
        dynamics.eval()
        decoder.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_cold_start_iou = 0.0 # Tracks absolute Open-Loop performance
        val_autoregressive_steps = 0
        val_cold_start_steps = 0 # Tracks steps for the cold start evaluation
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                vids_val = batch["video"].to(device)
                press_val = batch["pressures"].to(device)
                B_val, V_Time, Views, C, H, W = vids_val.shape
                
                curr_planes = encoder(vids_val[:, 0])
                h_val = None

                # --- VAL PHASE 1: BURN-IN ---
                # Kept strictly at BURN_IN_LENGTH so validation metrics remain 
                # completely stable and comparable epoch-to-epoch.
                for t in range(BURN_IN_LENGTH - 1):
                    action_val = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    _, h_val = dynamics(curr_planes, action_val, h_val)
                    curr_planes = encoder(vids_val[:, t+1])
                
                # --- VAL PHASE 2: AUTOREGRESSION ---
                for t in range(BURN_IN_LENGTH - 1, V_Time - 1):
                    # Apply the same clamping to validation pressures
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    # Feed the clamped action into the dynamics engine
                    pred_planes, h_val = dynamics(curr_planes, action_val_clamped, h_val)
                    
                    # Compute fast L1 loss on rays to track generalization
                    ray_o, ray_d, target = sample_orthographic_rays(
                        vids_val[:, t+1], num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                    rgb_p = ray_marcher.render_rays(decoder, pred_planes, ray_o, ray_d)
                    
                    # === Hybrid BCE+Dice Loss for Validation ===
                    raw_bce_val = bce_loss_fn(rgb_p, target)
                    loss_bce_val = raw_bce_val.view(B_val, -1).mean(dim=1)
                    loss_dice_val = dice_loss_per_batch(rgb_p, target)
                    
                    val_step_loss = (loss_bce_val + loss_dice_val).mean()
                    val_loss += val_step_loss.item()
                    val_autoregressive_steps += 1
                    
                    # Full Image IoU Calculation
                    # We render the full image for view 0 (Side 1) to get an accurate IoU metric without slowing down validation too much.
                    full_ray_o, full_ray_d = get_full_image_rays(H, W, view_idx=0, device=device)
                    full_ray_o = full_ray_o.unsqueeze(0).expand(B_val, -1, -1)
                    full_ray_d = full_ray_d.unsqueeze(0).expand(B_val, -1, -1)
                    
                    full_rgb_p = render_rays_chunked(ray_marcher, decoder, pred_planes, full_ray_o, full_ray_d, chunk_size=4096)
                    full_target = vids_val[:, t+1, 0].permute(0, 2, 3, 1).reshape(B_val, H*W, 1)

                    # Calculate IoU directly on the flat rays
                    val_iou += calculate_iou(full_rgb_p, full_target).item()

                    # ---------------------------------------------------------
                    # VALIDATION VISUALIZATION BLOCK
                    # ---------------------------------------------------------
                    # Log every 10 epochs, ONLY on the first validation batch
                    if (epoch + 1) % 10 == 0 and val_batch_idx == 0 and (t == (V_Time // 2) or t == (V_Time - 2)):
                        
                        stage_name = "Val_Middle" if t == (V_Time // 2) else "Val_Last"
                        
                        for v in range(Views):
                            real_frame = vids_val[0, t+1, v].detach().cpu()
                            full_ray_origins, full_ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                            full_ray_origins = full_ray_origins.unsqueeze(0)
                            full_ray_dirs = full_ray_dirs.unsqueeze(0)
                            
                            # Safely render the full image using the chunked method
                            single_plane_pred = {key: pred_planes[key][0:1] for key in pred_planes} 
                            full_rgb_pred = render_rays_chunked(
                                ray_marcher, decoder, single_plane_pred, full_ray_origins, full_ray_dirs, chunk_size=4096)
                            
                            pred_frame = full_rgb_pred.view(H, W, C).permute(2, 0, 1).detach().cpu()
                            
                            comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                            writer.add_image(f'Validation_Autoregressive_{stage_name}/Side_{v+1}', comparison_grid, epoch + 1)
                    
                    curr_planes = pred_planes
                
                # ---------------------------------------------------------
                # --- COLD START EVALUATION (NO BURN-IN) ---
                # ---------------------------------------------------------
                # The network is forced to simulate the entire sequence starting 
                # strictly from t=0, making it impossible to cheat using visual momentum.
                curr_planes_cold = encoder(vids_val[:, 0])
                h_val_cold = None
                
                for t in range(V_Time - 1):
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    pred_planes_cold, h_val_cold = dynamics(curr_planes_cold, action_val_clamped, h_val_cold)
                    
                    # Full Image IoU Calculation for view 0
                    full_ray_o_cold, full_ray_d_cold = get_full_image_rays(H, W, view_idx=0, device=device)
                    full_ray_o_cold = full_ray_o_cold.unsqueeze(0).expand(B_val, -1, -1)
                    full_ray_d_cold = full_ray_d_cold.unsqueeze(0).expand(B_val, -1, -1)
                    
                    full_rgb_p_cold = render_rays_chunked(ray_marcher, decoder, pred_planes_cold, full_ray_o_cold, full_ray_d_cold, chunk_size=4096)
                    full_target_cold = vids_val[:, t+1, 0].permute(0, 2, 3, 1).reshape(B_val, H*W, 1)

                    val_cold_start_iou += calculate_iou(full_rgb_p_cold, full_target_cold).item()
                    val_cold_start_steps += 1
                    
                    curr_planes_cold = pred_planes_cold

                # Clean up Validation Memory too
                del vids_val, press_val, curr_planes, h_val, full_rgb_p, full_target
                del curr_planes_cold, h_val_cold, full_rgb_p_cold, full_target_cold # Cleanup Cold Start variables
                
                if str(device) == "mps":
                    torch.mps.empty_cache()
                elif str(device) == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                    
        # Protect against zero-division if validation sequences are too short
        avg_val_loss = val_loss / max(1, val_autoregressive_steps)
        avg_val_iou = val_iou / max(1, val_autoregressive_steps) # Average the IoU over steps
        avg_val_cold_iou = val_cold_start_iou / max(1, val_cold_start_steps) # Average the Cold Start IoU
        
        # Print and Log metrics to TensorBoard
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val IoU: {avg_val_iou:.4f} | Cold Start IoU: {avg_val_cold_iou:.4f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/Validation_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Training/Validation_IoU', avg_val_iou, epoch + 1)
        writer.add_scalar('Validation/Cold_Start_IoU', avg_val_cold_iou, epoch + 1) # NEW: Un-cheatable metric
        
        # Track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # ==========================================
        # --- CHECKPOINT SAVING ---
        # ==========================================
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            
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
            'scheduler_base_lrs': scheduler.base_lrs 
        }
        
        if is_best:
            torch.save(checkpoint_dict, os.path.join(log_dir, "best_model.pth"))
            print(f"*** New Best Model Saved (Val Loss: {best_val_loss:.6f}) ***")

        # Save milestone checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_dict, os.path.join(log_dir, f"world_model_checkpoint_epoch_{epoch+1}.pth"))
            
        # ALWAYS save the latest state
        torch.save(checkpoint_dict, os.path.join(log_dir, "last_checkpoint.pth"))

    writer.close()

if __name__ == "__main__":
    main()