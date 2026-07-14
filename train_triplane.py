import os
# Use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import custom modules
from src.multiview_dataset import SoftRobotDataset
from src.utils.losses import dice_loss_per_batch, calculate_tv_loss
from src.renderer import VolumetricRayMarcher, sample_orthographic_rays, get_full_image_rays, render_rays_chunked

# Import the new Triplane E2E components
from src.encoder_triplane import TriplaneEncoder
from src.temporal_dynamics_triplane import DynamicsTriplane
from src.decoder_nof import NOFDecoder

def main():
    
    # 1. Configuration
    EXPERIMENT_NAME = "multiView_Triplane_JEPA"
    MASTER_DIR = r"/home/alp/Desktop/SoftRobot_Dataset_Hysteresis"
    
    # Automatically grab all folders inside MASTER_DIR except "old"
    DATA_DIRS = [
        os.path.join(MASTER_DIR, d) for d in os.listdir(MASTER_DIR) 
        if os.path.isdir(os.path.join(MASTER_DIR, d)) and d != "old"
    ]
    DATA_DIRS.sort() 
    print(f"Discovered {len(DATA_DIRS)} valid Run folders.")

    IMAGE_MODE = "mask"

    # Initialize TensorBoard Writer and Log Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/{EXPERIMENT_NAME}_{IMAGE_MODE.upper()}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")
    print(f"Checkpoints will be saved to: {log_dir}")
      
    RESUME_CHECKPOINT_PATH = '' # If left empty, training starts from scratch.
    
    BATCH_SIZE = 4 # or 4 if GPU memory allows
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000

    FRAME_STRIDE = 2 # Skip every other frame (15 fps) to reduce temporal redundancy and speed up training
    SEQUENCE_LENGTH = 24
    FEATURE_DIM = 64
    
    # Increased back to 4096 because Activation Checkpointing frees the required VRAM
    RAYS_PER_STEP = 8192 # More rays = better quality but slower training.
    BURN_IN_LENGTH = 5 # Number of frames to "burn in" the hidden state before autoregression begins
    TF_UNTIL = 400 # Number of epochs to decay teacher forcing from 1.0 to 0.0
    CURRICULUM_SCHEDULE = [30, 70] # Curriculum learning stages: crawl, walk, run
    VAL_PERCENTAGE = 0.15 # Percentage of pure bending cases to hold out for validation

    # Loss Weights
    lambda_latent = 1.0 # Controls JEPA latent consistency to prevent 3D hallucinations
    lambda_inverse = 0.5 # Controls penalty for predicting the wrong action history from physical change
    lambda_plane_sparse = 0.005 # Controls penalty for drawing unnecessary pixels on the latent planes
    lambda_tv = 0.01 # Controls penalty for high-frequency grid artifacts on the latent planes

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

    print(f"Initializing World Model Training on: {device}")

    # Initialize Dataset
    train_base = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, 
        seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE
    )
    
    # Validation Base: seq_len=None. Returns the full original sequences
    val_base = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, 
        seq_len=None, frame_stride=FRAME_STRIDE
    )

    # Validation Split
    all_bending_indices = []
    special_indices = []
    
    for idx, folder_path in enumerate(train_base.case_folders):
        folder_name = os.path.basename(folder_path)
        if folder_name.startswith("Case_"):
            all_bending_indices.append(idx)
        else:
            # Find Staircase, PE, Random Walk or any other non-standard folders
            special_indices.append(idx)
    
    # Set seed for reproducible splits across runs.
    random.seed(42)
    
    num_val_cases = int(len(all_bending_indices) * VAL_PERCENTAGE)
    
    # Randomly pick indices from ONLY the pure bending cases
    val_indices = random.sample(all_bending_indices, num_val_cases)
    
    # Training indices: All un-selected bending cases plus all special cases (Creep/PE/Random walk)
    train_indices = [i for i in all_bending_indices if i not in val_indices] + special_indices
    
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize models
    encoder = TriplaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = DynamicsTriplane(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=64).to(device)
    decoder = NOFDecoder(feature_dim=FEATURE_DIM).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device) 

    # Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=1e-6) # AdamW decouples weight decay from grad update
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Resume from checkpoint
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
        
        if 'scheduler_base_lrs' in checkpoint:
            scheduler.base_lrs = checkpoint['scheduler_base_lrs']
            
        start_epoch = checkpoint['epoch'] 
        best_val_loss = checkpoint['best_val_loss']

    # Define Loss Functions
    bce_loss_fn = nn.BCELoss(reduction='none') # reduction='none' so it outputs per-pixel losses

    # Step-Wise Curriculum Scheduler
    def get_curriculum_seq_len(current_epoch):
        # Ensure sequence length is always greater than BURN_IN_LENGTH so Autoregression always has frames to predict
        if current_epoch < CURRICULUM_SCHEDULE[0]:
            return BURN_IN_LENGTH + 4   # Crawl: Predict 4 steps into the future
        elif current_epoch < CURRICULUM_SCHEDULE[1]:
            return BURN_IN_LENGTH + 11  # Walk: Predict 11 steps into the future
        else:
            return SEQUENCE_LENGTH      # Run: Predict the full 24-frame sequence

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        # Calculate teacher forcing probability (Decays from 1.0 to 0.0 over the first TF_UNTIL epochs)
        tf_prob = max(0.0, 1.0 - (epoch / float(TF_UNTIL))) if TF_UNTIL > 0 else 0.0
        
        # Wraps the dataloader to show a progress bar for the current epoch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            videos = batch["video"].to(device)
            pressures = batch["pressures"].to(device) # Pressures are pre-normalized by Dataset
            
            # Extract specific views [B, Time, Views, C, H, W] 
            v_s1 = videos[:, :, 0]
            v_s2 = videos[:, :, 1]
            v_s3 = videos[:, :, 2]
            v_top = videos[:, :, 3]
            
            B, Time, C, H, W = v_s1.shape 
            optimizer.zero_grad()
            hidden_state = None 
            
            # Initial visual state
            current_triplane = encoder(v_top[:, 0], v_s1[:, 0], v_s2[:, 0], v_s3[:, 0])

            # Initialize variables for the Autoregressive Phase
            batch_sequence_loss = 0.0
            autoregressive_steps = 0
            
            # BURN-IN
            # 30% of the time, we force a "Cold Start". The network gets NO visual momentum history and must learn 
            # to break static inertia and map pressure to movement from a dead stop.
            current_burn_in = BURN_IN_LENGTH if random.random() < 0.70 else 1
            
            # PHASE 1: BURN-IN
            for t in range(current_burn_in - 1):
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                
                # Step the physics engine to build memory
                _, hidden_state = dynamics(current_triplane, action_t, hidden_state)
                
                # Force the visual state to reality (Teacher Forcing) for the next step
                current_triplane = encoder(v_top[:, t+1], v_s1[:, t+1], v_s2[:, t+1], v_s3[:, t+1])
                
            current_max_seq = get_curriculum_seq_len(epoch)
            time_limit = min(Time, current_max_seq)

            # PHASE 2: AUTOREGRESSION
            for t in range(current_burn_in - 1, time_limit - 1):
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                
                # Predict the next 3D state blindly using the dynamics engine
                triplane_next_pred, hidden_state = dynamics(current_triplane, action_t, hidden_state)

                # JEPA Latent Consistency Loss: Forces the network to predict the next latent state without peeking at the ground truth
                # Encode target frame. Detach to train Dynamics, not Encoder.
                triplane_next_true = encoder(v_top[:, t+1], v_s1[:, t+1], v_s2[:, t+1], v_s3[:, t+1])
                loss_latent = sum([F.mse_loss(triplane_next_pred[k], triplane_next_true[k].detach()) for k in ['xy', 'xz', 'yz']])
                # ====================================================================

                # Sequence Inverse Dynamics Loss: Forces the network to predict the action history from physical change
                loss_inverse = 0.0
                history_len = dynamics.history_len
                
                # Only calculate inverse loss if we have enough historical frames
                if t >= history_len - 1:
                    # Slice the last 'n' ground truth actions
                    target_action_seq = pressures[:, t - history_len + 1 : t + 1]
                    
                    # Predict the sequence of actions and calculate the loss
                    pred_action_seq = dynamics.predict_inverse_action_sequence(current_triplane, triplane_next_pred)
                    loss_inverse = F.mse_loss(pred_action_seq, target_action_seq)

                # End-to-end raycasting loss: Compare the predicted 3D state to the next frame's ground truth
                frames_next_true = videos[:, t+1] # Shape: [B, Views, C, H, W]
                
                # Shoot rays through the 3D space
                ray_origins, ray_dirs, target_pixels = sample_orthographic_rays(
                    frames_next_true, num_samples=RAYS_PER_STEP)
                
                # Render rays using Activation Checkpointing to drastically lower VRAM usage
                pred_pixels = torch.utils.checkpoint.checkpoint(ray_marcher.render_rays, decoder, triplane_next_pred, ray_origins, ray_dirs, use_reentrant=False)
                
                # Force all values perfectly into the [0, 1] range to prevent CUDA precision crashes
                pred_pixels = torch.clamp(pred_pixels, min=1e-5, max=1.0 - 1e-5)
                target_pixels = torch.clamp(target_pixels, min=0.0, max=1.0)
                
                # Calculate 2D pixel losses on the raycasted output
                raw_bce = bce_loss_fn(pred_pixels, target_pixels)
                loss_bce = raw_bce.view(B, -1).mean(dim=1) 
                loss_dice = dice_loss_per_batch(pred_pixels, target_pixels) 
                
                # Forces every pixel on the latent planes to remain 0.0 unless specifically needed to draw the robot
                loss_plane_sparsity = sum([torch.mean(torch.abs(triplane_next_pred[k])) for k in ['xy', 'xz', 'yz']])
                
                # TV Loss to prevent high-frequency grid artifacts
                loss_tv = calculate_tv_loss(triplane_next_pred)
                
                # Final summation
                step_loss = (loss_bce + loss_dice + (lambda_latent * loss_latent) + (lambda_inverse * loss_inverse) + (lambda_tv * loss_tv) + (lambda_plane_sparse * loss_plane_sparsity)).mean()
                
                batch_sequence_loss += step_loss
                autoregressive_steps += 1
                
                # Scheduled Sampling: Decides whether to self-correct using ground truth or run blind
                if random.random() < tf_prob:
                    # Recycle the true Triplane we just generated for JEPA
                    current_triplane = triplane_next_true
                else:
                    current_triplane = triplane_next_pred

            # We only average the loss and backpropagate if we predicted more than 0 steps
            if autoregressive_steps > 0:
                batch_sequence_loss = batch_sequence_loss / autoregressive_steps
                batch_sequence_loss.backward()
                
                # Gradient clipping prevents the network from blowing up during pure autoregression
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_sequence_loss.item()
                
        # Log the inverse and latent losses to tensorboard at the end of the epoch
        writer.add_scalar('Training/Inverse_Action_Loss', loss_inverse.item() if isinstance(loss_inverse, torch.Tensor) else 0.0, epoch + 1)
        writer.add_scalar('Training/Latent_Consistency_Loss', loss_latent.item() if isinstance(loss_latent, torch.Tensor) else 0.0, epoch + 1)
        writer.add_scalar('Training/Teacher_Forcing_Prob', tf_prob, epoch + 1)
        
        # Step the learning rate down appropriately per epoch
        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        
        # --------------------------------------------------------------------------------------------------------------
        # VALIDATION
        encoder.eval()
        dynamics.eval()
        decoder.eval()
        val_loss = 0.0
        val_autoregressive_steps = 0
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                vids_val = batch["video"].to(device)
                press_val = batch["pressures"].to(device)
                val_s1, val_s2, val_s3, val_top = vids_val[:,:,0], vids_val[:,:,1], vids_val[:,:,2], vids_val[:,:,3]
                B_val, V_Time, Views, C, H, W = vids_val.shape
                
                curr_trip = encoder(val_top[:, 0], val_s1[:, 0], val_s2[:, 0], val_s3[:, 0])
                h_val = None

                # VAL PHASE 1: BURN-IN
                for t in range(BURN_IN_LENGTH - 1):
                    action_val = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    _, h_val = dynamics(curr_trip, action_val, h_val)
                    curr_trip = encoder(val_top[:, t+1], val_s1[:, t+1], val_s2[:, t+1], val_s3[:, t+1])
                
                # VAL PHASE 2: AUTOREGRESSION
                for t in range(BURN_IN_LENGTH - 1, V_Time - 1):
                    # Apply the same clamping to validation pressures
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    # Feed the clamped action into the dynamics engine
                    pred_trip, h_val = dynamics(curr_trip, action_val_clamped, h_val)
                    
                    frames_next_true_val = vids_val[:, t+1]
                    
                    # Compute fast L1 loss on rays to track generalization
                    ray_o, ray_d, target_p = sample_orthographic_rays(
                        frames_next_true_val, num_samples=RAYS_PER_STEP
                    )
                    pred_p = ray_marcher.render_rays(decoder, pred_trip, ray_o, ray_d)
                    
                    #  CUDA BCELOSS fix
                    pred_p = torch.clamp(pred_p, min=1e-5, max=1.0 - 1e-5)
                    target_p = torch.clamp(target_p, min=0.0, max=1.0)
                    
                    # Hybrid BCE+Dice Loss for Validation
                    l_bce = bce_loss_fn(pred_p, target_p).view(B_val, -1).mean(dim=1)
                    l_dice = dice_loss_per_batch(pred_p, target_p)
                    
                    val_step_loss = (l_bce + l_dice).mean()
                    
                    val_loss += val_step_loss.item()
                    val_autoregressive_steps += 1

                    # VALIDATION VISUALIZATION
                    # Log every 10 epochs, ONLY on the first validation batch
                    if (epoch + 1) % 10 == 0 and val_batch_idx == 0 and (t == (V_Time // 2) or t == (V_Time - 2)):
                        stage_name = "Val_Middle" if t == (V_Time // 2) else "Val_Last"
                        
                        for v in range(Views):
                            real_frame = vids_val[0, t+1, v].detach().cpu()
                            full_ray_origins, full_ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                            full_ray_origins = full_ray_origins.unsqueeze(0)
                            full_ray_dirs = full_ray_dirs.unsqueeze(0)
                            
                            # Safely render the full image using the chunked method
                            single_pred_trip = {key: pred_trip[key][0:1] for key in pred_trip}
                            full_rgb_pred = render_rays_chunked(
                                ray_marcher, decoder, single_pred_trip, full_ray_origins, full_ray_dirs, chunk_size=4096
                            )
                            pred_frame = full_rgb_pred.view(H, W, C).permute(2, 0, 1).detach().cpu()
                            
                            comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                            writer.add_image(f'Validation_E2E_{stage_name}/Side_{v+1}', comparison_grid, epoch + 1)

                    curr_trip = pred_trip

        # Safely average the validation metrics
        safe_val_steps = max(1, val_autoregressive_steps)
        avg_val_loss = val_loss / safe_val_steps
        
        # Print and Log to TensorBoard
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/Validation_Loss', avg_val_loss, epoch + 1)
        # Track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # SAVE CHECKPOINT
        checkpoint_dict = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_indices': train_dataset.indices, 
            'val_indices': val_dataset.indices,  
            'encoder': encoder.state_dict(),
            'dynamics': dynamics.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        
        # Curriculum aware checkpointing: Only save after the curriculum has fully unlocked the 24-frame sequence
        if (epoch + 1) >= CURRICULUM_SCHEDULE[-1]:  # After the last curriculum phase
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint_dict, os.path.join(log_dir, "best_model.pth"))
                print(f"*** New Best Model Saved (Val Loss: {best_val_loss:.6f}) ***")

        # Save checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_dict, os.path.join(log_dir, f"world_model_checkpoint_epoch_{epoch+1}.pth"))
        
        # Always save the latest state so progress is never lost during sudden stops
        torch.save(checkpoint_dict, os.path.join(log_dir, "last_checkpoint.pth"))

    writer.close()

if __name__ == "__main__":
    main()