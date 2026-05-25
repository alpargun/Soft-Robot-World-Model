import os
# Use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import custom modules
from src.multiview_dataset import SoftRobotDataset
from src.utils.losses import dice_loss_per_batch

# Import the 2D model components
from src.decoder_2d import Decoder2D
from src.encoder_2d import Encoder2D
from src.temporal_dynamics_2d import Dynamics2D


def main():
    
    EXPERIMENT_NAME = "singleView12"
    MASTER_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis"
    
    # Automatically grab all folders inside MASTER_DIR except "old"
    DATA_DIRS = [
        os.path.join(MASTER_DIR, d) for d in os.listdir(MASTER_DIR) 
        if os.path.isdir(os.path.join(MASTER_DIR, d)) and d != "old"
    ]
    
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

    FRAME_STRIDE = 2 # Skip every other frame to force learning of dynamics, not just memorization.
    SEQUENCE_LENGTH = 24
    FEATURE_DIM = 64
    
    BURN_IN_LENGTH = 5 
    VAL_PERCENTAGE = 0.15 # Percentage of pure bending cases to hold out for validation

    # Check for GPU availability and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Initializing World Model Training on: {device} | Mode: {IMAGE_MODE.upper()}")

    # Initialize Dataset
    train_base = SoftRobotDataset(
        run_folders=DATA_DIRS, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE, 
        seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE
    )
    
    # Validation Base: seq_len=None returns the full original sequences
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
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize 2D Model Components
    encoder = Encoder2D(feature_dim=FEATURE_DIM).to(device)
    dynamics = Dynamics2D(feature_dim=FEATURE_DIM, action_dim=3, action_embed_dim=64).to(device)
    decoder = Decoder2D(feature_dim=FEATURE_DIM).to(device)

    # Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=1e-6) # AdamW decouples weight decay from grad update
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # RESUME CHECKPOINT
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

    # Step-Wise Curriculum Scheduler
    def get_curriculum_seq_len(current_epoch):
        # Ensure the total sequence length is always greater than BURN_IN_LENGTH (5)
        # so that Phase 2 (Autoregression) always has frames to predict.
        if current_epoch < 30:
            return BURN_IN_LENGTH + 4   # Crawl: Predict 4 steps into the future (Total 9)
        elif current_epoch < 70:
            return BURN_IN_LENGTH + 11  # Walk: Predict 11 steps into the future (Total 16)
        else:
            return SEQUENCE_LENGTH      # Run: Predict the full 24-frame sequence

    # 5. The Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        # Wraps the dataloader to show a progress bar for the current epoch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            # Extract just the first Side View (index 0)
            videos = batch["video"][:, :, 0].to(device) # only side view 1
            pressures = batch["pressures"].to(device) # Pressures are pre-normalized by Dataset!
            
            B, Time, C, H, W = videos.shape 
            
            optimizer.zero_grad()
            
            hidden_state = None 
            
            # Establish the initial visual state
            current_features = encoder(videos[:, 0])

            # Trackers for the Autoregressive Phase
            batch_sequence_loss = 0.0
            autoregressive_steps = 0
            
            # DYNAMIC BURN-IN
            # 30% of the time, we force a "Cold Start". The network gets NO visual momentum history and must learn to 
            # break static inertia and map pressure to movement from a dead stop.
            current_burn_in = BURN_IN_LENGTH if random.random() < 0.70 else 1
            
            for t in range(current_burn_in - 1):
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                
                # Step the physics engine to build memory
                _, hidden_state = dynamics(current_features, action_t, hidden_state)
                
                # Force the visual state to reality (Teacher Forcing) for the next step
                current_features = encoder(videos[:, t+1])
                
            # AUTOREGRESSION
            current_max_seq = get_curriculum_seq_len(epoch)
            time_limit = min(Time, current_max_seq)

            for t in range(current_burn_in - 1, time_limit - 1):
                
                action_t = torch.clamp(pressures[:, t], min=0.00001, max=1.0)
                frames_next_true = videos[:, t+1]
                
                # Predict next latent state
                features_next_pred, hidden_state = dynamics(current_features, action_t, hidden_state)

                # SEQUENCE INVERSE AUXILIARY LOSS
                loss_inverse = 0.0
                history_len = dynamics.history_len
                
                # Only calculate inverse loss if we have enough historical frames to compare against
                if t >= history_len - 1:
                    # Slice the last 'n' ground truth actions
                    target_action_seq = pressures[:, t - history_len + 1 : t + 1]
                    
                    # Predict the sequence of actions that caused this hysteresis state
                    pred_action_seq = dynamics.predict_inverse_action_sequence(current_features, features_next_pred)
                    
                    # Apply heavy MSE penalty
                    loss_inverse = F.mse_loss(pred_action_seq, target_action_seq)
                
                lambda_inverse = 2.0 # Strong weight to forcefully bind actions to physics

                # Render the 2D frame directly (No Ray Marching)
                rgb_pred = decoder(features_next_pred)
                
                # 2D Combined BCE and Dice Loss
                raw_bce = bce_loss_fn(rgb_pred, frames_next_true)
                loss_bce = raw_bce.view(B, -1).mean(dim=1) 
                loss_dice = dice_loss_per_batch(rgb_pred, frames_next_true)
                
                # Aggregate losses with appropriate weighting
                step_loss = (loss_bce + loss_dice + (lambda_inverse * loss_inverse)).mean()
                
                batch_sequence_loss += step_loss
                autoregressive_steps += 1
                
                # STRICT AUTOREGRESSION: Feed the predicted features back in for the next step
                current_features = features_next_pred

            # We only average the loss and backpropagate if we actually predicted steps
            if autoregressive_steps > 0:
                batch_sequence_loss = batch_sequence_loss / autoregressive_steps
                batch_sequence_loss.backward()
                
                # Gradient clipping prevents the network from blowing up during pure autoregression
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_sequence_loss.item()
                
        # Log the inverse loss to tensorboard at the end of the epoch
        writer.add_scalar('Training/Inverse_Action_Loss', loss_inverse.item() if isinstance(loss_inverse, torch.Tensor) else 0.0, epoch + 1)

        # Step the learning rate down appropriately per epoch
        scheduler.step()
            
        avg_loss = epoch_loss / len(dataloader)
        
        # VALIDATION PHASE
        encoder.eval()
        dynamics.eval()
        decoder.eval()
        val_loss = 0.0
        val_autoregressive_steps = 0
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                vids_val = batch["video"][:,:,0].to(device)
                press_val = batch["pressures"].to(device)
                B_val, V_Time, C, H, W = vids_val.shape
                
                curr_feat = encoder(vids_val[:, 0])
                h_val = None

                # BURN-IN
                # Kept strictly at BURN_IN_LENGTH so validation metrics remain 
                # completely stable and comparable epoch-to-epoch.
                for t in range(BURN_IN_LENGTH - 1):
                    action_val = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    _, h_val = dynamics(curr_feat, action_val, h_val)
                    curr_feat = encoder(vids_val[:, t+1])
                
                # AUTOREGRESSION
                for t in range(BURN_IN_LENGTH - 1, V_Time - 1):
                    # Apply the same clamping to validation pressures
                    action_val_clamped = torch.clamp(press_val[:, t], min=0.00001, max=1.0)
                    
                    # Feed the clamped action into the dynamics engine
                    pred_feat, h_val = dynamics(curr_feat, action_val_clamped, h_val)
                    
                    # Render the predicted features to RGB for loss calculation
                    rgb_p = decoder(pred_feat)
                    
                    # Hybrid BCE+Dice Loss for Validation
                    target = vids_val[:, t+1]
                    raw_bce_val = bce_loss_fn(rgb_p, target)
                    loss_bce_val = raw_bce_val.view(B_val, -1).mean(dim=1)
                    loss_dice_val = dice_loss_per_batch(rgb_p, target)
                    
                    val_loss += (loss_bce_val + loss_dice_val).mean().item()
                    val_autoregressive_steps += 1

                    # VALIDATION VISUALIZATION
                    # Log every 10 epochs, ONLY on the first validation batch
                    if (epoch + 1) % 10 == 0 and val_batch_idx == 0 and (t == (V_Time // 2) or t == (V_Time - 2)):
                        
                        stage_name = "Val_Middle" if t == (V_Time // 2) else "Val_Last"
                        real_frame = target[0].detach().cpu()
                        pred_frame = rgb_p[0].detach().cpu()
                        comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                        writer.add_image(f'Validation_Autoregressive_{stage_name}', comparison_grid, epoch + 1)

                    curr_feat = pred_feat

        # Safely average the validation metrics
        safe_val_steps = max(1, val_autoregressive_steps)
        avg_val_loss = val_loss / safe_val_steps
        
        # Print and Log to TensorBoard
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/Validation_Loss', avg_val_loss, epoch + 1)
        # Track LR visually
        writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # SAVE CHECKPOINT SAVING
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

        # Save checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_dict, os.path.join(log_dir, f"world_model_checkpoint_epoch_{epoch+1}.pth"))
            
        # Always save the latest state so progress is never lost during sudden stops
        torch.save(checkpoint_dict, os.path.join(log_dir, "last_checkpoint.pth"))

    writer.close()

if __name__ == "__main__":
    main()