import os
# Tell PyTorch to use the CPU for any missing Apple GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from multiview_dataset import SoftRobotDataset
from encoder import ResNetTriPlaneEncoder
from temporal_dynamics import TriPlaneDynamics
from decoder import TriPlaneDecoder
from volumetric_ray_marcher import VolumetricRayMarcher
from orthographic_ray_generator import sample_orthographic_rays
from visualization_helper import get_full_image_rays


def main():
    # 1. Configuration
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    IMAGE_MODE = "mask" # Change to "rgb" to automatically enable LPIPS perceptual loss!
    
    BATCH_SIZE = 2 # or 4 if GPU memory allows
    FEATURE_DIM = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 500
    RAYS_PER_STEP = 1024 # Number of rays to sample per time step for loss calculation (VRAM optimization). 2048 for better quality
    
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

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=f"runs/SoftRobot_Train_{IMAGE_MODE.upper()}")
    print("TensorBoard is active. Run 'tensorboard --logdir=runs' to view.")

    # 2. Initialize Dataset
    dataset = SoftRobotDataset(DATA_DIR, img_size=(128, 128), crop_size=600, image_mode=IMAGE_MODE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # ===========================================================================================================
    # OVERFIT TEST: Uncomment the following lines to quickly test the training loop on a single batch
    overfit_dataset = Subset(dataset, indices=[-1, -2]) # Grab only the last 2 cases
    # Pass the overfit_dataset to the dataloader instead of the full_dataset
    dataloader = DataLoader(overfit_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # ===========================================================================================================

    # 3. Initialize Model Components
    encoder = ResNetTriPlaneEncoder(feature_dim=FEATURE_DIM).to(device)
    dynamics = TriPlaneDynamics(feature_dim=FEATURE_DIM, action_dim=3).to(device)
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM).to(device)
    ray_marcher = VolumetricRayMarcher(num_samples=64).to(device)

    # 4. Optimizer Setup
    all_params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE)
    
    # L1 loss is often better for image data as it encourages sharper predictions compared to MSE which can be blurrier
    l1_loss_fn = nn.L1Loss() 
    
    if IMAGE_MODE == "rgb":
        # LPIPS is a perceptual loss function that is more robust than MSE. or "alex"
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) 

    # 5. The Training Loop
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        dynamics.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            videos = batch["video"].to(device)       # [B, Time, 4, 3, 128, 128]
            pressures = batch["pressures"].to(device) # [B, Time, 3]
            
            B, Time, Views, C, H, W = videos.shape 
            
            optimizer.zero_grad()
            batch_sequence_loss = 0.0
            
            hidden_state = None 
            
            # Autoregressive roll-out
            for t in range(Time - 1):
                frames_t = videos[:, t]           
                action_t = pressures[:, t]        
                frames_next_true = videos[:, t+1] 
                
                # Forward Pass
                tri_planes_t = encoder(frames_t)
                planes_next_pred, hidden_state = dynamics(tri_planes_t, action_t, hidden_state)
                
                # Render and Calculate Loss (Passing IMAGE_MODE controls the ray patch shape)
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                
                # Base L1 Loss
                step_loss = l1_loss_fn(rgb_pred, target_rgb)

                if IMAGE_MODE == "rgb":
                    # LPIPS expects images in [-1, 1] format and shape [B, C, H, W]. 
                    # Because we use "rgb" mode, ray_generator sampled perfect 16x16 patches.
                    pred_img = rgb_pred.view(-1, 3, 16, 16) * 2.0 - 1.0 
                    target_img = target_rgb.view(-1, 3, 16, 16) * 2.0 - 1.0

                    loss_perceptual = lpips_loss_fn(pred_img, target_img).mean()
                    
                    # Combine them (you can tune the weighting later)
                    step_loss = step_loss + (0.1 * loss_perceptual)

                batch_sequence_loss += step_loss
            
            # ---------------------------------------------------------
            # VISUALIZATION BLOCK (All 4 Views -> TensorBoard)
            # ---------------------------------------------------------
            if (epoch + 1) % 10 == 0 and batch_idx == 0 and t == (Time - 2):
                with torch.no_grad():
                    # Iterate through all 4 camera views
                    for v in range(Views):
                        
                        # 1. Extract the Real Frame for view 'v'
                        real_frame = frames_next_true[0, v].detach().cpu() # [3, H, W]
                        
                        # 2. Generate Full Image Rays for view 'v'
                        full_ray_origins, full_ray_dirs = get_full_image_rays(H, W, view_idx=v, device=device)
                        full_ray_origins = full_ray_origins.unsqueeze(0)  # Add batch dimension: [1, num_rays, 3]
                        full_ray_dirs = full_ray_dirs.unsqueeze(0)        # Add batch dimension: [1, num_rays, 3]
                        
                        # 3. Render the Predicted Frame
                        single_plane_pred = {key: planes_next_pred[key][0:1] for key in planes_next_pred} 
                        full_rgb_pred = ray_marcher.render_rays(decoder, single_plane_pred, full_ray_origins, full_ray_dirs)
                        
                        # Convert to [C, H, W] for TensorBoard
                        pred_frame = full_rgb_pred.view(H, W, 3).permute(2, 0, 1).detach().cpu()
                        
                        # 4. Save Image to TensorBoard (Side-by-side: Real | Predicted)
                        comparison_grid = torch.cat((real_frame, pred_frame), dim=2)
                        writer.add_image(f'Comparison_View/Side_{v+1}', comparison_grid, epoch + 1)
            # ---------------------------------------------------------

            # Average loss over the sequence and backpropagate
            batch_sequence_loss = batch_sequence_loss / (Time - 1)
            batch_sequence_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_sequence_loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Average Sequence Loss: {avg_loss:.6f}")
        
        # Write loss to TensorBoard
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)

        # Optional: Save checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'dynamics': dynamics.state_dict(),
                'decoder': decoder.state_dict(),
            }, f"world_model_checkpoint_epoch_{epoch+1}.pth")

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()