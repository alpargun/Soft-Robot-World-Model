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
    IMAGE_MODE = "rgb" # Change to "rgb" to automatically enable LPIPS perceptual loss!
    
    BATCH_SIZE = 2 # or 4 if GPU memory allows
    FEATURE_DIM = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 500
    RAYS_PER_STEP = 1600 # Number of rays to sample per time step for loss calculation (VRAM optimization). Increase for better quality
    
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/SoftRobot_Train_{IMAGE_MODE.upper()}_{timestamp}")
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
    decoder = TriPlaneDecoder(feature_dim=FEATURE_DIM, image_mode=IMAGE_MODE).to(device)
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
        
        # Calculate Teacher Forcing Ratio: 
        # Starts at 1.0 (100% help) and decays to 0.0 (0% help) by epoch 250
        tf_ratio = max(0.0, 1.0 - (epoch / (NUM_EPOCHS * 0.5)))
        
        for batch_idx, batch in enumerate(dataloader):
            videos = batch["video"].to(device)       
            pressures = batch["pressures"].to(device) 
            
            B, Time, Views, C, H, W = videos.shape 
            
            optimizer.zero_grad()
            batch_sequence_loss = 0.0
            
            hidden_state = None 
            
            # Establish the initial state
            current_tri_planes = encoder(videos[:, 0])
            
            for t in range(Time - 1):
                action_t = pressures[:, t]        
                frames_next_true = videos[:, t+1] 
                
                # Predict the next 3D state
                planes_next_pred, hidden_state = dynamics(current_tri_planes, action_t, hidden_state)
                
                # Render the current frame
                ray_origins, ray_dirs, target_rgb = sample_orthographic_rays(frames_next_true, num_samples=RAYS_PER_STEP, image_mode=IMAGE_MODE)
                rgb_pred = ray_marcher.render_rays(decoder, planes_next_pred, ray_origins, ray_dirs)
                
                # 1. Standard L1 Loss (Works on the flat [B, RAYS, 3] tensors)
                step_loss = l1_loss_fn(rgb_pred, target_rgb)
                
                # 2. Perceptual LPIPS Loss (Requires 2D image patches)
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
                if (epoch + 1) % 10 == 0 and batch_idx == 0 and t == (Time - 2):
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
                            writer.add_image(f'Comparison_View/Side_{v+1}', comparison_grid, epoch + 1)
                # ---------------------------------------------------------

            batch_sequence_loss = batch_sequence_loss / (Time - 1)
            batch_sequence_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_sequence_loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | TF Ratio: {tf_ratio:.2f} | Avg Seq Loss: {avg_loss:.6f}")
        writer.add_scalar('Training/Sequence_Loss', avg_loss, epoch + 1)
        writer.add_scalar('Training/TF_Ratio', tf_ratio, epoch + 1)

        if (epoch + 1) % 50 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'dynamics': dynamics.state_dict(),
                'decoder': decoder.state_dict(),
            }, f"world_model_checkpoint_epoch_{epoch+1}.pth")

    writer.close()

if __name__ == "__main__":
    main()