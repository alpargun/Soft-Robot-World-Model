import torch

# Measure IoU of the shapes and ignores the black background
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

# Calculates Total Variation loss to prevent high-frequency grid artifacts
def calculate_tv_loss(tri_planes):
    tv_loss = 0.0
    for key in ['xy', 'xz', 'yz']:
        plane = tri_planes[key]
        tv_h = torch.mean(torch.abs(plane[:, :, 1:, :] - plane[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(plane[:, :, :, 1:] - plane[:, :, :, :-1]))
        tv_loss += (tv_h + tv_w)
    return tv_loss / 3.0