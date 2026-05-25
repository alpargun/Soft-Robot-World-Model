# Measures IoU of the shapes and ignores the black background
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