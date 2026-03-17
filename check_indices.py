import torch

# Put your exact checkpoint path here
CHECKPOINT_PATH = "runs/resnetGN_decoderConcat_125cases_clampfix_MASK_2026-03-14_03-47-46/last_checkpoint.pth"

print(f"Loading checkpoint: {CHECKPOINT_PATH} ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

if 'val_indices' in checkpoint:
    print("\n=========================================")
    print(f"YOUR SAVED VALIDATION INDICES ARE:")
    print(checkpoint['val_indices'])
    print(f"Total validation cases: {len(checkpoint['val_indices'])}")
    print("=========================================\n")
else:
    print("Error: 'val_indices' not found in this checkpoint dict.")