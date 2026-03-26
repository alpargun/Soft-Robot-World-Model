import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    # UPDATE THIS to the path of your specific events.out.tfevents file
    LOG_DIR = "/Users/alp/Desktop/SoftRobot_WorldModel/runs/SoftRobot_Train_MASK_2026-03-04_18-01-05" 
    
    # The exact tags TensorBoard uses for your metrics
    TRAIN_TAG = 'Training/Sequence_Loss'
    VAL_TAG = 'Training/Validation_Loss'
    TF_TAG = 'Training/TF_Ratio'

    print(f"Loading TensorBoard logs from: {LOG_DIR}")
    
    # ==========================================
    # 2. EXTRACT DATA
    # ==========================================
    # Load the event accumulator
    event_acc = EventAccumulator(LOG_DIR)
    event_acc.Reload()

    # Extract Train Loss
    train_events = event_acc.Scalars(TRAIN_TAG)
    train_steps = [x.step for x in train_events]
    train_values = [x.value for x in train_events]

    # Extract Val Loss
    val_events = event_acc.Scalars(VAL_TAG)
    val_steps = [x.step for x in val_events]
    val_values = [x.value for x in val_events]

    # ==========================================
    # 3. PLOT FORMATTING (Academic Style)
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot lines
    ax.plot(train_steps, train_values, label='Training Loss', color='#1f77b4', linewidth=2)
    ax.plot(val_steps, val_values, label='Validation Loss', color='#ff7f0e', linewidth=2)

    # Add a vertical line where Teacher Forcing hits 0 (Epoch 100)
    ax.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='0% Teacher Forcing')

    # Labels and Title
    ax.set_title('Autoregressive World Model Convergence', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sequence Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    
    # Styling
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=11, loc='upper right', frameon=True, framealpha=0.9, shadow=True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ==========================================
    # 4. SAVE AND SHOW
    # ==========================================
    plt.tight_layout()
    output_filename = "Formal_Loss_Curve.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Plot successfully saved as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    plot_tensorboard_logs()