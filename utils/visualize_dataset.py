import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd

# Get the parent directory's path and add it to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from multiview_dataset import SoftRobotDataset

# ==========================================
# --- HYSTERESIS PLOTTING TOOL ---
# ==========================================
def plot_hysteresis_curve(dataset, sample_index=0):
    print(f"Loading node data for sample {sample_index} to calculate hysteresis...")
    case_folder = dataset.case_folders[sample_index]
    
    node_csv_path = glob.glob(os.path.join(case_folder, "*_NodeData.csv"))[0]
    press_csv_path = glob.glob(os.path.join(case_folder, "*_PressureProfile.csv"))[0]
    
    df_nodes = pd.read_csv(node_csv_path)
    df_press = pd.read_csv(press_csv_path)
    
    df_nodes.columns = df_nodes.columns.str.strip()
    df_press.columns = df_press.columns.str.strip()
    
    df_nodes['TotalDef_mm'] = np.sqrt(df_nodes['DefX(m)']**2 + df_nodes['DefY(m)']**2 + df_nodes['DefZ(m)']**2) * 1000.0
    max_def_per_time = df_nodes.groupby('Time(s)')['TotalDef_mm'].max().reset_index()
    
    df_press['Time(s)'] = df_press['Time(s)'].round(4)
    max_def_per_time['Time(s)'] = max_def_per_time['Time(s)'].round(4)
    
    merged_df = pd.merge(max_def_per_time, df_press, on='Time(s)', how='inner')
    merged_df['Active_Pressure_kPa'] = merged_df[['P1(kPa)', 'P2(kPa)', 'P3(kPa)']].max(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Active_Pressure_kPa'], merged_df['TotalDef_mm'], marker='o', linestyle='-', color='b', markersize=4)
    
    plt.title(f"Hysteresis Curve (Case {sample_index+1})", fontsize=16)
    plt.xlabel("Applied Pressure (kPa)", fontsize=14)
    plt.ylabel("Max Total Deformation (mm)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==========================================
# --- DEBUG: DYNAMIC MULTI-VIEW VISUALIZER ---
# ==========================================
def visualize_dynamic_multiview(dataset, sample_index=0):
    print(f"Loading sample {sample_index} for visualization...")
    sample = dataset[sample_index]
    
    video_tensor = sample["video"]             
    pressures_tensor = sample["pressures"]     
    
    video_np = video_tensor.permute(0, 1, 3, 4, 2).numpy() 
    pressures_np = (pressures_tensor * 100000.0).int().numpy()   
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    view_names = ["Side 1 (+X)", "Side 2 (+Y)", "Side 3 (-X)", "Top (+Z)"]
    
    p_init = pressures_np[0]
    title = fig.suptitle(f"Time: 0.00s | P1: {p_init[0]} | P2: {p_init[1]} | P3: {p_init[2]} Pa", fontsize=14)
    
    ims = []
    for i, ax in enumerate(axes):
        ax.axis('off')
        ax.set_title(view_names[i])
        im = ax.imshow(video_np[0, i])
        ims.append(im)
    
    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_array(video_np[frame_idx, i])
            
        p_curr = pressures_np[frame_idx]
        current_time = (frame_idx + 1) * (2.0 / 60)
        title.set_text(f"Time: {current_time:.2f}s | P1: {p_curr[0]} | P2: {p_curr[1]} | P3: {p_curr[2]} Pa")
        return ims + [title]
    
    anim = animation.FuncAnimation(fig, update, frames=len(video_np), interval=33, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = r"/Users/alp/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    
    # Toggle "mask", "rgb", or "grayscale" directly here!
    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode="mask")
    
    # Run the debug visualizer and explicitly call the hysteresis plot function
    plot_hysteresis_curve(dataset, sample_index=0)
    visualize_dynamic_multiview(dataset, sample_index=0)