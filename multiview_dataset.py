import os
import glob
import re

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation

class SoftRobotDataset(Dataset):
    def __init__(self, run_folder, img_size=(128, 128), crop_size=600, image_mode="mask"):
        """
        Args:
            run_folder (str): Path to the main 'Run_YYYY-MM-DD_...' directory.
            img_size (tuple): Target (Height, Width) for the neural network.
            crop_size (int): Center crop square dimension BEFORE resizing.
            image_mode (str): "mask" (silhouette), "rgb" (color), or "grayscale".
        """
        self.run_folder = run_folder
        self.img_size = img_size
        self.crop_size = crop_size
        self.image_mode = image_mode.lower()
        
        # Cache as dictionary to hold processed data in RAM ---
        self.cache = {}
        
        folders = glob.glob(os.path.join(run_folder, "Case_*"))
        self.case_folders = sorted(folders, key=lambda x: int(re.search(r'Case_(\d+)', os.path.basename(x)).group(1)))
        
        if len(self.case_folders) == 0:
            print(f"Warning: No Case folders found in {run_folder}")
            
    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        # Skip OpenCV processing if it is in the cache
        if idx in self.cache:
            return self.cache[idx]
            
        case_folder = self.case_folders[idx]
        
        # 1. LOAD THE PRESSURE PROFILE
        csv_path = glob.glob(os.path.join(case_folder, "*_PressureProfile.csv"))[0]
        df = pd.read_csv(csv_path)
        pressures_kpa = df.iloc[:, 1:4].values.astype(np.float32) 
        
        # 2. LOAD THE VIDEOS
        views = ["ViewSide1", "ViewSide2", "ViewSide3", "ViewTop"]
        all_views_frames = []
        num_frames_in_video = 0
        
        for view_name in views:
            video_path = glob.glob(os.path.join(case_folder, f"*{view_name}.avi"))[0]
            cap = cv2.VideoCapture(video_path)
            
            frames = []
            while True: # Reads video exactly as rendered, preserving the temporal ramp-up/ramp-down sequence!
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # CROP LOGIC
                if self.crop_size is not None:
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2
                    half = self.crop_size // 2
                    frame = frame[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
                
                # --- MULTI-MODE PROCESSING ---
                if self.image_mode == "rgb":
                    frame_c = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_c, self.img_size, interpolation=cv2.INTER_AREA)
                    frame_tensor = frame_resized.astype(np.float32) / 255.0
                    
                elif self.image_mode == "grayscale":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_resized = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_AREA)
                    # Merge to 3 channels to keep ResNet architecture consistent
                    frame_c = cv2.merge([frame_resized, frame_resized, frame_resized])
                    frame_tensor = frame_c.astype(np.float32) / 255.0

                elif self.image_mode == "mask":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, 20, 100)

                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=2)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    mask = np.zeros_like(gray)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                    mask = cv2.erode(mask, kernel, iterations=2)
                    
                    # Ultimate cleanup pass to ensure a single solid silhouette with no holes or noise
                    final_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    clean_mask = np.zeros_like(mask)
                    if final_contours:
                        true_largest = max(final_contours, key=cv2.contourArea)
                        cv2.drawContours(clean_mask, [true_largest], -1, 255, thickness=cv2.FILLED)
                    
                    # KEY FIX: INTER_NEAREST prevents halo artifacts during downscaling
                    mask_resized = cv2.resize(clean_mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                    mask_3c = cv2.merge([mask_resized, mask_resized, mask_resized])
                    frame_tensor = mask_3c.astype(np.float32) / 255.0 
                
                # Channel-first format: [C, H, W]
                frame_tensor = np.transpose(frame_tensor, (2, 0, 1))
                frames.append(frame_tensor)
                
            cap.release()
            video_tensor = torch.tensor(np.array(frames)) # [Time, 3, 128, 128]
            all_views_frames.append(video_tensor)
            num_frames_in_video = len(frames)
            
        videos = torch.stack(all_views_frames, dim=1) # [Time, Views, C, H, W]
        
        # 3. ALIGN TIME & STRICT NORMALIZATION
        aligned_pressures = pressures_kpa[-num_frames_in_video:]
        
        # Convert kPa to Pa to accurately map the physical values
        aligned_pressures_pa = aligned_pressures * 1000.0
        
        # Establishing a hard physical boundary to prevent mathematical collapse or negative vacuums
        MIN_PRESSURE = 1.0 
        aligned_pressures_pa = np.clip(aligned_pressures_pa, a_min=MIN_PRESSURE, a_max=100000.0)
        
        # Normalize to [0, 1] for stable neural network inputs (Assuming 100 kPa is max)
        pressures_tensor = torch.tensor(aligned_pressures_pa / 100000.0) 

        # Save to RAM before returning
        result = {"video": videos, "pressures": pressures_tensor}
        self.cache[idx] = result
        
        return result

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
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    
    # Toggle "mask", "rgb", or "grayscale" directly here!
    dataset = SoftRobotDataset(run_folder=DATA_DIR, img_size=(128, 128), crop_size=600, image_mode="mask")
    
    # Run the debug visualizer and explicitly call the hysteresis plot function
    plot_hysteresis_curve(dataset, sample_index=0)
    visualize_dynamic_multiview(dataset, sample_index=0)