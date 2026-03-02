import os
import glob
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MultiViewSoftRobotDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 128), crop_size=800):
        """
        Loads 4 synced views and a sequential pressure tensor [Time, 3]
        """
        self.data_dir = data_dir
        self.views_to_load = ["Side1", "Side2", "Side3", "Top"]
        
        transform_list = []
        if crop_size:
            transform_list.append(T.CenterCrop(crop_size))
        transform_list.append(T.Resize(size=image_size, antialias=True))
        self.spatial_transform = T.Compose(transform_list)
        
        self.case_folders = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("Case_")
        ])
        
        print(f"Found {len(self.case_folders)} cases. Loading {len(self.views_to_load)} views per case.")

    def __len__(self):
        return len(self.case_folders)

    def _read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        video_np = np.array(frames, dtype=np.float32) / 255.0 
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)
        video_tensor = self.spatial_transform(video_tensor)
        return video_tensor

    def __getitem__(self, idx):
        folder_path = self.case_folders[idx]
        
        # 1. Load Multi-View Videos
        view_tensors = []
        for view in self.views_to_load:
            video_pattern = os.path.join(folder_path, f"*_View{view}.avi")
            video_file = glob.glob(video_pattern)[0]
            v_tensor = self._read_video(video_file) 
            view_tensors.append(v_tensor)
            
        # Shape: [Time=60, Views=4, Channels=3, H=128, W=128]
        multi_view_video = torch.stack(view_tensors, dim=1)
            
        # 2. Parse the CSV for Sequential Pressures
        csv_pattern = os.path.join(folder_path, "*_Data.csv")
        csv_file = glob.glob(csv_pattern)[0]
        
        # Read only the necessary columns to save RAM
        df = pd.read_csv(csv_file, usecols=['Time(s)', ' Inst_P1(Pa)', ' Inst_P2(Pa)', ' Inst_P3(Pa)'])
        
        # Drop duplicates to isolate the 60 unique time steps, and ensure they are in order
        df_unique = df.drop_duplicates(subset=['Time(s)']).sort_values(by='Time(s)')
        
        # Extract columns as numpy arrays
        p1_seq = df_unique[' Inst_P1(Pa)'].values
        p2_seq = df_unique[' Inst_P2(Pa)'].values
        p3_seq = df_unique[' Inst_P3(Pa)'].values
        
        # Stack into [60, 3] and normalize to [0, 1] based on 100k Max Pressure
        pressures_seq = np.stack([p1_seq, p2_seq, p3_seq], axis=1)
        pressures_tensor = torch.tensor(pressures_seq, dtype=torch.float32) / 100000.0
        
        return {
            "video": multi_view_video,           # Shape: [60, 4, 3, H, W]
            "pressures": pressures_tensor        # Shape: [60, 3]
        }


# ==========================================
# --- DEBUG: DYNAMIC MULTI-VIEW VISUALIZER ---
# ==========================================
def visualize_dynamic_multiview(dataset, sample_index=0):
    """
    Plays all 4 views side-by-side and dynamically updates the instantaneous 
    pressure text for every frame.
    """
    print(f"Loading sample {sample_index} for visualization...")
    sample = dataset[sample_index]
    
    video_tensor = sample["video"]             # Shape: [Time=60, Views=4, C=3, H, W]
    pressures_tensor = sample["pressures"]     # Shape: [Time=60, 3]
    
    # Un-normalize data back to raw formats for viewing
    video_np = video_tensor.permute(0, 1, 3, 4, 2).numpy() # [60, 4, H, W, 3]
    pressures_np = (pressures_tensor * 100000.0).int().numpy()   # [60, 3] in Pascals
    
    # Setup Figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    view_names = ["Side 1 (+X)", "Side 2 (+Y)", "Side 3 (-X)", "Top (+Z)"]
    
    # Initialize the plot with Frame 0
    p_init = pressures_np[0]
    title = fig.suptitle(f"Time: 0.00s | P1: {p_init[0]} | P2: {p_init[1]} | P3: {p_init[2]} Pa", fontsize=14)
    
    ims = []
    for i, ax in enumerate(axes):
        ax.axis('off')
        ax.set_title(view_names[i])
        im = ax.imshow(video_np[0, i])
        ims.append(im)
    
    # Animation Update Function
    def update(frame_idx):
        # Update the 4 images
        for i, im in enumerate(ims):
            im.set_array(video_np[frame_idx, i])
            
        # Update the title with the instantaneous pressure for this exact frame
        p_curr = pressures_np[frame_idx]
        current_time = (frame_idx + 1) * (2.0 / 60)
        title.set_text(f"Time: {current_time:.2f}s | P1: {p_curr[0]} | P2: {p_curr[1]} | P3: {p_curr[2]} Pa")
        
        return ims + [title]
    
    # Play Animation (~33ms interval = 30 FPS)
    anim = animation.FuncAnimation(fig, update, frames=len(video_np), interval=33, repeat=True)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# --- DEBUG ---
# ==========================================
if __name__ == "__main__":
    # Point this to your new fast-export run folder
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_15-07-54"
    
    dataset = MultiViewSoftRobotDataset(DATA_DIR, image_size=(128, 128), crop_size=800)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        videos = batch["video"]
        actions = batch["pressures"]
        
        print(f"Batch Video Shape: {videos.shape}") 
        print(f"Batch Actions Shape: {actions.shape}") 
        # Actions should now print exactly [Batch_Size, 60, 3]!
        break

    # Visualize the first case
    visualize_dynamic_multiview(dataset, sample_index=0)
