import os
import glob
import re
import random

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SoftRobotDataset(Dataset):
    def __init__(self, run_folders, img_size=(128, 128), crop_size=600, image_mode="mask", seq_len=24, frame_stride=2):
        """
        Args:
            run_folders (list or str): A single path OR a list of paths to your dataset directories.
            img_size (tuple): Target (Height, Width) for the neural network.
            crop_size (int): Center crop square dimension BEFORE resizing.
            image_mode (str): "mask" (silhouette), "rgb" (color), or "grayscale".
        """
        # Ensure it's a list even if a single string is passed
        if isinstance(run_folders, str):
            self.run_folders = [run_folders]
        else:
            self.run_folders = run_folders
            
        self.img_size = img_size
        self.crop_size = crop_size
        self.image_mode = image_mode.lower()
        self.seq_len = seq_len 
        self.frame_stride = frame_stride # Temporal stride to skip frames
        
        self.case_folders = []
        
        # Iterate through every master folder provided
        for folder in self.run_folders:
            # Find valid subdirectories in this specific folder
            subdirs = [
                os.path.join(folder, d) for d in os.listdir(folder) 
                if os.path.isdir(os.path.join(folder, d)) and 
                   (d.startswith("Case_") or "Staircase" in d)
            ]
            
            # Sort them locally so the order is deterministic
            def smart_sort(folder_path):
                folder_name = os.path.basename(folder_path)
                match = re.search(r'Case_(\d+)', folder_name)
                if match:
                    return (0, int(match.group(1))) 
                elif "Staircase" in folder_name:
                    return (1, folder_name)  
                else:
                    return (2, folder_name)  
                    
            sorted_subdirs = sorted(subdirs, key=smart_sort)
            
            # Append the absolute paths to our master list
            self.case_folders.extend(sorted_subdirs)
        
        if len(self.case_folders) == 0:
            print(f"Warning: No valid folders found in the provided paths.")
        else:
            print(f"Successfully discovered {len(self.case_folders)} total sequence folders across {len(self.run_folders)} directories.")
            
    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]

        # Embed the specific parameters into the filename
        cache_name = f"processed_cache_{self.image_mode}_{self.img_size[0]}x{self.img_size[1]}_crop{self.crop_size}.pt"
        cache_file_path = os.path.join(case_folder, cache_name)
        
        # Skip OpenCV processing if it is in the cache (Load directly from disk)
        if os.path.exists(cache_file_path):
            result = torch.load(cache_file_path, weights_only=True)
        else:
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

            # Save to DISK before returning
            result = {"video": videos, "pressures": pressures_tensor}
            torch.save(result, cache_file_path)

        # ==========================================
        # --- TEMPORAL SUBSAMPLING ---
        # ==========================================
        videos = result["video"]          # Shape: [Time, Views, C, H, W]
        pressures = result["pressures"]   # Shape: [Time, 3]
        
        # 1. Apply the Temporal Stride
        # Skips redundant frames to force the network to learn translation, not memorization.
        videos = videos[::self.frame_stride]
        pressures = pressures[::self.frame_stride]
        
        total_subsampled_frames = videos.shape[0]
        
        # 2. Dynamic Temporal Chunking & Cold Start Anchoring
        # self.seq_len represents the number of STRIDED frames (23 frames ~ 1.5 seconds)
        if self.seq_len is not None and total_subsampled_frames > self.seq_len:
            max_start = total_subsampled_frames - self.seq_len
            
            # Cold Start Anchoring: Force 30% of the training sequences to start exactly at Frame 0.
            if random.random() < 0.30:
                start_idx = 0
            else:
                start_idx = random.randint(0, max_start)
                
            end_idx = start_idx + self.seq_len
            
            videos = videos[start_idx:end_idx]
            pressures = pressures[start_idx:end_idx]
            
        return {"video": videos, "pressures": pressures}