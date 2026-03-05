import os
import glob
import re
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

def get_tip_displacement(obj_files, physical_box_size_mm=150.0):
    """
    Calculates the total displacement of the robot's tip across a sequence of meshes.
    Assumes the base of the robot is at the origin/bottom and the tip moves the most.
    """
    mm_per_voxel = physical_box_size_mm / 2.0 # The 3D grid spans from -1.0 to 1.0 (length of 2.0)
    
    displacements_mm = []
    initial_tip_pos = None
    
    for f in obj_files:
        mesh = o3d.io.read_triangle_mesh(f)
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) == 0:
            displacements_mm.append(0.0)
            continue
            
        # Assuming the tip is the point furthest from the origin or highest on the Z/Y axis.
        # Here we track the point with the maximum distance from the base.
        # You may need to change 'vertices[:, 2]' to 0 or 1 depending on your Ansys UP axis.
        tip_index = np.argmax(vertices[:, 2]) 
        current_tip_pos = vertices[tip_index]
        
        if initial_tip_pos is None:
            initial_tip_pos = current_tip_pos
            displacements_mm.append(0.0)
        else:
            # Euclidean distance from the resting position
            dist_voxels = np.linalg.norm(current_tip_pos - initial_tip_pos)
            dist_mm = dist_voxels * mm_per_voxel
            displacements_mm.append(dist_mm)
            
    return displacements_mm

def main():
    # 1. Configuration - UPDATE THESE
    DATA_DIR = r"/Users/alp/Desktop/SoftRobot_Dataset_Hysteresis/Run_2026-03-01_23-47-27"
    TEST_CASE_IDX = -1 # Matches the [-1] subset from your inference script
    PHYSICAL_BOX_SIZE_MM = 150.0 # UPDATE THIS to the real size of your Ansys bounding box!

    # 2. Get the AI Predicted Displacements
    print("Loading AI Predicted 3D Meshes...")
    obj_files = sorted(glob.glob("AI_Predicted_Frame_*.obj"), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    
    if len(obj_files) != 60:
        print(f"Warning: Expected 60 .obj files, found {len(obj_files)}. Did you run inference_3d_sequence.py yet?")
        return

    ai_displacements = get_tip_displacement(obj_files, PHYSICAL_BOX_SIZE_MM)

    # 3. Get the Ansys Ground Truth Displacements
    print("Loading Ansys Ground Truth CSVs...")
    folders = glob.glob(os.path.join(DATA_DIR, "Case_*"))
    case_folders = sorted(folders, key=lambda x: int(re.search(r'Case_(\d+)', os.path.basename(x)).group(1)))
    target_case_folder = case_folders[TEST_CASE_IDX]

    node_csv_path = glob.glob(os.path.join(target_case_folder, "*_NodeData.csv"))[0]
    press_csv_path = glob.glob(os.path.join(target_case_folder, "*_PressureProfile.csv"))[0]
    
    df_nodes = pd.read_csv(node_csv_path)
    df_press = pd.read_csv(press_csv_path)
    df_nodes.columns = df_nodes.columns.str.strip()
    df_press.columns = df_press.columns.str.strip()
    
    # Calculate Ground Truth Total Deflection
    df_nodes['TotalDef_mm'] = np.sqrt(df_nodes['DefX(m)']**2 + df_nodes['DefY(m)']**2 + df_nodes['DefZ(m)']**2) * 1000.0
    ansys_def_per_time = df_nodes.groupby('Time(s)')['TotalDef_mm'].max().reset_index()
    
    # Align pressures for the X-axis
    df_press['Time(s)'] = df_press['Time(s)'].round(4)
    ansys_def_per_time['Time(s)'] = ansys_def_per_time['Time(s)'].round(4)
    merged_df = pd.merge(ansys_def_per_time, df_press, on='Time(s)', how='inner')
    active_pressures = merged_df[['P1(kPa)', 'P2(kPa)', 'P3(kPa)']].max(axis=1).values
    ansys_displacements = merged_df['TotalDef_mm'].values

    # 4. Plot the Hysteresis Comparison
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # Ensure arrays match in length (sometimes Ansys has extra sub-steps, so we align by frame count)
    min_len = min(len(active_pressures), len(ai_displacements), len(ansys_displacements))
    
    plt.plot(active_pressures[:min_len], ansys_displacements[:min_len], 
             marker='o', linestyle='-', color='blue', label='Ansys (Ground Truth)', markersize=5)
    
    plt.plot(active_pressures[:min_len], ai_displacements[:min_len], 
             marker='x', linestyle='--', color='red', label='AI World Model (Predicted)', markersize=5)
    
    plt.title("Hysteresis Curve: Ansys vs AI Prediction", fontsize=16)
    plt.xlabel("Applied Pressure (kPa)", fontsize=14)
    plt.ylabel("Max Tip Deflection (mm)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Hysteresis_Evaluation_Plot.png")
    plt.show()

if __name__ == "__main__":
    main()