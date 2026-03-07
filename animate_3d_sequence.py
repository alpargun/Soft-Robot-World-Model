import open3d as o3d
import glob
import time
import re
import os
import numpy as np 

def sort_key(filename):
    """Extracts the frame number to ensure chronological sorting."""
    # os.path.basename ensures we only look at the file name, not the folder path numbers
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else 0

def main():
    # ==========================================
    # --- 1. AUTO-FIND LATEST EXTRACTION ---
    # ==========================================
    # Automatically grabs the newest 'mesh_extraction' folder based on modification time
    extraction_folders = sorted(glob.glob("mesh_extraction_*"), key=os.path.getmtime)
    
    if not extraction_folders:
        print("Error: No 'mesh_extraction_...' folders found in the current directory.")
        return
        
    TARGET_DIR = extraction_folders[-1]
    print(f"Loading meshes from latest run: {TARGET_DIR}")

    # Gather all predicted .obj frames using the new naming convention
    search_pattern = os.path.join(TARGET_DIR, "frame_*.obj")
    obj_files = sorted(glob.glob(search_pattern), key=sort_key)
    
    if len(obj_files) == 0:
        print(f"Error: No .obj files found in {TARGET_DIR}.")
        return

    # ==========================================
    # --- 2. PRE-LOAD AND COLORIZE ---
    # ==========================================
    print(f"Loading {len(obj_files)} meshes into memory. This may take a few seconds...")
    
    meshes = []
    for f in obj_files:
        mesh = o3d.io.read_triangle_mesh(f)
        mesh.compute_vertex_normals() # Calculates lighting shadows
        mesh.paint_uniform_color([0.2, 0.5, 0.8]) # Paints the robot a clean Ansys-style blue
        meshes.append(mesh)

    print("Starting 3D Viewer. Use your mouse to rotate and scroll to zoom!")
    
    # ==========================================
    # --- 3. INITIALIZE VISUALIZER ---
    # ==========================================
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AI Soft Robot Autoregressive Rollout", width=1024, height=768)
    
    # Render settings
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True # Ensures hollow parts of the mesh render correctly
    opt.background_color = np.asarray([0.1, 0.1, 0.15]) # Dark grey/blue background
    
    # ==========================================
    # --- 4. THE ANIMATION LOOP ---
    # ==========================================
    # We add the first mesh to establish the camera tracking bounds
    current_mesh = meshes[0]
    vis.add_geometry(current_mesh)
    
    frame_idx = 0
    while True:
        vis.clear_geometries()
        
        # CRITICAL: reset_bounding_box=False prevents the camera from violently snapping 
        # back to the center every single frame.
        vis.add_geometry(meshes[frame_idx], reset_bounding_box=False)
        
        if not vis.poll_events():
            break # Breaks loop if the user clicks the 'X' to close the window
            
        vis.update_renderer()
        time.sleep(1 / 30.0) # Lock to roughly 30 FPS
        
        # Loop back to frame 0 when the sequence ends
        frame_idx = (frame_idx + 1) % len(meshes)
        
    vis.destroy_window()
    print("Viewer closed.")

if __name__ == "__main__":
    main()