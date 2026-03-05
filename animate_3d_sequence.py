import open3d as o3d
import glob
import time
import re

def sort_key(filename):
    """Extracts the frame number to ensure chronological sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def main():
    # 1. Gather all predicted .obj frames
    obj_files = sorted(glob.glob("AI_Predicted_Frame_*.obj"), key=sort_key)
    
    if len(obj_files) == 0:
        print("Error: No .obj files found in the current directory.")
        return

    print(f"Loading {len(obj_files)} meshes into memory. This may take a few seconds...")
    
    # 2. Pre-load and colorize meshes for smooth 30FPS playback
    meshes = []
    for f in obj_files:
        mesh = o3d.io.read_triangle_mesh(f)
        mesh.compute_vertex_normals() # Calculates lighting shadows
        mesh.paint_uniform_color([0.2, 0.5, 0.8]) # Paints the robot a clean Ansys-style blue
        meshes.append(mesh)

    print("Starting 3D Viewer. Use your mouse to rotate and scroll to zoom!")
    
    # 3. Initialize the Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AI Soft Robot Autoregressive Rollout", width=1024, height=768)
    
    # Render settings
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True # Ensures hollow parts of the mesh render correctly
    opt.background_color = np.asarray([0.1, 0.1, 0.15]) # Dark grey/blue background
    
    # 4. The Animation Loop
    # We add the first mesh to establish the camera tracking bounds
    current_mesh = meshes[0]
    vis.add_geometry(current_mesh)
    
    frame_idx = 0
    while True:
        # Clear the old frame and load the next one
        vis.clear_geometries()
        
        # CRITICAL: reset_bounding_box=False prevents the camera from violently snapping 
        # back to the center every single frame.
        vis.add_geometry(meshes[frame_idx], reset_bounding_box=False)
        
        # Step the visualizer forward
        if not vis.poll_events():
            break # Breaks loop if the user clicks the 'X' to close the window
            
        vis.update_renderer()
        time.sleep(1 / 30.0) # Lock to roughly 30 FPS
        
        # Loop back to frame 0 when the sequence ends
        frame_idx = (frame_idx + 1) % len(meshes)
        
    vis.destroy_window()
    print("Viewer closed.")

if __name__ == "__main__":
    # We need numpy just for the background color array
    import numpy as np 
    main()