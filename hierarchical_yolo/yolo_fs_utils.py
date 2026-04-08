import os
import shutil

def enforce_symlinks(json_paths: list[str], src_data_dir: str, depth_dest_dir: str) -> None:
    """Safely constructs image file hard links pointing back to the master data pool."""
    print("  -> Enforcing Hard Links for image directories...")
    for json_path in json_paths:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        
        master_img_dir = os.path.abspath(os.path.join(src_data_dir, basename, "images"))
        yolo_img_dir = os.path.join(depth_dest_dir, basename, "images")
        
        # Ensure the master image dir actually exists
        os.makedirs(master_img_dir, exist_ok=True)
        
        # If a legacy directory symlink exists from a previous run, nuke it
        if os.path.islink(yolo_img_dir):
            os.unlink(yolo_img_dir)
            
        # Create the actual localized images directory
        os.makedirs(yolo_img_dir, exist_ok=True)
        
        # Iterate through the master directory and hard link individual files
        if os.path.exists(master_img_dir):
            for filename in os.listdir(master_img_dir):
                master_file = os.path.join(master_img_dir, filename)
                yolo_file = os.path.join(yolo_img_dir, filename)
                
                # Only link files, and only if they don't already exist in the target
                if os.path.isfile(master_file) and not os.path.exists(yolo_file):
                    try:
                        os.link(master_file, yolo_file)
                    except OSError as e:
                        print(f"Warning: Failed to hard link {filename}. {e}")
