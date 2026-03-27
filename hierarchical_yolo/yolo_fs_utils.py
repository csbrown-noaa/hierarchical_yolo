import os
import shutil

def enforce_symlinks(json_paths: list[str], src_data_dir: str, depth_dest_dir: str) -> None:
    """Safely constructs image directory symlinks pointing back to the master data pool."""
    print("  -> Enforcing Symlinks for image directories...")
    for json_path in json_paths:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        
        master_img_dir = os.path.abspath(os.path.join(src_data_dir, basename, "images"))
        yolo_img_dir = os.path.join(depth_dest_dir, basename, "images")
        
        # Ensure the master image dir actually exists so the symlink is valid
        os.makedirs(master_img_dir, exist_ok=True)
        # Ensure the destination's parent directory exists
        os.makedirs(os.path.dirname(yolo_img_dir), exist_ok=True)
        
        # If pycocowriter accidentally created an empty images dir previously, nuke it
        if os.path.isdir(yolo_img_dir) and not os.path.islink(yolo_img_dir):
            shutil.rmtree(yolo_img_dir)
            
        # Plant the symlink
        if not os.path.exists(yolo_img_dir):
            os.symlink(master_img_dir, yolo_img_dir)
