import os
import shutil
import random
from pathlib import Path

def preprocess_dataset(src_dir, dest_dir, split_ratio=0.8):
    """
    1. Reads original labels, subtracts 1 from the class ID.
    2. Splits dataset into train and val sets.
    """
    src_images = Path(src_dir) / 'images'
    src_labels = Path(src_dir) / 'filtered_labels'
    
    # Create destination directories
    train_img_dir = Path(dest_dir) / 'train' / 'images'
    train_lbl_dir = Path(dest_dir) / 'train' / 'labels'
    val_img_dir = Path(dest_dir) / 'val' / 'images'
    val_lbl_dir = Path(dest_dir) / 'val' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all image files
    image_files = [f for f in os.listdir(src_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_split(files, out_img_dir, out_lbl_dir):
        for img_file in files:
            # Copy image
            shutil.copy(src_images / img_file, out_img_dir / img_file)
            
            # Process label
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_lbl_path = src_labels / lbl_file
            dest_lbl_path = out_lbl_dir / lbl_file
            
            if src_lbl_path.exists():
                with open(src_lbl_path, 'r') as f_in, open(dest_lbl_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if parts:
                            # Subtract 1 from class ID
                            new_class_id = int(parts[0]) - 1
                            # Safety check: ensure class ID is within 0-5
                            if 0 <= new_class_id <= 5:
                                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                                f_out.write(new_line)

    print("Processing training set...")
    process_split(train_files, train_img_dir, train_lbl_dir)
    
    print("Processing validation set...")
    process_split(val_files, val_img_dir, val_lbl_dir)
    
    print(f"Done! Train: {len(train_files)}, Val: {len(val_files)}")
    print(f"Processed dataset saved to: {dest_dir}")

if __name__ == "__main__":
    # Adjust paths as needed
    SOURCE_DATASET = r"e:\sync\courses\AI4701\hw\bhw\dataset"
    DEST_DATASET = r"e:\sync\courses\AI4701\hw\bhw\dataset_yolo"
    
    preprocess_dataset(SOURCE_DATASET, DEST_DATASET)
