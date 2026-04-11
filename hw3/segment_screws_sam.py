import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import glob

def segment_with_sam(image_path, output_path, mask_generator):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Processing {os.path.basename(image_path)} with SAM...")
    
    # Generate masks automatically
    masks = mask_generator.generate(image_rgb)
    
    print(f"  Found {len(masks)} potential masks.")
    
    # Filter masks (optional, based on your knowledge of the scene)
    # We want to keep masks that are likely to be screws.
    # SAM returns a list of dicts, each containing 'segmentation' (bool array), 'area', 'predicted_iou', etc.
    
    # Let's filter out very large masks (like the background) or very small noise
    valid_masks = []
    for mask_data in masks:
        area = mask_data['area']
        # Typical screw area from previous experiments is roughly between 2000 and 100000 pixels
        if 2000 < area < 150000:
            valid_masks.append(mask_data)
            
    print(f"  Kept {len(valid_masks)} valid masks after size filtering.")
            
    # Create overlay
    overlay = np.zeros_like(image)
    mask_overall = np.zeros(image.shape[:2], dtype=bool)
    
    # Sort masks by area to draw larger ones first (so smaller ones are drawn on top if they overlap)
    valid_masks = sorted(valid_masks, key=(lambda x: x['area']), reverse=True)
    
    for mask_data in valid_masks:
        m = mask_data['segmentation']
        color = np.random.randint(50, 255, size=(3,), dtype=int).tolist()
        overlay[m] = color
        mask_overall = np.logical_or(mask_overall, m)
        
    # Blend
    alpha = 0.5
    result = image.copy()
    result[mask_overall] = cv2.addWeighted(image[mask_overall], 1 - alpha, overlay[mask_overall], alpha, 0)
    
    # Draw boundaries
    for mask_data in valid_masks:
        m = mask_data['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
        
    cv2.imwrite(output_path, result)
    print(f"  Saved to {output_path}")

def main():
    # 1. Setup SAM
    sam_checkpoint = "sam_vit_b_01ec64.pth" # Make sure this file is downloaded!
    model_type = "vit_b"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(sam_checkpoint):
        print(f"Error: {sam_checkpoint} not found.")
        print("Please download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return
        
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Create automatic mask generator
    # We can tweak these parameters to get better results
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32, # Increase for more dense point sampling
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,  # Minimum area to keep
    )
    
    # 2. Process images
    data_dir = r"e:\sync\courses\AI4701\hw\hw3\data"
    output_dir = r"e:\sync\courses\AI4701\hw\hw3\output_sam"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    image_paths.sort()
    
    for p in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(p))
        segment_with_sam(p, output_path, mask_generator)

if __name__ == "__main__":
    main()
