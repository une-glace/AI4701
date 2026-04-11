import cv2
import numpy as np
import torch
import os
import glob

# Import Grounding DINO
from groundingdino.util.inference import load_model as load_gdino_model, predict
import groundingdino.datasets.transforms as T
from PIL import Image

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

def load_image_for_dino(image_path):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = Image.open(image_path).convert("RGB")
    image_transformed, _ = transform(image_source, None)
    return image_source, image_transformed

def segment_with_grounded_sam(image_path, output_path, dino_model, sam_predictor, text_prompt, device, box_threshold, text_threshold, iou_threshold):
    print(f"Processing {os.path.basename(image_path)}...")
    
    # 1. Load image
    image_source, image_transformed = load_image_for_dino(image_path)
    image_cv2 = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # 2. Predict Bounding Boxes with Grounding DINO
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    
    print(f"  Grounding DINO found {len(boxes)} boxes for '{text_prompt}'")
    
    if len(boxes) == 0:
        print("  No objects found. Skipping.")
        cv2.imwrite(output_path, image_cv2)
        return
        
    # Convert normalized DINO boxes to absolute pixel coordinates [x1, y1, x2, y2]
    H, W, _ = image_cv2.shape
    boxes_xyxy = []
    for box in boxes:
        # DINO returns normalized [cx, cy, w, h]
        cx, cy, w, h = box
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy.append([x1, y1, x2, y2])
        
    boxes_xyxy = torch.tensor(boxes_xyxy).to(device)
    
    # === NEW: Apply NMS (Non-Maximum Suppression) to remove redundant/overlapping boxes ===
    # Convert logits to a tensor
    scores = torch.tensor(logits).to(device)
    # Use torchvision's nms to keep only the best box when they overlap heavily
    import torchvision
    keep_indices = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold=iou_threshold)
    
    boxes_xyxy = boxes_xyxy[keep_indices]
    print(f"  After NMS (iou={iou_threshold}), keeping {len(boxes_xyxy)} boxes.")
    
    # 3. Set image for SAM
    sam_predictor.set_image(image_rgb)
    
    # 4. Predict Masks with SAM using the DINO boxes as prompts
    # transform_boxes maps coordinates to the 1024x1024 space SAM expects
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_rgb.shape[:2])
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False, # We only want the best mask for each box
    )
    
    # 5. Visualize Results
    overlay = np.zeros_like(image_cv2)
    mask_overall = np.zeros(image_cv2.shape[:2], dtype=bool)
    
    # Process masks (shape: N, 1, H, W)
    # Define a single color for all masks (e.g., a light blue)
    uniform_color = [255, 144, 30] # BGR format, roughly a nice blue

    for i in range(masks.shape[0]):
        m = masks[i, 0].cpu().numpy()
        overlay[m] = uniform_color
        mask_overall = np.logical_or(mask_overall, m)
        
    # Blend overlay
    alpha = 0.5
    result = image_cv2.copy()
    result[mask_overall] = cv2.addWeighted(image_cv2[mask_overall], 1 - alpha, overlay[mask_overall], alpha, 0)
    
    # Draw boundaries and DINO boxes
    # for i in range(masks.shape[0]):
    #     m = masks[i, 0].cpu().numpy().astype(np.uint8)
    #     contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(result, contours, -1, (0, 0, 255), 2) # Red outline for mask
        
        # Optional: Draw the DINO bounding box in green
        # box = boxes_xyxy[i].cpu().numpy().astype(int)
        # cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
    cv2.imwrite(output_path, result)
    print(f"  Saved to {output_path}")

def main():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Setup Grounding DINO ---
    # Path to the config file inside the GroundingDINO repo
    dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = "groundingdino_swint_ogc.pth"
    
    if not os.path.exists(dino_checkpoint):
        print(f"Error: {dino_checkpoint} not found. Please download it.")
        return
        
    print("Loading Grounding DINO...")
    dino_model = load_gdino_model(dino_config, dino_checkpoint, device=device)
    
    # --- 2. Setup SAM ---
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    
    if not os.path.exists(sam_checkpoint):
        print(f"Error: {sam_checkpoint} not found. Please download it.")
        return
        
    print("Loading SAM...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # --- 3. Process Images ---
    data_dir = r"e:\sync\courses\AI4701\hw\hw3\data"
    base_output_dir = r"e:\sync\courses\AI4701\hw\hw3\output_grounded_sam"
    
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    image_paths.sort()
    
    # Text prompt for DINO to find the bounding boxes
    text_prompt = "screw . bolt" 
    
    # Define parameters here for easy tuning and folder naming
    box_thresh = 0.1
    text_thresh = 0.15
    iou_thresh = 0.3
    
    # Create the parameterized folder name
    folder_name = f"{box_thresh}_{text_thresh}"
    if iou_thresh is not None:
        folder_name += f"_{iou_thresh}"
        
    output_dir = os.path.join(base_output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for p in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(p))
        segment_with_grounded_sam(
            p, output_path, dino_model, sam_predictor, text_prompt, device, 
            box_threshold=box_thresh, text_threshold=text_thresh, iou_threshold=iou_thresh
        )

if __name__ == "__main__":
    main()