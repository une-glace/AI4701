import cv2
import numpy as np
import os
import glob

def segment_screws(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening to remove small noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find external contours to get filled masks
    contours_ext, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(opening)
    cv2.drawContours(filled_mask, contours_ext, -1, 255, -1)
    
    # Connected components on filled mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    
    screw_mask_final = np.zeros_like(opening)
    
    screw_count = 0
    for i in range(1, num_labels):
        filled_area = stats[i, cv2.CC_STAT_AREA]
        if filled_area < 2000: # Ignore noise
            continue
            
        # Get original area
        comp_mask = np.uint8(labels == i)
        original_area = np.sum(opening[comp_mask == 1] > 0)
        
        ratio = original_area / filled_area
        
        # Nuts have ratio < 0.75 because of the large hole
        if ratio > 0.75:
            # It's a screw! Add its ORIGINAL pixels to the final mask
            # We use the original 'opening' mask to keep the exact shape
            screw_mask_final = cv2.bitwise_or(screw_mask_final, cv2.bitwise_and(opening, opening, mask=comp_mask))
            screw_count += 1
            
    print(f"{os.path.basename(image_path)}: Found {screw_count} screws.")
    
    # Now label the screws
    ret, screw_labels = cv2.connectedComponents(screw_mask_final)
    
    # Create color overlay
    overlay = np.zeros_like(img)
    for label in range(1, ret):
        color = np.random.randint(50, 255, size=(3,), dtype=int).tolist()
        overlay[screw_labels == label] = color
        
    # Blend
    alpha = 0.5
    mask_overall = screw_labels > 0
    result = img.copy()
    result[mask_overall] = cv2.addWeighted(img[mask_overall], 1 - alpha, overlay[mask_overall], alpha, 0)
    
    # Draw boundaries
    contours_screws, _ = cv2.findContours(screw_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours_screws, -1, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, result)

def main():
    data_dir = r"e:\sync\courses\AI4701\hw\hw3\data"
    output_dir = r"e:\sync\courses\AI4701\hw\hw3\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    image_paths.sort()
    for p in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(p))
        segment_screws(p, output_path)

if __name__ == "__main__":
    main()
