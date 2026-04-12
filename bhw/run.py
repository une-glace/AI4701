import os
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

def process_video(video_path, model, mask_output_dir):
    """
    Process a single video: perform tracking and count unique objects.
    Extracts a mask from the middle frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return [0, 0, 0, 0, 0]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    
    # Use sets to keep track of unique track IDs for each class (0 to 4)
    # Class 5 is 'Others' which we might detect but don't need to count for the output format
    unique_ids = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    
    frame_idx = 0
    video_name = Path(video_path).stem
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8 tracking with ByteTrack
        # persist=True tells the tracker that the frames are sequential
        # conf=0.5 过滤掉低置信度的背景噪点（避免传送带划痕被识别为螺丝）
        results = model.track(frame, persist=True, tracker="custom_tracker.yaml", conf=0.5, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            
            for track_id, class_id, box in zip(track_ids, class_ids, boxes.xyxy):
                # We only care about the 5 screw types (classes 0-4)
                if class_id in unique_ids:
                    # 第三步绝招：边缘鬼影过滤
                    # 只统计中心点在画面安全区内的螺丝，避免边缘截断导致的 ID Switch
                    x1, y1, x2, y2 = box.tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    img_width = frame.shape[1]
                    img_height = frame.shape[0]
                    
                    # 设定边缘 50 像素为“危险区”，不进行计数登记
                    MARGIN = 50
                    if (MARGIN < center_x < img_width - MARGIN) and (MARGIN < center_y < img_height - MARGIN):
                        unique_ids[class_id].add(track_id)
        
        # Save mask for the middle frame
        if frame_idx == mid_frame_idx:
            # Plot the results on the frame (this draws boxes and masks)
            annotated_frame = results[0].plot()
            mask_save_path = os.path.join(mask_output_dir, f"{video_name}_mask.png")
            cv2.imwrite(mask_save_path, annotated_frame)
            
        frame_idx += 1
        
    cap.release()
    
    # Return counts for classes 0 to 4
    counts = [len(unique_ids[c]) for c in range(5)]
    return counts

def main():
    parser = argparse.ArgumentParser(description="Video Screw Counting with YOLOv8-seg and ByteTrack")
    parser.add_argument("--data_dir", type=str, required=True, help="/path/to/test_videos_folder")
    parser.add_argument("--output_path", type=str, required=True, help="./result.npy")
    parser.add_argument("--output_time_path", type=str, required=True, help="./time.txt")
    parser.add_argument("--mask_output_path", type=str, required=True, help="./mask_folder/")
    # You might want to pass the model weight path as well, defaulting to best.pt
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to YOLOv8-seg weights")
    
    args = parser.parse_args()
    
    # Create mask output directory if it doesn't exist
    os.makedirs(args.mask_output_path, exist_ok=True)
    
    # Load the YOLO model
    print(f"Loading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please make sure you have trained the model and provided the correct weight path.")
        return

    start_time = time.time()
    
    results_dict = {}
    
    # Process all video files in the data directory
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(valid_extensions)]
    
    for video_file in video_files:
        video_path = os.path.join(args.data_dir, video_file)
        video_name = Path(video_file).stem
        print(f"Processing {video_file}...")
        
        counts = process_video(video_path, model, args.mask_output_path)
        results_dict[video_name] = counts
        print(f"  Counts: {counts}")
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save result dictionary to .npy
    np.save(args.output_path, results_dict)
    print(f"Results saved to {args.output_path}")
    
    # Save total time to time.txt
    with open(args.output_time_path, 'w') as f:
        f.write(f"{total_time:.2f}")
    print(f"Total time {total_time:.2f}s saved to {args.output_time_path}")

if __name__ == "__main__":
    main()
