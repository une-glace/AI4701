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
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            
            for track_id, class_id in zip(track_ids, class_ids):
                # We only care about the 5 screw types (classes 0-4)
                if class_id in unique_ids:
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
