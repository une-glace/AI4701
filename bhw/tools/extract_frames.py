import cv2
import os
import glob
import shutil

def extract_frames(video_path, output_dir, frame_interval=15):
    """
    Extract one frame every `frame_interval` frames from a video and save to an output directory.
    """
    # Remove old directory if it exists to avoid mixing old and new extracted frames
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {os.path.basename(video_path)} - FPS: {fps}, Total Frames: {total_frames}, Interval: {frame_interval} frames")
    
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract 1 frame every 15 frames
        if frame_idx % frame_interval == 0:
            out_filename = f"frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Finished {os.path.basename(video_path)}: Saved {saved_count} frames to {output_dir}")

def main():
    input_dir = r"e:\sync\courses\AI4701\hw\bhw\vedio_exp"
    # Find all MOV files
    video_files = glob.glob(os.path.join(input_dir, "*.MOV"))
    
    if not video_files:
        print(f"No .MOV files found in {input_dir}")
        return

    for video_path in video_files:
        # Get video name without extension (e.g., 'IMG_2374')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Create a separate folder for each video
        output_dir = os.path.join(input_dir, video_name)
        
        extract_frames(video_path, output_dir, frame_interval=15)

if __name__ == "__main__":
    main()
