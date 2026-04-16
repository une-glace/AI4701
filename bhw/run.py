import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

LOCAL_ULTRALYTICS = ROOT / "ultralytics_main"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4", ".AVI", ".MKV"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run screw counting on a folder of videos.")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing input videos.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save result.npy.")
    parser.add_argument("--output_time_path", type=str, required=True, help="Path to save time.txt.")
    parser.add_argument("--mask_output_path", type=str, required=True, help="Folder to save mask overlays.")
    parser.add_argument("--weights", type=str, default=str(ROOT / "weights" / "best.pt"), help="YOLO weights path.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config name/path.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.2, help="IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--device", type=str, default="", help="Device to use. Leave empty to auto-select.")
    parser.add_argument("--project", type=str, default=str(ROOT / "runs" / "video_track_count"), help="Intermediate output root.")
    return parser.parse_args()


def detect_device(user_device: str) -> str:
    if user_device:
        return user_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_video_files(data_dir: Path):
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {ext.lower() for ext in VIDEO_EXTENSIONS}
    )


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def choose_middle_visualization(vis_dir: Path, total_frames: int) -> Path | None:
    candidates = sorted(vis_dir.glob("frame_*.jpg"))
    if not candidates:
        return None

    if total_frames <= 0:
        return candidates[len(candidates) // 2]

    target = max(1, total_frames // 2)

    def frame_index(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except Exception:
            return 0

    return min(candidates, key=lambda path: abs(frame_index(path) - target))


def parse_summary_counts(summary_path: Path) -> dict[str, int]:
    counts = {}
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    in_count_section = False
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "# Counts (by class)":
            in_count_section = True
            continue
        if not in_count_section or ":" not in line:
            continue
        name, value = line.split(":", 1)
        name = name.strip()
        value = value.strip()
        try:
            counts[name] = int(value)
        except ValueError:
            continue
    return counts


def get_target_class_names(weights_path: Path) -> list[str]:
    model = YOLO(str(weights_path))
    names = model.names
    if isinstance(names, dict):
        # The background class is 0, so we extract classes 1 to 5 (Screw1 to Screw5)
        return [str(names[idx]) for idx in range(1, 6) if idx in names]
    return [str(names[idx]) for idx in range(1, min(6, len(names)))]


def run_single_video(video_path: Path, args, device: str, target_class_names: list[str]) -> tuple[list[int], Path]:
    total_frames = get_frame_count(video_path)
    save_every = max(1, total_frames // 2) if total_frames > 1 else 1

    command = [
        sys.executable,
        str(ROOT / "video_track_count.py"),
        "--source",
        str(video_path),
        "--weights",
        str(Path(args.weights)),
        "--tracker",
        str(args.tracker),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--imgsz",
        str(args.imgsz),
        "--device",
        device,
        "--save-every",
        str(save_every),
        "--project",
        str(args.project),
    ]

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    subprocess.run(command, cwd=str(ROOT), check=True, env=env)

    project_dir = Path(args.project) / video_path.stem
    summary_path = project_dir / "summary.txt"
    vis_dir = project_dir / "vis"

    raw_counts = parse_summary_counts(summary_path)
    counts = [raw_counts.get(class_name, 0) for class_name in target_class_names]

    chosen_vis = choose_middle_visualization(vis_dir, total_frames)
    if chosen_vis is None:
        raise FileNotFoundError(f"No visualization frames found in {vis_dir}")

    return counts, chosen_vis


def save_mask_image(source_image: Path, destination_path: Path):
    image = cv2.imread(str(source_image))
    if image is None:
        raise RuntimeError(f"Failed to read visualization image: {source_image}")
    cv2.imwrite(str(destination_path), image)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path)
    output_time_path = Path(args.output_time_path)
    mask_output_dir = Path(args.mask_output_path)
    weights_path = Path(args.weights)

    if not data_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {data_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_time_path.parent.mkdir(parents=True, exist_ok=True)

    video_files = get_video_files(data_dir)
    if not video_files:
        raise FileNotFoundError(f"No video files found in: {data_dir}")

    device = detect_device(args.device)
    target_class_names = get_target_class_names(weights_path)
    if len(target_class_names) < 5:
        raise RuntimeError("The loaded model does not expose at least 5 target classes.")

    start_time = time.time()
    results = {}

    for video_path in video_files:
        counts, chosen_vis = run_single_video(video_path, args, device, target_class_names)
        results[video_path.stem] = counts
        mask_dst = mask_output_dir / f"{video_path.stem}_mask.png"
        save_mask_image(chosen_vis, mask_dst)

    elapsed = time.time() - start_time
    np.save(output_path, results)
    output_time_path.write_text(f"{elapsed:.2f}", encoding="utf-8")

    print(f"Processed {len(video_files)} videos.")
    print(f"Results saved to: {output_path}")
    print(f"Time saved to: {output_time_path}")
    print(f"Mask images saved to: {mask_output_dir}")


if __name__ == "__main__":
    main()
