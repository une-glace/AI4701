from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from screw_counting.yolo_pipeline import YoloScrewCountingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count 5 screw types in top-view images.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing test images.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to output .npy file.")
    parser.add_argument(
        "--output_time_path",
        type=Path,
        required=True,
        help="Path to output total runtime in seconds.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=PROJECT_ROOT / "models" / "best.pt",
        help="Path to trained YOLO checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda or 0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=None,
        help="Optional directory to save debug visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = YoloScrewCountingPipeline(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        debug_dir=args.debug_dir,
    )
    pipeline.run_and_save(
        data_dir=args.data_dir,
        output_path=args.output_path,
        output_time_path=args.output_time_path,
    )


if __name__ == "__main__":
    main()
