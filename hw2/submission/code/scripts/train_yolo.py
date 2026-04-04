from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for 5-way screw detection and counting.")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "dataset" / "dataset.yaml",
        help="Path to YOLO dataset.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base YOLO model checkpoint or model name.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="YOLO device string, e.g. 0 or cpu.")
    parser.add_argument(
        "--project",
        type=Path,
        default=project_root / "models" / "yolo_runs",
        help="Training output directory.",
    )
    parser.add_argument("--name", type=str, default="screw_yolo11s")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        pretrained=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
