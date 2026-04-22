from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    raw_yaml = project_root / "screw.yaml"
    sliced_yaml = project_root / "screw_sliced.yaml"

    model = YOLO("yolo26m-seg.pt")
    train_results = model.train(
    data=str(raw_yaml),
    epochs=100,
    batch=2,
    imgsz=1024,

    mosaic=0.1,
    mixup=0.1,
    copy_paste=0.3,

    scale=0.9,
    degrees=5.0,

    amp=True,
)
    # print(train_results)