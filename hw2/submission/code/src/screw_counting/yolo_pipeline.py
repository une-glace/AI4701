from __future__ import annotations

import time
from pathlib import Path

import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_yolo_device(device: str) -> str | int | None:
    if device == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device == "cuda":
        return 0
    return device


class YoloScrewCountingPipeline:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        debug_dir: Path | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run `pip install -r requirements.txt` first."
            ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint_path}")

        self.model = YOLO(str(checkpoint_path))
        self.class_names = self._load_class_names()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = resolve_yolo_device(device)
        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _load_class_names(self) -> list[str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return [names[idx] for idx in sorted(names)]
        if isinstance(names, list):
            return names
        raise ValueError("Failed to read class names from YOLO model.")

    def infer_image(self, image_path: Path) -> list[int]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return [0 for _ in self.class_names]

        result = results[0]
        counts = np.zeros(len(self.class_names), dtype=np.int32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else []
        for class_id in classes:
            if 0 <= class_id < len(self.class_names):
                counts[class_id] += 1

        if self.debug_dir is not None:
            plotted = result.plot()
            save_path = self.debug_dir / f"{image_path.stem}_pred.jpg"
            from PIL import Image

            Image.fromarray(plotted[..., ::-1]).save(save_path)

        return counts.tolist()

    def run_directory(self, data_dir: Path) -> dict[str, list[int]]:
        if not data_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {data_dir}")

        image_paths = sorted(
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in {data_dir}")

        results: dict[str, list[int]] = {}
        for image_path in image_paths:
            results[image_path.stem] = self.infer_image(image_path)
        return results

    def run_and_save(self, data_dir: Path, output_path: Path, output_time_path: Path) -> None:
        start_time = time.perf_counter()
        result_dict = self.run_directory(data_dir)
        elapsed = time.perf_counter() - start_time

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_time_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result_dict)
        output_time_path.write_text(f"{elapsed:.6f}\n", encoding="utf-8")
        print(f"Total processing time (s): {elapsed:.6f}")
