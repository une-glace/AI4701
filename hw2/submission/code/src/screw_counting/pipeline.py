from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .classifier import TorchScrewClassifier
from .detection import draw_candidates, extract_candidates, load_bgr_image


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ScrewCountingPipeline:
    def __init__(self, checkpoint_path: Path, device: str = "auto", debug_dir: Path | None = None) -> None:
        self.classifier = TorchScrewClassifier(checkpoint_path=checkpoint_path, device=device)
        self.class_names = self.classifier.class_names
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def infer_image(self, image_path: Path) -> list[int]:
        image_bgr = load_bgr_image(image_path)
        candidates = extract_candidates(image_bgr)
        counts = np.zeros(len(self.class_names), dtype=np.int32)

        predictions = []
        for candidate in candidates:
            pred = self.classifier.predict_with_tta(candidate.crop)
            counts[pred.label_index] += 1
            predictions.append((candidate, pred))

        if self.debug_dir is not None:
            vis = draw_candidates(image_bgr, [item[0] for item in predictions])
            for candidate, pred in predictions:
                x0, y0, _, _ = candidate.bbox
                cv2.putText(
                    vis,
                    f"{pred.label_name}:{pred.confidence:.2f}",
                    (x0, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(self.debug_dir / f"{image_path.stem}_debug.png"), vis)

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
