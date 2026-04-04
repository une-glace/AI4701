from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from screw_counting.detection import extract_candidates, load_bgr_image


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-extract screw candidate crops for manual labeling.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Input directory of corrected top-view images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for unlabeled crops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crops_dir = args.output_dir / "unlabeled"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    image_paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop_name", "source_image", "x0", "y0", "x1", "y1"])
        crop_count = 0
        for image_path in image_paths:
            image_bgr = load_bgr_image(image_path)
            candidates = extract_candidates(image_bgr)
            for idx, candidate in enumerate(candidates):
                crop_name = f"{image_path.stem}_obj_{idx:03d}.png"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), candidate.crop)
                x0, y0, x1, y1 = candidate.bbox
                writer.writerow([crop_name, image_path.name, x0, y0, x1, y1])
                crop_count += 1

    print(f"Saved {crop_count} candidate crops to {crops_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
