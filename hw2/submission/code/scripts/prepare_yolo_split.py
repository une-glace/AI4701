from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a flat YOLO export into train/val split folders.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Dataset directory containing flat images/, labels/ and classes.txt.",
    )
    parser.add_argument("--val_count", type=int, default=2, help="Number of images reserved for validation.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them into split folders.",
    )
    return parser.parse_args()


def ensure_yolo_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    names_block = ", ".join(f"'{name}'" for name in class_names)
    yaml_text = "\n".join(
        [
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"names: [{names_block}]",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    classes_path = dataset_dir / "classes.txt"

    if not images_dir.exists() or not labels_dir.exists() or not classes_path.exists():
        raise FileNotFoundError("Expected dataset/images, dataset/labels and dataset/classes.txt to exist.")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError("classes.txt is empty.")

    flat_images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(flat_images) <= args.val_count:
        raise ValueError("Validation count is too large for the number of images.")

    split_map: dict[str, list[Path]] = {
        "train": flat_images[:-args.val_count],
        "val": flat_images[-args.val_count:],
    }

    for split_name in split_map:
        (images_dir / split_name).mkdir(parents=True, exist_ok=True)
        (labels_dir / split_name).mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if args.copy else shutil.move
    for split_name, image_paths in split_map.items():
        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")

            dst_image = images_dir / split_name / image_path.name
            dst_label = labels_dir / split_name / label_path.name

            if not dst_image.exists():
                transfer(str(image_path), str(dst_image))
            if not dst_label.exists():
                transfer(str(label_path), str(dst_label))

    yaml_path = ensure_yolo_yaml(dataset_dir=dataset_dir, class_names=class_names)

    print("YOLO split prepared successfully.")
    print(f"Train images: {len(split_map['train'])}")
    print(f"Val images: {len(split_map['val'])}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
