from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    area: float
    crop: np.ndarray


def load_bgr_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def preprocess_for_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    block_size = 51 if min(image_bgr.shape[:2]) > 512 else 31
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        6,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, binary


def split_touching_instances(binary_mask: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    separated = np.zeros_like(binary_mask)
    separated[markers > 1] = 255
    return separated


def extract_candidates(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.2,
    padding_ratio: float = 0.1,
) -> list[Candidate]:
    height, width = image_bgr.shape[:2]
    image_area = float(height * width)

    _, binary = preprocess_for_mask(image_bgr)
    separated = split_touching_instances(binary)
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Candidate] = []
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        pad_x = int(round(w * padding_ratio))
        pad_y = int(round(h * padding_ratio))
        x0 = max(x - pad_x, 0)
        y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, width)
        y1 = min(y + h + pad_y, height)
        crop = image_bgr[y0:y1, x0:x1].copy()
        candidates.append(
            Candidate(
                bbox=(x0, y0, x1, y1),
                contour=contour,
                area=float(area),
                crop=crop,
            )
        )

    candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return candidates


def draw_candidates(image_bgr: np.ndarray, candidates: Iterable[Candidate]) -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, candidate in enumerate(candidates, start=1):
        x0, y0, x1, y1 = candidate.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(idx),
            (x0, max(y0 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas
