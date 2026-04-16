from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np


def get_class_name(names: dict | list, class_id: int) -> str:
    """Return a stable class display name from Ultralytics names mapping/list."""
    if isinstance(names, dict):
        return str(names.get(class_id, f"Class{class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"Class{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    """Use a short fixed palette to keep class color consistent across frames."""
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 0),
        (128, 0, 128),
    ]
    return palette[class_id % len(palette)]


def box_center_in_roi(box: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def draw_overlay(
    frame: np.ndarray,
    result: Any,
    names: dict | list,
    count_by_class: dict[str, int],
    total_count: int,
    roi: tuple[int, int, int, int],
    track_history: dict[int, deque],
    counted_display_ids: dict[int, int],
    roi_outside_alpha: float,
    filtered_boxes: np.ndarray | None = None,
    filtered_cls_ids: np.ndarray | None = None,
    filtered_confs: np.ndarray | None = None,
    filtered_track_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Render masks, boxes, track labels and count panel on the current frame.

    The caller can pass post-NMS boxes to keep visualization aligned with counting logic.
    """
    vis = frame.copy()
    rx1, ry1, rx2, ry2 = roi

    # Dim excluded border area while keeping the ROI unchanged.
    outside_alpha = max(0.0, min(float(roi_outside_alpha), 1.0))
    if outside_alpha > 0:
        dimmed = cv2.convertScaleAbs(vis, alpha=max(0.0, 1.0 - outside_alpha), beta=0)
        dimmed[ry1:ry2, rx1:rx2] = vis[ry1:ry2, rx1:rx2]
        vis = dimmed
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # Draw with post-NMS detection set when provided; fallback to raw detector output.
    if filtered_boxes is not None:
        draw_boxes = filtered_boxes
        draw_cls_ids = filtered_cls_ids
        draw_confs = filtered_confs
        draw_track_ids = filtered_track_ids

        # Keep masks consistent with filtered track IDs.
        if result.masks is not None and result.boxes is not None and result.boxes.id is not None:
            raw_track_ids = result.boxes.id.int().cpu().numpy().tolist()
            raw_cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32).tolist()
            kept_set = set(filtered_track_ids.tolist()) if filtered_track_ids is not None else set()
            kept_indices = [i for i, tid in enumerate(raw_track_ids) if tid in kept_set]
            mask_polys = [result.masks.xy[i] for i in kept_indices]
            mask_cls_ids = [int(raw_cls_ids[i]) for i in kept_indices]
        else:
            mask_polys = []
            mask_cls_ids = []
    else:
        if result.boxes is not None and len(result.boxes) > 0:
            draw_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            draw_cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
            draw_confs = result.boxes.conf.cpu().numpy()
            track_tensor = result.boxes.id
            draw_track_ids = track_tensor.int().cpu().numpy().astype(int) if track_tensor is not None else None
        else:
            draw_boxes = None
            draw_cls_ids = None
            draw_confs = None
            draw_track_ids = None

        mask_polys = result.masks.xy if result.masks is not None else []
        mask_cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32).tolist() if result.boxes is not None else []

    if len(mask_polys) > 0:
        overlay = vis.copy()
        for polygon, cls_id in zip(mask_polys, mask_cls_ids):
            poly_i = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [poly_i], class_color(int(cls_id)))
        vis = cv2.addWeighted(overlay, 0.32, vis, 0.68, 0)

    if draw_boxes is not None and len(draw_boxes) > 0:
        for idx, box in enumerate(draw_boxes):
            if not box_center_in_roi(box, roi):
                continue

            cls_id = int(draw_cls_ids[idx])
            conf = float(draw_confs[idx])
            tid = int(draw_track_ids[idx]) if draw_track_ids is not None else -1
            x1, y1, x2, y2 = box
            color = class_color(cls_id)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            class_name = get_class_name(names, cls_id)
            counted_id = counted_display_ids.get(tid, None) if tid >= 0 else None
            if counted_id is not None:
                label = f"{class_name} #{counted_id} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if tid >= 0 and tid in track_history:
                cv2.putText(
                    vis,
                    f"n={len(track_history[tid])}",
                    (x1, min(vis.shape[0] - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

    # Count summary panel in upper-left corner.
    y = 28
    cv2.rectangle(vis, (10, 10), (360, 20 + 28 * (len(count_by_class) + 2)), (0, 0, 0), -1)
    cv2.putText(vis, f"Total unique: {total_count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28
    for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
        cv2.putText(vis, f"{class_name}: {count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28

    return vis
