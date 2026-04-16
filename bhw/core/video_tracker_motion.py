from __future__ import annotations

import cv2
import numpy as np


def detect_optflow_points(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_dist: int,
) -> np.ndarray | None:
    """Detect sparse corners used by LK optical flow."""
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_dist,
        blockSize=7,
    )


def project_center_to_map(box: np.ndarray, h_mat: np.ndarray | None) -> np.ndarray | None:
    """Project current bbox center to reference-map coordinates."""
    if h_mat is None:
        return None
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, h_mat)
    return proj.reshape(2).astype(float)


def migrate_map_centers(
    counted_instances: list[dict],
    h_old_to_new: np.ndarray,
) -> None:
    """Move all recorded map centers to a new reference frame when ref frame slides."""
    for item in counted_instances:
        old_center = item.get("map_center")
        if old_center is None:
            continue
        pt = np.array([[[float(old_center[0]), float(old_center[1])]]], dtype=np.float32)
        try:
            new_center = cv2.perspectiveTransform(pt, h_old_to_new)
            item["map_center"] = new_center.reshape(2).astype(float)
        except Exception:
            # If migration fails, drop this center to avoid incorrect future dedup.
            item["map_center"] = None


def estimate_optflow_affine(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray, lk_params: dict):
    """Estimate robust affine motion between adjacent frames with LK flow + RANSAC."""
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None, 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if next_pts is None or status is None:
        return None, None, None, 0.0

    good_prev = prev_pts[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]
    if len(good_prev) < 6 or len(good_next) < 6:
        return None, good_next.reshape(-1, 1, 2) if good_next is not None else None, status, 0.0

    mat_2x3, inliers = cv2.estimateAffinePartial2D(
        good_prev.reshape(-1, 1, 2),
        good_next.reshape(-1, 1, 2),
        method=cv2.RANSAC,
    )
    if mat_2x3 is None:
        return None, good_next.reshape(-1, 1, 2), status, 0.0

    inlier_ratio = 0.0
    if inliers is not None and len(inliers) > 0:
        inlier_ratio = float(np.mean(inliers.astype(np.float32)))

    h_mat = np.eye(3, dtype=np.float32)
    h_mat[:2, :3] = mat_2x3
    return h_mat, good_next.reshape(-1, 1, 2), status, inlier_ratio


def estimate_phasecorr_translation(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Estimate pure translation as a fallback when affine motion fails."""
    prev32 = prev_gray.astype(np.float32)
    curr32 = curr_gray.astype(np.float32)
    try:
        shift, response = cv2.phaseCorrelate(prev32, curr32)
    except Exception:
        return None, 0.0

    if not np.isfinite(shift[0]) or not np.isfinite(shift[1]):
        return None, float(response)

    h_mat = np.eye(3, dtype=np.float32)
    h_mat[0, 2] = float(shift[0])
    h_mat[1, 2] = float(shift[1])
    return h_mat, float(response)
