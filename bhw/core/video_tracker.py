#!/usr/bin/env python3
"""
tracker.py (legacy entry: video_tracker.py)

Single-video screw detection, tracking and unique counting.

Example:
    python core/tracker.py --source ./test_videos/IMG_2376.MOV --weights ./weights/best.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from core.video_tracker_debug import init_map_stats, print_runtime_report, write_summary_file
    from core.video_tracker_motion import (
        detect_optflow_points,
        estimate_optflow_affine,
        estimate_phasecorr_translation,
        migrate_map_centers,
        project_center_to_map,
    )
    from core.video_tracker_visualization import draw_overlay, get_class_name
except ImportError:
    from video_tracker_debug import init_map_stats, print_runtime_report, write_summary_file
    from video_tracker_motion import (
        detect_optflow_points,
        estimate_optflow_affine,
        estimate_phasecorr_translation,
        migrate_map_centers,
        project_center_to_map,
    )
    from video_tracker_visualization import draw_overlay, get_class_name


def _prefer_local_ultralytics() -> None:
    script_dir = Path(__file__).resolve().parent
    local_pkg_dirs = [
        script_dir / "ultralytics_main",
        script_dir / "ultralytics-main",
    ]
    for local_pkg_dir in local_pkg_dirs:
        if local_pkg_dir.exists():
            local_pkg_path = str(local_pkg_dir)
            if local_pkg_path not in sys.path:
                sys.path.insert(0, local_pkg_path)
            break


_prefer_local_ultralytics()


# Keep CLI concise; advanced tuning lives in this internal defaults table.
_INTERNAL_DEFAULTS: dict[str, object] = {
    "stride_aware_conf": True,
    "stride_conf_boost": 0.03,
    "stride_aware_dedup": True,
    "fast_grab_skip": True,
    "frame_nms_iou": 0.4,
    "frame_hard_iou": 0.85,
    "roi_margin": 0.1,
    "roi_outside_alpha": 0.18,
    "min_track_frames": 3,
    "min_track_seconds": None,
    "class_history_seconds": 5.0,
    "track_buffer_seconds": 5.0,
    "recount_center_ratio": 0.035,
    "recount_iou": 0.15,
    "split_sep_iou_max": 0.28,
    "split_sep_dist_scale": 0.30,
    "split_sep_local_diag_scale": 0.35,
    "split_sep_min_dist_px": 6.0,
    "recount_hard_iou": 0.6,
    "local_max_stale_seconds": 2.0,
    "map_max_stale_seconds": 30.0,
    "map_dedup": True,
    "map_center_ratio": 0.02,
    "map_update_every": 3,
    "map_min_inlier_ratio": 0.35,
    "map_fallback_translation": True,
    "map_max_consecutive_failures": 3,
    "map_debug_summary": True,
    "ref_update_every": 30,
    "min_edge_distance": 6,
    "min_track_conf": 0.25,
    "ignore_background": True,
    "allow_cross_class_relink": False,
    "optflow_max_corners": 1000,
    "optflow_quality": 0.01,
    "optflow_min_dist": 8,
    "debug_dups": False,
}


def _apply_internal_defaults(args: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(args))
    for key, value in _INTERNAL_DEFAULTS.items():
        if not hasattr(merged, key):
            setattr(merged, key, value)
    return merged


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video screw tracking and counting")
    parser.add_argument("--source", type=str, required=True, help="input video path")
    parser.add_argument("--weights", type=str, default="weights/seg-best.pt", help="YOLO weights path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="tracker config")
    parser.add_argument("--conf", type=float, default=0.15, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="iou threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="inference image size")
    parser.add_argument("--device", type=str, default="cuda", help="device, e.g. 0 or cpu")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="save visualization every N frames (0 disables image saving)",
    )
    parser.add_argument(
        "--infer-fps",
        type=float,
        default=None,
        help="target inference FPS for frame skipping; ignored if --infer-stride is set",
    )
    parser.add_argument(
        "--infer-stride",
        type=int,
        default=None,
        help="run inference every N frames (1 means no skipping)",
    )
    parser.add_argument("--project", type=str, default="runs/video_track_count", help="output root directory")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="optional class filter")
    parser.add_argument("--save-video", action="store_true", help="also save annotated video")
    return parser.parse_args(args_list)


@dataclass(frozen=True)
class SplitThresholds:
    frame_scale: float
    center_ratio: float
    recount_iou: float
    split_iou_max: float
    split_dist_scale: float
    split_local_diag_scale: float
    split_min_dist_px: float


@dataclass
class ActiveTracks:
    ids: set[int]
    boxes: dict[int, np.ndarray]
    confs: dict[int, float]


def center_roi(frame_shape: tuple[int, int, int] | tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    margin = max(0.0, min(margin, 0.49))
    min_side = max(1, min(width, height))
    inset = int(round(min_side * margin))
    inset = min(inset, min_side // 2)
    x1 = inset
    y1 = inset
    x2 = width - inset
    y2 = height - inset

    # Keep ROI valid for extremely small images.
    if x2 <= x1:
        x1 = max(0, width // 2 - 1)
        x2 = min(width, x1 + 2)
    if y2 <= y1:
        y1 = max(0, height // 2 - 1)
        y2 = min(height, y1 + 2)
    return x1, y1, x2, y2


def box_center_in_roi(box: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def box_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def box_too_close_to_edge(box: np.ndarray, frame_w: int, frame_h: int, edge_dist: int) -> bool:
    x1, y1, x2, y2 = map(float, box)
    return x1 <= edge_dist or y1 <= edge_dist or x2 >= frame_w - edge_dist or y2 >= frame_h - edge_dist




def suppress_overlapping_detections(
    boxes: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    track_ids: np.ndarray | None,
    iou_thresh: float,
    hard_iou_any_class: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if len(boxes) == 0 or iou_thresh <= 0:
        return boxes, cls_ids, confs, track_ids

    order = np.argsort(-confs)
    keep: list[int] = []

    for idx in order:
        candidate = boxes[idx]
        suppressed = False
        for kept_idx in keep:
            iou_val = bbox_iou(candidate, boxes[kept_idx])
            same_cls = int(cls_ids[idx]) == int(cls_ids[kept_idx])
            # For class-consistent boxes use normal NMS, but still hard-suppress near-identical overlaps.
            if (same_cls and iou_val >= iou_thresh) or (iou_val >= hard_iou_any_class):
                suppressed = True
                break
        if not suppressed:
            keep.append(int(idx))

    keep_idx = np.array(keep, dtype=int)
    boxes = boxes[keep_idx]
    cls_ids = cls_ids[keep_idx]
    confs = confs[keep_idx]
    if track_ids is not None:
        track_ids = track_ids[keep_idx]
    return boxes, cls_ids, confs, track_ids


def stable_class_id(history_items) -> int:
    votes = defaultdict(float)
    for item in history_items:
        cls_id = int(item["cls_id"])
        conf = float(item.get("conf", 1.0))
        votes[cls_id] += max(1e-3, conf)
    return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]


def resolve_tracker_path(tracker_arg: str) -> Path | None:
    candidate = Path(tracker_arg)
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    local_candidates = [
        script_dir / tracker_arg,
        script_dir / "ultralytics_main" / "ultralytics" / "cfg" / "trackers" / tracker_arg,
        script_dir / "ultralytics-main" / "ultralytics" / "cfg" / "trackers" / tracker_arg,
    ]
    for path in local_candidates:
        if path.exists():
            return path

    ultra_mod = sys.modules.get("ultralytics")
    ultra_file = getattr(ultra_mod, "__file__", None)
    if ultra_file:
        pkg_candidate = Path(ultra_file).resolve().parent / "cfg" / "trackers" / tracker_arg
        if pkg_candidate.exists():
            return pkg_candidate

    return None


def build_runtime_tracker_config(
    tracker_arg: str,
    output_dir: Path,
    tracker_fps: float,
    track_buffer_seconds: float | None,
) -> tuple[str, int | None]:
    if track_buffer_seconds is None:
        return tracker_arg, None

    resolved = resolve_tracker_path(tracker_arg)
    if resolved is None:
        return tracker_arg, None

    buffer_frames = max(1, int(round(max(0.05, float(track_buffer_seconds)) * max(1.0, float(tracker_fps)))))

    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except Exception:
        return str(resolved), None

    updated = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("track_buffer:"):
            indent = line[: len(line) - len(line.lstrip())]
            comment = ""
            if "#" in line:
                comment = " #" + line.split("#", 1)[1].strip()
            updated.append(f"{indent}track_buffer: {buffer_frames}{comment}")
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        updated.append(f"track_buffer: {buffer_frames}")

    runtime_tracker = output_dir / f"{resolved.stem}.runtime.yaml"
    runtime_tracker.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return str(runtime_tracker), buffer_frames


def is_duplicate_count(
    candidate_box: np.ndarray,
    counted_instances: list[dict],
    center_ratio: float,
    recount_iou: float,
    frame_scale: float,
    candidate_cls: int | None = None,
    allow_cross_class: bool = False,
    hard_iou_any_class: float = 0.75,
    # Use frame recency to ignore stale counted boxes when camera/view has shifted too much.
    counted_last_seen: dict | None = None,
    current_frame: int = 0,
    max_stale_frames: int = 60,
    active_track_ids: set[int] | None = None,
    active_iou_gate: float = 0.45,
) -> tuple[bool, int | None]:
    center_threshold = frame_scale * center_ratio
    for idx, item in enumerate(counted_instances):
        tid = item.get("track_id")
        # Hard duplicate guard: if two boxes are almost identical, treat as same instance regardless of class.
        if bbox_iou(candidate_box, item["box"]) >= hard_iou_any_class:
            return True, idx
        item_cls = item.get("cls_id", None)
        if not allow_cross_class and candidate_cls is not None and item_cls is not None:
            if int(item_cls) != int(candidate_cls):
                continue
        # For currently visible counted tracks, use a stricter IoU gate.
        if active_track_ids is not None and tid in active_track_ids:
            if bbox_iou(candidate_box, item["box"]) >= active_iou_gate:
                return True, idx
            continue
        # Skip stale counted boxes whose cached position is likely no longer reliable.
        if counted_last_seen is not None:
            last_seen = counted_last_seen.get(tid, item.get("frame_idx", 0))
            if current_frame - last_seen > max_stale_frames:
                continue
        if box_center_distance(candidate_box, item["box"]) <= center_threshold:
            return True, idx
        if bbox_iou(candidate_box, item["box"]) >= recount_iou:
            return True, idx
    return False, None


def is_duplicate_count_map(
    candidate_center: np.ndarray,
    counted_instances: list[dict],
    center_ratio: float,
    map_scale: float,
    candidate_cls: int | None = None,
    allow_cross_class: bool = False,
    counted_last_seen: dict | None = None,
    current_frame: int = 0,
    max_stale_frames: int = 60,
    active_track_ids: set[int] | None = None,
    active_radius_scale: float = 0.35,
) -> tuple[bool, int | None]:
    radius = center_ratio * map_scale
    for idx, item in enumerate(counted_instances):
        tid = item.get("track_id")
        item_cls = item.get("cls_id", None)
        if not allow_cross_class and candidate_cls is not None and item_cls is not None:
            if int(item_cls) != int(candidate_cls):
                continue
        if counted_last_seen is not None:
            last_seen = counted_last_seen.get(tid, item.get("frame_idx", 0))
            if current_frame - last_seen > max_stale_frames:
                continue
        center = item.get("map_center", None)
        if center is None:
            continue
        dist = float(np.hypot(candidate_center[0] - center[0], candidate_center[1] - center[1]))
        if active_track_ids is not None and tid in active_track_ids:
            if dist <= radius * active_radius_scale:
                return True, idx
            continue
        if dist <= radius:
            return True, idx
    return False, None


def boxes_clearly_separated(
    box_a: np.ndarray,
    box_b: np.ndarray,
    split_cfg: SplitThresholds,
) -> bool:
    box_a_f = box_a.astype(float)
    box_b_f = box_b.astype(float)

    iou_val = bbox_iou(box_a_f, box_b_f)
    dist_val = box_center_distance(box_a_f, box_b_f)

    wa = max(1.0, float(box_a_f[2] - box_a_f[0]))
    ha = max(1.0, float(box_a_f[3] - box_a_f[1]))
    wb = max(1.0, float(box_b_f[2] - box_b_f[0]))
    hb = max(1.0, float(box_b_f[3] - box_b_f[1]))
    split_local_diag_scale = max(0.05, float(split_cfg.split_local_diag_scale))
    split_dist_scale = max(0.05, float(split_cfg.split_dist_scale))
    split_min_dist_px = max(1.0, float(split_cfg.split_min_dist_px))

    local_gate = split_local_diag_scale * min(math.hypot(wa, ha), math.hypot(wb, hb))
    global_gate = float(split_cfg.frame_scale) * max(1e-4, float(split_cfg.center_ratio)) * split_dist_scale
    dist_gate = max(split_min_dist_px, min(global_gate, local_gate))

    base_iou_gate = max(0.12, min(0.45, float(split_cfg.recount_iou) * 1.4))
    tuned_iou_gate = max(0.05, min(0.80, float(split_cfg.split_iou_max)))
    iou_gate = max(base_iou_gate, tuned_iou_gate)
    return iou_val < iou_gate and dist_val > dist_gate


def looks_like_instance_split(
    candidate_track_id: int,
    candidate_box: np.ndarray,
    matched_instance_idx: int,
    counted_instance_index: dict[int, int],
    active_tracks: ActiveTracks,
    split_cfg: SplitThresholds,
) -> bool:
    """Detect likely one-to-many split: another active track already maps to the same counted instance but is far away."""
    for tid in active_tracks.ids:
        tid_int = int(tid)
        if tid_int == candidate_track_id:
            continue
        if counted_instance_index.get(tid_int, None) != matched_instance_idx:
            continue

        anchor_box = active_tracks.boxes.get(tid_int, None)
        if anchor_box is None:
            continue

        if boxes_clearly_separated(candidate_box, anchor_box, split_cfg):
            return True

    return False


def should_split_off_counted_track(
    track_id: int,
    track_box: np.ndarray,
    mapped_instance_idx: int,
    counted_instances: list[dict],
    counted_instance_index: dict[int, int],
    active_tracks: ActiveTracks,
    split_cfg: SplitThresholds,
) -> tuple[bool, int | None]:
    """For already-counted tracks, decide whether this track should detach as a new instance after a split."""
    sibling_tids = [
        int(tid)
        for tid in active_tracks.ids
        if counted_instance_index.get(int(tid), None) == mapped_instance_idx and int(tid) in active_tracks.boxes
    ]
    if track_id not in sibling_tids and track_id in active_tracks.boxes:
        sibling_tids.append(track_id)

    if len(sibling_tids) < 2:
        return False, None

    has_clear_separation = False
    for tid in sibling_tids:
        if int(tid) == int(track_id):
            continue
        if boxes_clearly_separated(track_box, active_tracks.boxes[int(tid)], split_cfg):
            has_clear_separation = True
            break
    if not has_clear_separation:
        return False, None

    prev_box_raw = counted_instances[mapped_instance_idx].get("box", track_box.astype(float))
    prev_box = np.asarray(prev_box_raw, dtype=float)

    def keep_score(tid: int) -> tuple[float, float, float]:
        box_i = np.asarray(active_tracks.boxes[int(tid)], dtype=float)
        return (
            float(bbox_iou(box_i, prev_box)),
            float(active_tracks.confs.get(int(tid), 0.0)),
            -float(box_center_distance(box_i, prev_box)),
        )

    keep_tid = max(sibling_tids, key=keep_score)
    return int(keep_tid) != int(track_id), int(keep_tid)

def process_video(args: argparse.Namespace) -> tuple[dict[str, int], Path]:
    # Delay heavy import so `--help` works even if runtime deps are not ready.
    from ultralytics import YOLO

    args = _apply_internal_defaults(args)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    weights = Path(args.weights)
    if not weights.exists():
        # Allow using Ultralytics' built-in model names as weights, e.g. "yolo11n.pt".
        weights = Path(args.weights)

    model = YOLO(str(weights))
    if args.device:
        model.to(args.device)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    if args.infer_stride is not None:
        if args.infer_stride < 1:
            raise ValueError("--infer-stride must be >= 1")
        infer_stride = int(args.infer_stride)
        target_infer_fps = float(fps) / float(infer_stride)
    else:
        target_infer_fps = float(args.infer_fps) if args.infer_fps is not None else float(fps)
        if target_infer_fps <= 0:
            infer_stride = 1
            target_infer_fps = float(fps)
        else:
            infer_stride = max(1, int(round(fps / target_infer_fps)))
            target_infer_fps = float(fps) / float(infer_stride)

    if args.min_track_seconds is not None and args.min_track_seconds <= 0:
        raise ValueError("--min-track-seconds must be > 0")
    min_track_seconds = float(args.min_track_seconds) if args.min_track_seconds is not None else float(args.min_track_frames) / float(fps)
    min_track_seconds = max(min_track_seconds, 1.0 / float(fps))
    effective_min_track_frames = max(1, int(math.ceil(min_track_seconds * target_infer_fps)))

    if args.class_history_seconds <= 0:
        raise ValueError("--class-history-seconds must be > 0")
    class_history_seconds = max(float(args.class_history_seconds), min_track_seconds)
    history_maxlen = max(10, int(math.ceil(class_history_seconds * target_infer_fps)), effective_min_track_frames * 2)

    effective_conf = float(args.conf)
    if args.stride_aware_conf and infer_stride > 1:
        conf_boost = max(0.0, float(args.stride_conf_boost)) * math.log2(float(infer_stride))
        effective_conf = min(0.95, max(0.001, effective_conf + conf_boost))

    # Key point: sparse inference increases ID switches, so dedup gates are scaled with stride.
    dedup_stride_scale = 1.0
    if args.stride_aware_dedup and infer_stride > 1:
        dedup_stride_scale = min(2.2, math.sqrt(float(infer_stride)))
    effective_recount_center_ratio = args.recount_center_ratio * dedup_stride_scale
    effective_map_center_ratio = args.map_center_ratio * dedup_stride_scale

    project_dir = Path(args.project) / source.stem
    vis_dir = project_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if args.track_buffer_seconds is not None and args.track_buffer_seconds <= 0:
        raise ValueError("--track-buffer-seconds must be > 0")
    map_min_inlier_ratio = float(max(0.0, min(args.map_min_inlier_ratio, 1.0)))
    map_max_consecutive_failures = max(1, int(args.map_max_consecutive_failures))
    tracker_for_run, effective_track_buffer = build_runtime_tracker_config(
        args.tracker,
        project_dir,
        target_infer_fps,
        args.track_buffer_seconds,
    )

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(project_dir / f"{source.stem}_annotated.mp4"), fourcc, fps, (frame_width, frame_height))

    names = model.names
    roi = center_roi((frame_height, frame_width), args.roi_margin)
    track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_maxlen))
    counted_track_ids: set[int] = set()
    # Remember counted class by track_id so late class stabilization can correct stats.
    counted_label_map: dict[int, int] = {}
    counted_display_ids: dict[int, int] = {}
    next_counted_display_id = 1
    counted_instances: list[dict] = []
    counted_instance_index: dict[int, int] = {}
    counted_last_seen: dict[int, int] = {}
    count_by_class = defaultdict(int)
    frame_idx = 0
    inferred_frames = 0
    grabbed_skip_frames = 0
    saved_vis_count = 0
    save_every = int(args.save_every)
    if save_every < 0:
        raise ValueError("--save-every must be >= 0")
    frame_scale = float(max(frame_width, frame_height))
    max_stale_local = int(max(0.0, args.local_max_stale_seconds) * fps)
    max_stale_map = int(max(0.0, args.map_max_stale_seconds) * fps)
    split_cfg = SplitThresholds(
        frame_scale=frame_scale,
        center_ratio=effective_recount_center_ratio,
        recount_iou=float(args.recount_iou),
        split_iou_max=float(args.split_sep_iou_max),
        split_dist_scale=float(args.split_sep_dist_scale),
        split_local_diag_scale=float(args.split_sep_local_diag_scale),
        split_min_dist_px=float(args.split_sep_min_dist_px),
    )

    # Key point: map-dedup projects detections to a sliding reference frame for motion-robust dedup.
    ref_gray = None
    last_h_map = np.eye(3, dtype=np.float32)
    last_ref_update_frame = 0
    map_scale = float(max(frame_width, frame_height))
    map_motion_valid = True
    map_consecutive_failures = 0
    map_stats = init_map_stats()

    # Sparse optical-flow state used to estimate inter-frame motion.
    prev_gray_opt = None
    prev_pts = None
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    max_corners = args.optflow_max_corners
    quality_level = args.optflow_quality
    min_dist = args.optflow_min_dist

    # Progress and timing state.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    start_time = time.time()
    progress = tqdm(total=total_frames, unit="frame") if tqdm is not None and total_frames > 0 else None
    last_progress_print = 0
    empty_result = SimpleNamespace(boxes=None, masks=None)
    use_fast_grab_skip = bool(args.fast_grab_skip and writer is None and infer_stride > 1)

    while True:
        frame_idx += 1
        infer_this_frame = frame_idx == 1 or (frame_idx - 1) % infer_stride == 0
        save_this_frame = save_every > 0 and (frame_idx == 1 or frame_idx % save_every == 0)
        need_full_frame = writer is not None or infer_this_frame or save_this_frame
        vis_boxes = vis_cls_ids = vis_confs = vis_track_ids = None
        active_tracks = ActiveTracks(ids=set(), boxes={}, confs={})

        if use_fast_grab_skip and not need_full_frame:
            success = cap.grab()
            if not success:
                break
            grabbed_skip_frames += 1
            if progress is not None:
                progress.update(1)
            else:
                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    if frame_idx - last_progress_print >= max(1, total_frames // 100):
                        print(f"Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                        last_progress_print = frame_idx
                elif frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames")
            continue

        success, frame = cap.read()
        if not success:
            break

        result = empty_result
        if infer_this_frame:
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_for_run,
                conf=effective_conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=args.classes,
                verbose=False,
            )
            result = results[0]
            inferred_frames += 1

        if infer_this_frame and args.map_dedup:
            ref_update_every = getattr(args, "ref_update_every", 90)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ref_gray is None:
                ref_gray = curr_gray
                last_h_map = np.eye(3, dtype=np.float32)
                map_motion_valid = True
                map_consecutive_failures = 0
                last_ref_update_frame = frame_idx
                prev_gray_opt = curr_gray
                prev_pts = detect_optflow_points(curr_gray, max_corners, quality_level, min_dist)
            else:
                # Key point: estimate prev->curr motion, then accumulate inverse to maintain curr->ref mapping.
                h_prev_to_curr = None
                tracked_pts = None
                inlier_ratio = 0.0
                used_phase_fallback = False
                if prev_gray_opt is not None and prev_pts is not None and len(prev_pts) >= 6:
                    map_stats["motion_attempts"] += 1
                    h_prev_to_curr, tracked_pts, _, inlier_ratio = estimate_optflow_affine(prev_gray_opt, curr_gray, prev_pts, lk_params)
                    if h_prev_to_curr is not None and inlier_ratio < map_min_inlier_ratio:
                        h_prev_to_curr = None

                if h_prev_to_curr is None and args.map_fallback_translation and prev_gray_opt is not None:
                    h_prev_to_curr, phase_response = estimate_phasecorr_translation(prev_gray_opt, curr_gray)
                    used_phase_fallback = h_prev_to_curr is not None and phase_response > 0.01
                    if not used_phase_fallback:
                        h_prev_to_curr = None
                        map_stats["phase_fallback_failures"] += 1
                    else:
                        map_stats["phase_fallback_successes"] += 1

                if h_prev_to_curr is not None:
                    try:
                        h_curr_to_prev = np.linalg.inv(h_prev_to_curr)
                        last_h_map = (last_h_map @ h_curr_to_prev).astype(np.float32)
                        map_motion_valid = True
                        map_consecutive_failures = 0
                        map_stats["motion_successes"] += 1
                    except np.linalg.LinAlgError:
                        map_consecutive_failures += 1
                        map_stats["motion_failures"] += 1
                else:
                    map_consecutive_failures += 1
                    map_stats["motion_failures"] += 1

                if map_consecutive_failures >= map_max_consecutive_failures:
                    map_motion_valid = False

                redetect_interval = max(1, args.map_update_every)
                need_redetect = (
                    tracked_pts is None
                    or len(tracked_pts) < 20
                    or frame_idx % redetect_interval == 0
                )
                prev_pts = detect_optflow_points(curr_gray, max_corners, quality_level, min_dist) if need_redetect else tracked_pts
                prev_gray_opt = curr_gray

                # Periodically slide reference frame to reduce long-horizon drift.
                if frame_idx - last_ref_update_frame >= ref_update_every:
                    try:
                        h_old_to_new = np.linalg.inv(last_h_map)
                        migrate_map_centers(counted_instances, h_old_to_new)
                    except np.linalg.LinAlgError:
                        pass

                    ref_gray = curr_gray
                    last_h_map = np.eye(3, dtype=np.float32)
                    map_motion_valid = True
                    map_consecutive_failures = 0
                    last_ref_update_frame = frame_idx
                    prev_gray_opt = curr_gray
                    prev_pts = detect_optflow_points(curr_gray, max_corners, quality_level, min_dist)
        if infer_this_frame and result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
            track_ids = result.boxes.id.int().cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            boxes, cls_ids, confs, track_ids = suppress_overlapping_detections(
                boxes,
                cls_ids,
                confs,
                track_ids,
                args.frame_nms_iou,
                args.frame_hard_iou,
            )
            active_tracks = ActiveTracks(
                ids={int(tid) for tid in track_ids.tolist()} if track_ids is not None else set(),
                boxes={
                    int(tid): boxes[idx].astype(float)
                    for idx, tid in enumerate(track_ids.tolist())
                } if track_ids is not None else {},
                confs={
                    int(tid): float(confs[idx])
                    for idx, tid in enumerate(track_ids.tolist())
                } if track_ids is not None else {},
            )
            vis_boxes, vis_cls_ids, vis_confs, vis_track_ids = boxes, cls_ids, confs, track_ids

            # Key point: clear histories of suppressed tracks to avoid stale-history recounts.
            all_track_ids_this_frame = set(result.boxes.id.int().cpu().numpy().tolist())
            suppressed_track_ids = all_track_ids_this_frame - active_tracks.ids
            for suppressed_tid in suppressed_track_ids:
                if int(suppressed_tid) not in counted_track_ids:
                    track_history[int(suppressed_tid)].clear()

            for cls_id, track_id, box, conf in zip(cls_ids, track_ids, boxes, confs):
                track_id_int = int(track_id)
                if not box_center_in_roi(box, roi):
                    continue
                if box_too_close_to_edge(box, frame_width, frame_height, args.min_edge_distance):
                    continue

                history = track_history[track_id_int]
                history.append({"cls_id": int(cls_id), "box": box.astype(float), "conf": float(conf)})

                if len(history) < effective_min_track_frames:
                    continue

                mean_conf = float(np.mean([item["conf"] for item in history]))
                if mean_conf < args.min_track_conf:
                    continue

                stable_cls = stable_class_id(history)
                class_name = get_class_name(names, stable_cls)
                if args.ignore_background and class_name.lower() in {"__background__", "background"}:
                    continue

                force_new_from_split = False
                # Already-counted tracks usually refresh existing instance state instead of recounting.
                if track_id_int in counted_track_ids:
                    instance_idx = counted_instance_index.get(track_id_int, None)
                    if instance_idx is not None:
                        force_new_from_split, keep_tid = should_split_off_counted_track(
                            track_id=track_id_int,
                            track_box=box.astype(float),
                            mapped_instance_idx=int(instance_idx),
                            counted_instances=counted_instances,
                            counted_instance_index=counted_instance_index,
                            active_tracks=active_tracks,
                            split_cfg=split_cfg,
                        )
                        if force_new_from_split:
                            counted_instance_index.pop(track_id_int, None)
                            counted_display_ids.pop(track_id_int, None)
                            if keep_tid is not None:
                                counted_instance_index[int(keep_tid)] = int(instance_idx)
                                counted_instances[int(instance_idx)]["track_id"] = int(keep_tid)
                                counted_last_seen[int(keep_tid)] = frame_idx
                            if args.debug_dups:
                                print(
                                    f"[split-detach] frame={frame_idx} track={track_id_int} "
                                    f"from_instance={instance_idx} keep_tid={keep_tid}"
                                )

                    if not force_new_from_split:
                        # Keep normal refresh path for genuinely same instance.
                        counted_last_seen[track_id_int] = frame_idx
                        if instance_idx is not None:
                            it = counted_instances[instance_idx]
                            it["box"] = box.astype(float)
                            it["frame_idx"] = frame_idx
                            if args.map_dedup and map_motion_valid:
                                mc = project_center_to_map(box.astype(float), last_h_map)
                                if mc is not None:
                                    it["map_center"] = mc
                        prev_cls = counted_label_map.get(track_id_int, None)
                        if prev_cls is not None and int(prev_cls) != int(stable_cls):
                            old_name = get_class_name(names, int(prev_cls))
                            new_name = class_name
                            count_by_class[old_name] = max(0, count_by_class.get(old_name, 0) - 1)
                            count_by_class[new_name] += 1
                            counted_label_map[track_id_int] = int(stable_cls)
                            if instance_idx is not None:
                                counted_instances[instance_idx]["cls_id"] = int(stable_cls)
                        continue

                map_center = None
                if args.map_dedup:
                    if map_motion_valid:
                        map_center = project_center_to_map(box.astype(float), last_h_map)
                    else:
                        map_stats["disabled_frames"] += 1
                    if map_center is None:
                        map_stats["center_unavailable"] += 1
                dup_local = False
                dup_local_idx = None
                dup_map = False
                dup_map_idx = None
                if not force_new_from_split:
                    dup_local, dup_local_idx = is_duplicate_count(
                        box, counted_instances,
                        effective_recount_center_ratio, args.recount_iou, frame_scale,
                        candidate_cls=int(stable_cls),
                        allow_cross_class=args.allow_cross_class_relink,
                        hard_iou_any_class=args.recount_hard_iou,
                        counted_last_seen=counted_last_seen,
                        current_frame=frame_idx,
                        max_stale_frames=max_stale_local,
                        active_track_ids=active_tracks.ids,
                    )

                if not force_new_from_split and args.map_dedup and map_center is not None:
                    map_stats["dedup_checks"] += 1
                    dup_map, dup_map_idx = is_duplicate_count_map(
                        map_center,
                        counted_instances,
                        effective_map_center_ratio,
                        map_scale,
                        candidate_cls=int(stable_cls),
                        allow_cross_class=args.allow_cross_class_relink,
                        counted_last_seen=counted_last_seen,
                        current_frame=frame_idx,
                        max_stale_frames=max_stale_map,
                        active_track_ids=active_tracks.ids,
                    )
                    if dup_map:
                        map_stats["dedup_hits"] += 1
                if not force_new_from_split and (dup_local or dup_map):
                    matched_idx = dup_local_idx if dup_local_idx is not None else dup_map_idx
                    split_force_new_count = False
                    if matched_idx is not None:
                        # If another active track already occupies this counted instance and is clearly separated,
                        # treat this candidate as a newly split instance instead of re-linking.
                        split_force_new_count = looks_like_instance_split(
                            candidate_track_id=track_id_int,
                            candidate_box=box,
                            matched_instance_idx=int(matched_idx),
                            counted_instance_index=counted_instance_index,
                            active_tracks=active_tracks,
                            split_cfg=split_cfg,
                        )

                    if matched_idx is not None and not split_force_new_count:
                        matched_item = counted_instances[matched_idx]
                        matched_tid = int(matched_item.get("track_id", -1))
                        counted_track_ids.add(track_id_int)
                        counted_instance_index[track_id_int] = matched_idx
                        counted_last_seen[track_id_int] = frame_idx
                        counted_label_map[track_id_int] = int(matched_item.get("cls_id", stable_cls))
                        if matched_tid in counted_display_ids:
                            candidate_display_id = counted_display_ids[matched_tid]
                            # Avoid showing the same display ID on two different active instances in one frame.
                            has_conflict = False
                            for active_tid in active_tracks.ids:
                                if int(active_tid) == track_id_int:
                                    continue
                                if counted_display_ids.get(int(active_tid), None) != candidate_display_id:
                                    continue
                                if counted_instance_index.get(int(active_tid), None) != matched_idx:
                                    has_conflict = True
                                    break
                            if not has_conflict:
                                counted_display_ids[track_id_int] = candidate_display_id
                        # Keep owner stable when it's still active; only transfer ownership if old owner is gone.
                        adopt_new_owner = matched_tid < 0 or matched_tid not in active_tracks.ids
                        if adopt_new_owner:
                            matched_item["track_id"] = track_id_int
                            matched_item["box"] = box.astype(float)
                        else:
                            owner_box = active_tracks.boxes.get(matched_tid, box.astype(float))
                            matched_item["box"] = owner_box.astype(float)
                        matched_item["frame_idx"] = frame_idx
                        if args.map_dedup:
                            if adopt_new_owner and map_center is not None:
                                matched_item["map_center"] = map_center
                            elif (not adopt_new_owner) and map_motion_valid:
                                owner_box = active_tracks.boxes.get(matched_tid, None)
                                if owner_box is not None:
                                    owner_map_center = project_center_to_map(owner_box.astype(float), last_h_map)
                                    if owner_map_center is not None:
                                        matched_item["map_center"] = owner_map_center
                    if not split_force_new_count:
                        if args.debug_dups:
                            reason = "local" if dup_local else "map"
                            action = "relink" if matched_idx is not None else "skip"
                            print(f"[dup-{action}] frame={frame_idx} track={track_id} cls={stable_cls} reason={reason}")
                        continue
                    if args.debug_dups:
                        reason = "local" if dup_local else "map"
                        print(
                            f"[dup-split-new] frame={frame_idx} track={track_id} cls={stable_cls} "
                            f"reason={reason} matched_idx={matched_idx}"
                        )

                # Passed all gates: count as a new unique instance.
                item = {
                    "track_id": track_id_int,
                    "cls_id": int(stable_cls),
                    "box": box.astype(float),
                    "map_center": map_center,
                    "frame_idx": frame_idx,
                }
                counted_instances.append(item)
                counted_instance_index[track_id_int] = len(counted_instances) - 1
                if track_id_int not in counted_display_ids:
                    counted_display_ids[track_id_int] = next_counted_display_id
                    next_counted_display_id += 1
                count_by_class[class_name] += 1
                counted_track_ids.add(track_id_int)
                counted_label_map[track_id_int] = int(stable_cls)
                counted_last_seen[track_id_int] = frame_idx
                if args.debug_dups:
                    print(f"[count] frame={frame_idx} track={track_id} cls={stable_cls} box={box.tolist()} map_center={None if map_center is None else [float(map_center[0]), float(map_center[1])]}")
        need_vis = writer is not None or save_this_frame
        if need_vis:
            total_unique = len(counted_instances)
            vis = draw_overlay(
                frame,
                result,
                names,
                count_by_class,
                total_unique,
                roi,
                track_history,
                counted_display_ids,
                args.roi_outside_alpha,
                filtered_boxes=vis_boxes,
                filtered_cls_ids=vis_cls_ids,
                filtered_confs=vis_confs,
                filtered_track_ids=vis_track_ids,
            )

            if writer is not None:
                writer.write(vis)

            if save_this_frame:
                out_path = vis_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), vis)
                saved_vis_count += 1

        # update progress display
        if progress is not None:
            progress.update(1)
        else:
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                # print periodically to avoid flooding
                if frame_idx - last_progress_print >= max(1, total_frames // 100):
                    print(f"Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                    last_progress_print = frame_idx
            elif frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")

    cap.release()
    if writer is not None:
        writer.release()

    if progress is not None:
        progress.close()

    summary_path = project_dir / "summary.txt"
    elapsed = time.time() - start_time
    annotated_video_path = project_dir / f"{source.stem}_annotated.mp4" if writer is not None else None

    write_summary_file(
        summary_path=summary_path,
        source=source,
        weights=weights,
        frames=frame_idx,
        unique_tracks=len(counted_instances),
        count_by_class=count_by_class,
        map_debug_summary=args.map_debug_summary,
        map_stats=map_stats,
    )

    print_runtime_report(
        frame_idx=frame_idx,
        inferred_frames=inferred_frames,
        target_infer_fps=target_infer_fps,
        infer_stride=infer_stride,
        effective_conf=effective_conf,
        base_conf=float(args.conf),
        stride_aware_conf=bool(args.stride_aware_conf),
        effective_min_track_frames=effective_min_track_frames,
        min_track_seconds=min_track_seconds,
        history_maxlen=history_maxlen,
        dedup_stride_scale=dedup_stride_scale,
        effective_recount_center_ratio=effective_recount_center_ratio,
        effective_map_center_ratio=effective_map_center_ratio,
        split_sep_iou_max=float(args.split_sep_iou_max),
        split_sep_dist_scale=float(args.split_sep_dist_scale),
        split_sep_local_diag_scale=float(args.split_sep_local_diag_scale),
        split_sep_min_dist_px=float(args.split_sep_min_dist_px),
        map_debug_summary=bool(args.map_debug_summary),
        map_stats=map_stats,
        effective_track_buffer=effective_track_buffer,
        track_buffer_seconds=args.track_buffer_seconds,
        grabbed_skip_frames=grabbed_skip_frames,
        use_fast_grab_skip=use_fast_grab_skip,
        unique_tracks=len(counted_instances),
        count_by_class=count_by_class,
        saved_vis_count=saved_vis_count,
        summary_path=summary_path,
        annotated_video_path=annotated_video_path,
        elapsed=elapsed,
    )

    return dict(count_by_class), vis_dir


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
