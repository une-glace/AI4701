#!/usr/bin/env python3
"""
video_track_count.py

逐帧检测 + 跟踪 + 计数，并按固定间隔保存可视化叠加图。

示例：
    python video_track_count.py --source ../IMG_2376.MOV --weights weights/seg-best.pt --save-every 10

输出：
    runs/video_track_count/<video_stem>/vis/frame_000010.jpg
    runs/video_track_count/<video_stem>/summary.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
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


def _prefer_local_ultralytics() -> None:
    script_dir = Path(__file__).resolve().parent
    local_pkg_dir = script_dir / "ultralytics-main"
    if local_pkg_dir.exists():
        local_pkg_path = str(local_pkg_dir)
        if local_pkg_path not in sys.path:
            sys.path.insert(0, local_pkg_path)


_prefer_local_ultralytics()


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video instance tracking and counting")
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
    parser.add_argument(
        "--stride-aware-conf",
        action="store_true",
        default=True,
        help="auto-raise detector confidence when infer stride > 1",
    )
    parser.add_argument("--no-stride-aware-conf", action="store_false", dest="stride_aware_conf")
    parser.add_argument(
        "--stride-conf-boost",
        type=float,
        default=0.03,
        help="confidence boost coefficient applied as boost*log2(infer_stride)",
    )
    parser.add_argument(
        "--stride-aware-dedup",
        action="store_true",
        default=True,
        help="auto-expand dedup radius when infer stride > 1 to reduce recount after track ID switches",
    )
    parser.add_argument("--no-stride-aware-dedup", action="store_false", dest="stride_aware_dedup")
    parser.add_argument(
        "--fast-grab-skip",
        action="store_true",
        default=True,
        help="use VideoCapture.grab() to skip non-inference/non-save frames for faster decoding",
    )
    parser.add_argument("--no-fast-grab-skip", action="store_false", dest="fast_grab_skip")
    parser.add_argument(
        "--frame-nms-iou",
        type=float,
        default=0.4,
        help="suppress overlapping detections in the same frame using class-agnostic NMS",
    )
    parser.add_argument(
        "--frame-hard-iou",
        type=float,
        default=0.85,
        help="hard suppression IoU for near-identical boxes even across classes",
    )
    parser.add_argument(
        "--roi-margin",
        type=float,
        default=0.1,
        help="uniform border inset ratio based on the shorter frame side (e.g. 0.20 means inset=20%% of min(H,W))",
    )
    parser.add_argument(
        "--roi-outside-alpha",
        type=float,
        default=0.18,
        help="darkening strength for area outside ROI (0=no darkening, 1=full black)",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="base minimum observations at source FPS before counting",
    )
    parser.add_argument(
        "--min-track-seconds",
        type=float,
        default=None,
        help="minimum observed time before counting; default derives from min-track-frames/source_fps",
    )
    parser.add_argument(
        "--class-history-seconds",
        type=float,
        default=5.0,
        help="history window in seconds for class voting",
    )
    parser.add_argument(
        "--track-buffer-seconds",
        type=float,
        default=5.0,
        help="keep lost tracks for this many seconds by rewriting tracker track_buffer",
    )
    parser.add_argument(
        "--recount-center-ratio",
        type=float,
        default=0.035,
        help="suppression radius for already-counted instances, relative to max(frame_w, frame_h)",
    )
    parser.add_argument(
        "--recount-iou",
        type=float,
        default=0.15,
        help="extra IoU check when suppressing repeated counts",
    )
    parser.add_argument(
        "--split-sep-iou-max",
        type=float,
        default=0.28,
        help="max IoU still considered as two separated instances during split handling (larger is more permissive)",
    )
    parser.add_argument(
        "--split-sep-dist-scale",
        type=float,
        default=0.30,
        help="distance gate scale for split handling, relative to recount center radius (smaller is more permissive)",
    )
    parser.add_argument(
        "--split-sep-local-diag-scale",
        type=float,
        default=0.35,
        help="distance gate scale for split handling, relative to the smaller bbox diagonal (smaller is more permissive)",
    )
    parser.add_argument(
        "--split-sep-min-dist-px",
        type=float,
        default=6.0,
        help="minimum center distance in pixels to consider split (smaller is more permissive)",
    )
    parser.add_argument(
        "--recount-hard-iou",
        type=float,
        default=0.6,
        help="hard duplicate IoU for near-identical instances even across classes",
    )
    parser.add_argument(
        "--local-max-stale-seconds",
        type=float,
        default=2.0,
        help="local dedup only considers counted instances seen within this many seconds",
    )
    parser.add_argument(
        "--map-max-stale-seconds",
        type=float,
        default=30.0,
        help="map dedup keeps counted instances for this many seconds to handle re-entry",
    )
    parser.add_argument(
        "--map-dedup",
        action="store_true",
        default=True,
        help="deduplicate by mapping detections into a reference frame (recommended for complex camera motion)",
    )
    parser.add_argument("--no-map-dedup", action="store_false", dest="map_dedup")
    parser.add_argument(
        "--map-center-ratio",
        type=float,
        default=0.02,
        help="dedup radius in reference map coordinates, relative to max(ref_w, ref_h)",
    )
    parser.add_argument(
        "--map-update-every",
        type=int,
        default=3,
        help="refresh optical-flow keypoints every N frames",
    )
    parser.add_argument(
        "--map-min-inlier-ratio",
        type=float,
        default=0.35,
        help="minimum RANSAC inlier ratio to accept optical-flow affine update",
    )
    parser.add_argument(
        "--map-fallback-translation",
        action="store_true",
        default=True,
        help="fallback to phase-correlation translation when optical-flow affine fails",
    )
    parser.add_argument("--no-map-fallback-translation", action="store_false", dest="map_fallback_translation")
    parser.add_argument(
        "--map-max-consecutive-failures",
        type=int,
        default=3,
        help="temporarily disable map projection after this many consecutive motion-estimation failures",
    )
    parser.add_argument(
        "--map-debug-summary",
        action="store_true",
        default=True,
        help="print map-dedup and motion-estimation summary at the end",
    )
    parser.add_argument("--no-map-debug-summary", action="store_false", dest="map_debug_summary")
    parser.add_argument(
    "--ref-update-every",
    type=int,
    default=30,
    help="每隔 N 帧把当前帧设为新参考帧（建议设为 fps*3 左右）",
    )
    parser.add_argument(
        "--min-edge-distance",
        type=int,
        default=6,
        help="skip detections too close to image border (pixels)",
    )
    parser.add_argument(
        "--min-track-conf",
        type=float,
        default=0.25,
        help="minimum mean confidence over track history before counting",
    )
    parser.add_argument("--project", type=str, default="runs/video_track_count", help="output root directory")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="optional class filter")
    parser.add_argument("--ignore-background", action="store_true", default=True, help="skip class named background")
    parser.add_argument("--no-ignore-background", action="store_false", dest="ignore_background")
    parser.add_argument("--save-video", action="store_true", help="also save annotated video")
    parser.add_argument(
        "--allow-cross-class-relink",
        action="store_true",
        help="allow relinking to a counted instance even when class differs (disabled by default)",
    )
    # appearance embedding removed; rely on map/local dedup
    parser.add_argument("--optflow-max-corners", type=int, default=1000, help="max corners for goodFeaturesToTrack")
    parser.add_argument("--optflow-quality", type=float, default=0.01, help="qualityLevel for goodFeaturesToTrack")
    parser.add_argument("--optflow-min-dist", type=int, default=8, help="minDistance for goodFeaturesToTrack")
    parser.add_argument("--debug-dups", action="store_true", help="print duplicate-check debug info")
    return parser.parse_args(args_list)


def get_class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"Class{class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"Class{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
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


def detect_optflow_points(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_dist: int,
) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_dist,
        blockSize=7,
    )


def project_center_to_map(box: np.ndarray, h_mat: np.ndarray | None) -> np.ndarray | None:
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
    """把所有已计数实例的 map_center 从旧参考坐标系迁移到新参考坐标系"""
    for item in counted_instances:
        old_center = item.get("map_center")
        if old_center is None:
            continue
        pt = np.array([[[float(old_center[0]), float(old_center[1])]]], dtype=np.float32)
        try:
            new_center = cv2.perspectiveTransform(pt, h_old_to_new)
            item["map_center"] = new_center.reshape(2).astype(float)
        except Exception:
            # 迁移失败则清空，宁可漏掉去重也不用错误坐标
            item["map_center"] = None




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
    counted_last_seen: dict | None = None,   # ← 新增参数
    current_frame: int = 0,                  # ← 新增参数
    max_stale_frames: int = 60,              # ← 新增参数
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
        # ← 新增：跳过坐标已经过时的实例（出镜太久，坐标不可信）
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
    frame_scale: float,
    center_ratio: float,
    recount_iou: float,
    split_iou_max: float,
    split_dist_scale: float,
    split_local_diag_scale: float,
    split_min_dist_px: float,
) -> bool:
    box_a_f = box_a.astype(float)
    box_b_f = box_b.astype(float)

    iou_val = bbox_iou(box_a_f, box_b_f)
    dist_val = box_center_distance(box_a_f, box_b_f)

    wa = max(1.0, float(box_a_f[2] - box_a_f[0]))
    ha = max(1.0, float(box_a_f[3] - box_a_f[1]))
    wb = max(1.0, float(box_b_f[2] - box_b_f[0]))
    hb = max(1.0, float(box_b_f[3] - box_b_f[1]))
    split_local_diag_scale = max(0.05, float(split_local_diag_scale))
    split_dist_scale = max(0.05, float(split_dist_scale))
    split_min_dist_px = max(1.0, float(split_min_dist_px))

    local_gate = split_local_diag_scale * min(math.hypot(wa, ha), math.hypot(wb, hb))
    global_gate = frame_scale * max(1e-4, float(center_ratio)) * split_dist_scale
    dist_gate = max(split_min_dist_px, min(global_gate, local_gate))

    base_iou_gate = max(0.12, min(0.45, float(recount_iou) * 1.4))
    tuned_iou_gate = max(0.05, min(0.80, float(split_iou_max)))
    iou_gate = max(base_iou_gate, tuned_iou_gate)
    return iou_val < iou_gate and dist_val > dist_gate


def looks_like_instance_split(
    candidate_track_id: int,
    candidate_box: np.ndarray,
    matched_instance_idx: int,
    counted_instance_index: dict[int, int],
    active_track_ids: set[int],
    active_track_boxes: dict[int, np.ndarray],
    frame_scale: float,
    center_ratio: float,
    recount_iou: float,
    split_iou_max: float,
    split_dist_scale: float,
    split_local_diag_scale: float,
    split_min_dist_px: float,
) -> bool:
    """Detect likely one-to-many split: another active track already maps to the same counted instance but is far away."""
    for tid in active_track_ids:
        tid_int = int(tid)
        if tid_int == candidate_track_id:
            continue
        if counted_instance_index.get(tid_int, None) != matched_instance_idx:
            continue

        anchor_box = active_track_boxes.get(tid_int, None)
        if anchor_box is None:
            continue

        if boxes_clearly_separated(
            candidate_box,
            anchor_box,
            frame_scale,
            center_ratio,
            recount_iou,
            split_iou_max,
            split_dist_scale,
            split_local_diag_scale,
            split_min_dist_px,
        ):
            return True

    return False


def should_split_off_counted_track(
    track_id: int,
    track_box: np.ndarray,
    mapped_instance_idx: int,
    counted_instances: list[dict],
    counted_instance_index: dict[int, int],
    active_track_ids: set[int],
    active_track_boxes: dict[int, np.ndarray],
    active_track_confs: dict[int, float],
    frame_scale: float,
    center_ratio: float,
    recount_iou: float,
    split_iou_max: float,
    split_dist_scale: float,
    split_local_diag_scale: float,
    split_min_dist_px: float,
) -> tuple[bool, int | None]:
    """For already-counted tracks, decide whether this track should detach as a new instance after a split."""
    sibling_tids = [
        int(tid)
        for tid in active_track_ids
        if counted_instance_index.get(int(tid), None) == mapped_instance_idx and int(tid) in active_track_boxes
    ]
    if track_id not in sibling_tids and track_id in active_track_boxes:
        sibling_tids.append(track_id)

    if len(sibling_tids) < 2:
        return False, None

    has_clear_separation = False
    for tid in sibling_tids:
        if int(tid) == int(track_id):
            continue
        if boxes_clearly_separated(
            track_box,
            active_track_boxes[int(tid)],
            frame_scale,
            center_ratio,
            recount_iou,
            split_iou_max,
            split_dist_scale,
            split_local_diag_scale,
            split_min_dist_px,
        ):
            has_clear_separation = True
            break
    if not has_clear_separation:
        return False, None

    prev_box_raw = counted_instances[mapped_instance_idx].get("box", track_box.astype(float))
    prev_box = np.asarray(prev_box_raw, dtype=float)

    def keep_score(tid: int) -> tuple[float, float, float]:
        box_i = np.asarray(active_track_boxes[int(tid)], dtype=float)
        return (
            float(bbox_iou(box_i, prev_box)),
            float(active_track_confs.get(int(tid), 0.0)),
            -float(box_center_distance(box_i, prev_box)),
        )

    keep_tid = max(sibling_tids, key=keep_score)
    return int(keep_tid) != int(track_id), int(keep_tid)


def get_box_center(box: np.ndarray) -> tuple[float, float]:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return cx, cy


def estimate_optflow_affine(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray, lk_params: dict):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None, 0.0

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if next_pts is None or status is None:
        return None, None, None, 0.0

    # keep good points
    good_prev = prev_pts[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]
    if len(good_prev) < 6 or len(good_next) < 6:
        return None, good_next.reshape(-1, 1, 2) if good_next is not None else None, status, 0.0

    # estimate affine transform (robust)
    M, inliers = cv2.estimateAffinePartial2D(good_prev.reshape(-1, 1, 2), good_next.reshape(-1, 1, 2), method=cv2.RANSAC)
    if M is None:
        return None, good_next.reshape(-1, 1, 2), status, 0.0

    inlier_ratio = 0.0
    if inliers is not None and len(inliers) > 0:
        inlier_ratio = float(np.mean(inliers.astype(np.float32)))

    # convert to 3x3
    H = np.eye(3, dtype=np.float32)
    H[:2, :3] = M
    return H, good_next.reshape(-1, 1, 2), status, inlier_ratio


def estimate_phasecorr_translation(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray | None, float]:
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


def draw_overlay(
    frame: np.ndarray,
    result,
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
    vis = frame.copy()
    rx1, ry1, rx2, ry2 = roi

    # Gently dim the excluded border area while keeping the ROI unchanged.
    outside_alpha = max(0.0, min(float(roi_outside_alpha), 1.0))
    if outside_alpha > 0:
        dimmed = cv2.convertScaleAbs(vis, alpha=max(0.0, 1.0 - outside_alpha), beta=0)
        dimmed[ry1:ry2, rx1:rx2] = vis[ry1:ry2, rx1:rx2]
        vis = dimmed
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # 确定最终用于绘制的数据：优先用 NMS 后传入的，否则退回原始 result
    if filtered_boxes is not None:
        draw_boxes = filtered_boxes
        draw_cls_ids = filtered_cls_ids
        draw_confs = filtered_confs
        draw_track_ids = filtered_track_ids
        # mask 也按 NMS 保留的 track_id 过滤
        if result.masks is not None and result.boxes is not None and result.boxes.id is not None:
            raw_track_ids = result.boxes.id.int().cpu().numpy().tolist()
            raw_cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32).tolist()
            kept_set = set(filtered_track_ids.tolist()) if filtered_track_ids is not None else set()
            kept_indices = [i for i, tid in enumerate(raw_track_ids) if tid in kept_set]
            mask_polys = [result.masks.xy[i] for i in kept_indices]
            # Keep mask class IDs aligned with the original mask indices.
            mask_cls_ids = [int(raw_cls_ids[i]) for i in kept_indices]
        else:
            mask_polys = []
            mask_cls_ids = []
    else:
        if result.boxes is not None and len(result.boxes) > 0:
            draw_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            draw_cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
            draw_confs = result.boxes.conf.cpu().numpy()
            t = result.boxes.id
            draw_track_ids = t.int().cpu().numpy().astype(int) if t is not None else None
        else:
            draw_boxes = None
            draw_cls_ids = None
            draw_confs = None
            draw_track_ids = None
        mask_polys = result.masks.xy if result.masks is not None else []
        mask_cls_ids = (
            result.boxes.cls.cpu().numpy().astype(np.int32).tolist()
            if result.boxes is not None else []
        )

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

    # Count summary in the upper-left corner.
    y = 28
    cv2.rectangle(vis, (10, 10), (360, 20 + 28 * (len(count_by_class) + 2)), (0, 0, 0), -1)
    cv2.putText(vis, f"Total unique: {total_count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28
    for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
        cv2.putText(vis, f"{class_name}: {count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28

    return vis

def process_video(args: argparse.Namespace) -> tuple[dict[str, int], Path]:
    # Delay heavy import so `--help` works even if runtime deps are not ready.
    from ultralytics import YOLO

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

    # With sparse inference, tracker IDs are more likely to switch; widen dedup gates moderately.
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
    # map from track_id -> last recorded cls_id for counted tracks (allows corrections)
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

    # Reference-map dedup / motion estimation state for complex camera motion.
    ref_gray = None
    last_h_map = np.eye(3, dtype=np.float32)
    last_ref_update_frame = 0
    map_scale = float(max(frame_width, frame_height))
    map_motion_valid = True
    map_consecutive_failures = 0
    map_stats = {
        "motion_attempts": 0,
        "motion_successes": 0,
        "motion_failures": 0,
        "phase_fallback_successes": 0,
        "phase_fallback_failures": 0,
        "disabled_frames": 0,
        "center_unavailable": 0,
        "dedup_checks": 0,
        "dedup_hits": 0,
    }

    # sparse optical flow state
    prev_gray_opt = None
    prev_pts = None
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    max_corners = args.optflow_max_corners
    quality_level = args.optflow_quality
    min_dist = args.optflow_min_dist

    # no appearance model (ReID) in this configuration

    # progress / timing setup
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
                # 通过相邻帧光流估计 prev->curr 变换，再累计得到 curr->ref 变换。
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

                # 每隔 ref_update_every 帧，滑动更新参考帧
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
        # === 新增：供可视化使用的 NMS 后数据，默认 None ===
        vis_boxes = vis_cls_ids = vis_confs = vis_track_ids = None
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
            active_track_ids = {int(tid) for tid in track_ids.tolist()} if track_ids is not None else set()
            active_track_boxes = {
                int(tid): boxes[idx].astype(float)
                for idx, tid in enumerate(track_ids.tolist())
            } if track_ids is not None else {}
            active_track_confs = {
                int(tid): float(confs[idx])
                for idx, tid in enumerate(track_ids.tolist())
            } if track_ids is not None else {}
            # === 新增：清空被 NMS 压制的 track 的历史，防止未来帧触发误计数 ===
            all_track_ids_this_frame = set(result.boxes.id.int().cpu().numpy().tolist())
            suppressed_track_ids = all_track_ids_this_frame - active_track_ids
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
                # If this track was already counted, check if its stable class changed
                if track_id_int in counted_track_ids:
                    instance_idx = counted_instance_index.get(track_id_int, None)
                    if instance_idx is not None:
                        force_new_from_split, keep_tid = should_split_off_counted_track(
                            track_id=track_id_int,
                            track_box=box.astype(float),
                            mapped_instance_idx=int(instance_idx),
                            counted_instances=counted_instances,
                            counted_instance_index=counted_instance_index,
                            active_track_ids=active_track_ids,
                            active_track_boxes=active_track_boxes,
                            active_track_confs=active_track_confs,
                            frame_scale=frame_scale,
                            center_ratio=effective_recount_center_ratio,
                            recount_iou=args.recount_iou,
                            split_iou_max=args.split_sep_iou_max,
                            split_dist_scale=args.split_sep_dist_scale,
                            split_local_diag_scale=args.split_sep_local_diag_scale,
                            split_min_dist_px=args.split_sep_min_dist_px,
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
                        active_track_ids=active_track_ids,
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
                        active_track_ids=active_track_ids,
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
                            active_track_ids=active_track_ids,
                            active_track_boxes=active_track_boxes,
                            frame_scale=frame_scale,
                            center_ratio=effective_recount_center_ratio,
                            recount_iou=args.recount_iou,
                            split_iou_max=args.split_sep_iou_max,
                            split_dist_scale=args.split_sep_dist_scale,
                            split_local_diag_scale=args.split_sep_local_diag_scale,
                            split_min_dist_px=args.split_sep_min_dist_px,
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
                            for active_tid in active_track_ids:
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
                        adopt_new_owner = matched_tid < 0 or matched_tid not in active_track_ids
                        if adopt_new_owner:
                            matched_item["track_id"] = track_id_int
                            matched_item["box"] = box.astype(float)
                        else:
                            owner_box = active_track_boxes.get(matched_tid, box.astype(float))
                            matched_item["box"] = owner_box.astype(float)
                        matched_item["frame_idx"] = frame_idx
                        if args.map_dedup:
                            if adopt_new_owner and map_center is not None:
                                matched_item["map_center"] = map_center
                            elif (not adopt_new_owner) and map_motion_valid:
                                owner_box = active_track_boxes.get(matched_tid, None)
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

                # proceed to count and record
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
        # === 新增：保存 NMS 后结果供可视化用 ===
        vis_boxes, vis_cls_ids, vis_confs, vis_track_ids = boxes, cls_ids, confs, track_ids
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
                filtered_boxes=vis_boxes,       # 新增
                filtered_cls_ids=vis_cls_ids,   # 新增
                filtered_confs=vis_confs,       # 新增
                filtered_track_ids=vis_track_ids, # 新增
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
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"source: {source}\n")
        f.write(f"weights: {weights}\n")
        f.write(f"frames: {frame_idx}\n")
        f.write(f"unique_tracks: {len(counted_instances)}\n")
        if args.map_debug_summary:
            f.write("\n# Map Summary\n")
            f.write(f"motion_attempts: {map_stats['motion_attempts']}\n")
            f.write(f"motion_successes: {map_stats['motion_successes']}\n")
            f.write(f"motion_failures: {map_stats['motion_failures']}\n")
            f.write(f"phase_fallback_successes: {map_stats['phase_fallback_successes']}\n")
            f.write(f"phase_fallback_failures: {map_stats['phase_fallback_failures']}\n")
            f.write(f"dedup_checks: {map_stats['dedup_checks']}\n")
            f.write(f"dedup_hits: {map_stats['dedup_hits']}\n")
            f.write(f"center_unavailable: {map_stats['center_unavailable']}\n")
            f.write(f"map_disabled_events: {map_stats['disabled_frames']}\n")
        f.write("\n# Counts (by class)\n")
        for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
            f.write(f"{class_name}: {count}\n")

    print("Done.")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames inferred: {inferred_frames} (target infer_fps={target_infer_fps:.3f}, stride={infer_stride})")
    print(
        f"Effective conf: {effective_conf:.3f} (base_conf={args.conf:.3f}, stride_aware_conf={args.stride_aware_conf})"
    )
    print(
        f"Effective min_track_frames: {effective_min_track_frames} "
        f"(min_track_seconds={min_track_seconds:.3f}, history_maxlen={history_maxlen})"
    )
    print(
        f"Dedup stride scale: {dedup_stride_scale:.3f} "
        f"(recount_center_ratio={effective_recount_center_ratio:.5f}, map_center_ratio={effective_map_center_ratio:.5f})"
    )
    print(
        "Split tuning: "
        f"iou_max={args.split_sep_iou_max:.3f}, "
        f"dist_scale={args.split_sep_dist_scale:.3f}, "
        f"local_diag_scale={args.split_sep_local_diag_scale:.3f}, "
        f"min_dist_px={args.split_sep_min_dist_px:.1f}"
    )
    if args.map_debug_summary:
        dedup_hit_rate = (
            float(map_stats["dedup_hits"]) / float(map_stats["dedup_checks"])
            if map_stats["dedup_checks"] > 0
            else 0.0
        )
        motion_success_rate = (
            float(map_stats["motion_successes"]) / float(map_stats["motion_attempts"])
            if map_stats["motion_attempts"] > 0
            else 0.0
        )
        print(
            "Map summary: "
            f"motion_success={map_stats['motion_successes']}/{map_stats['motion_attempts']} ({motion_success_rate:.1%}), "
            f"phase_fallback={map_stats['phase_fallback_successes']}/{map_stats['phase_fallback_successes'] + map_stats['phase_fallback_failures']}, "
            f"dedup_hits={map_stats['dedup_hits']}/{map_stats['dedup_checks']} ({dedup_hit_rate:.1%}), "
            f"center_unavailable={map_stats['center_unavailable']}, "
            f"map_disabled_events={map_stats['disabled_frames']}"
        )
    if effective_track_buffer is not None:
        print(
            f"Effective track_buffer: {effective_track_buffer} frames "
            f"(~{float(args.track_buffer_seconds):.2f}s at infer_fps={target_infer_fps:.2f})"
        )
    print(f"Frames skipped by grab: {grabbed_skip_frames} (fast_grab_skip={use_fast_grab_skip})")
    print(f"Unique tracks: {len(counted_instances)}")
    print(f"Saved visualizations: {saved_vis_count}")
    print(f"Summary saved to: {summary_path}")
    if writer is not None:
        print(f"Annotated video saved to: {project_dir / f'{source.stem}_annotated.mp4'}")

    # processing time and throughput
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Processing time: {h:02d}:{m:02d}:{s:02d} ({elapsed:.2f}s)")
    if elapsed > 0:
        print(f"Average processing FPS: {frame_idx/elapsed:.2f}")

    return dict(count_by_class), vis_dir


def main() -> None:
    args = parse_args()
    process_video(args)

if __name__ == "__main__":
    main()
