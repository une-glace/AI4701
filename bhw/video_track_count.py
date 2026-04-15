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
from pathlib import Path
import sys

import cv2
import numpy as np
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

LOCAL_ULTRALYTICS = Path(__file__).resolve().parent / "ultralytics_main"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video instance tracking and counting")
    parser.add_argument("--source", type=str, required=True, help="input video path")
    parser.add_argument("--weights", type=str, default="weights/seg-best.pt", help="YOLO weights path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="tracker config")
    parser.add_argument("--conf", type=float, default=0.15, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="iou threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="inference image size")
    parser.add_argument("--device", type=str, default="cuda", help="device, e.g. 0 or cpu")
    parser.add_argument("--save-every", type=int, default=10, help="save visualization every N frames")
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
        help="only count a track after it appears in the ROI for N frames",
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
    return parser.parse_args()


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


def stable_class_id(class_ids) -> int:
    votes = defaultdict(int)
    for cls_id in class_ids:
        votes[int(cls_id)] += 1
    return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]


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


def get_box_center(box: np.ndarray) -> tuple[float, float]:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return cx, cy


def estimate_optflow_affine(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray, lk_params: dict):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if next_pts is None or status is None:
        return None, None, None

    # keep good points
    good_prev = prev_pts[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]
    if len(good_prev) < 6 or len(good_next) < 6:
        return None, good_next.reshape(-1, 1, 2) if good_next is not None else None, status

    # estimate affine transform (robust)
    M, inliers = cv2.estimateAffinePartial2D(good_prev.reshape(-1, 1, 2), good_next.reshape(-1, 1, 2), method=cv2.RANSAC)
    if M is None:
        return None, good_next.reshape(-1, 1, 2), status

    # convert to 3x3
    H = np.eye(3, dtype=np.float32)
    H[:2, :3] = M
    return H, good_next.reshape(-1, 1, 2), status


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

    if result.masks is not None:
        overlay = vis.copy()
        mask_polys = result.masks.xy
        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32) if result.boxes is not None else []
        for polygon, cls_id in zip(mask_polys, cls_ids):
            poly_i = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [poly_i], class_color(int(cls_id)))
        vis = cv2.addWeighted(overlay, 0.32, vis, 0.68, 0)

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
        confs = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id
        track_ids = track_ids.int().cpu().numpy().astype(int) if track_ids is not None else None

        for idx, box in enumerate(boxes):
            if not box_center_in_roi(box, roi):
                continue

            cls_id = int(cls_ids[idx])
            conf = float(confs[idx])
            tid = int(track_ids[idx]) if track_ids is not None else -1
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


def main() -> None:
    args = parse_args()

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

    project_dir = Path(args.project) / source.stem
    vis_dir = project_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(project_dir / f"{source.stem}_annotated.mp4"), fourcc, fps, (frame_width, frame_height))

    names = model.names
    roi = center_roi((frame_height, frame_width), args.roi_margin)
    track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(10, args.min_track_frames * 2)))
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
    saved_vis_count = 0
    save_every = max(1, int(args.save_every))
    frame_scale = float(max(frame_width, frame_height))
    max_stale_local = int(max(0.0, args.local_max_stale_seconds) * fps)
    max_stale_map = int(max(0.0, args.map_max_stale_seconds) * fps)

    # Reference-map dedup / motion estimation state for complex camera motion.
    ref_gray = None
    last_h_map = np.eye(3, dtype=np.float32)
    last_ref_update_frame = 0
    map_scale = float(max(frame_width, frame_height))

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

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            verbose=False,
        )
        result = results[0]

        if args.map_dedup:
            ref_update_every = getattr(args, "ref_update_every", 90)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ref_gray is None:
                ref_gray = curr_gray
                last_h_map = np.eye(3, dtype=np.float32)
                last_ref_update_frame = frame_idx
                prev_gray_opt = curr_gray
                prev_pts = detect_optflow_points(curr_gray, max_corners, quality_level, min_dist)
            else:
                # 通过相邻帧光流估计 prev->curr 变换，再累计得到 curr->ref 变换。
                h_prev_to_curr = None
                tracked_pts = None
                if prev_gray_opt is not None and prev_pts is not None and len(prev_pts) >= 6:
                    h_prev_to_curr, tracked_pts, _ = estimate_optflow_affine(prev_gray_opt, curr_gray, prev_pts, lk_params)

                if h_prev_to_curr is not None:
                    try:
                        h_curr_to_prev = np.linalg.inv(h_prev_to_curr)
                        last_h_map = (last_h_map @ h_curr_to_prev).astype(np.float32)
                    except np.linalg.LinAlgError:
                        pass

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
                    last_ref_update_frame = frame_idx
                    prev_gray_opt = curr_gray
                    prev_pts = detect_optflow_points(curr_gray, max_corners, quality_level, min_dist)

        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
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

            for cls_id, track_id, box, conf in zip(cls_ids, track_ids, boxes, confs):
                track_id_int = int(track_id)
                if not box_center_in_roi(box, roi):
                    continue
                if box_too_close_to_edge(box, frame_width, frame_height, args.min_edge_distance):
                    continue

                history = track_history[track_id_int]
                history.append({"cls_id": int(cls_id), "box": box.astype(float), "conf": float(conf)})

                if len(history) < args.min_track_frames:
                    continue

                mean_conf = float(np.mean([item["conf"] for item in history]))
                if mean_conf < args.min_track_conf:
                    continue

                stable_cls = stable_class_id(item["cls_id"] for item in history)
                class_name = get_class_name(names, stable_cls)
                if args.ignore_background and class_name.lower() in {"__background__", "background"}:
                    continue

                # If this track was already counted, check if its stable class changed
                if track_id_int in counted_track_ids:
                    # ← 新增：每帧刷新已计数实例的坐标，防止坐标过时导致去重失效
                    counted_last_seen[track_id_int] = frame_idx
                    instance_idx = counted_instance_index.get(track_id_int, None)
                    if instance_idx is not None:
                        it = counted_instances[instance_idx]
                        it["box"] = box.astype(float)
                        it["frame_idx"] = frame_idx
                        if args.map_dedup:
                            mc = project_center_to_map(box.astype(float), last_h_map)
                            if mc is not None:
                                it["map_center"] = mc
                    # 原有的 class 修正逻辑保持不变
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
                dup_local, dup_local_idx = is_duplicate_count(
                    box, counted_instances,
                    args.recount_center_ratio, args.recount_iou, frame_scale,
                    candidate_cls=int(stable_cls),
                    allow_cross_class=args.allow_cross_class_relink,
                    hard_iou_any_class=args.recount_hard_iou,
                    counted_last_seen=counted_last_seen,
                    current_frame=frame_idx,
                    max_stale_frames=max_stale_local,
                    active_track_ids=active_track_ids,
                )
                map_center = project_center_to_map(box.astype(float), last_h_map) if args.map_dedup else None
                dup_map = False
                dup_map_idx = None
                if args.map_dedup and map_center is not None:
                    dup_map, dup_map_idx = is_duplicate_count_map(
                        map_center,
                        counted_instances,
                        args.map_center_ratio,
                        map_scale,
                        candidate_cls=int(stable_cls),
                        allow_cross_class=args.allow_cross_class_relink,
                        counted_last_seen=counted_last_seen,
                        current_frame=frame_idx,
                        max_stale_frames=max_stale_map,
                        active_track_ids=active_track_ids,
                    )
                if dup_local or dup_map:
                    matched_idx = dup_local_idx if dup_local_idx is not None else dup_map_idx
                    if matched_idx is not None:
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
                        # Re-link new tracker id to existing counted instance rather than recount.
                        matched_item["track_id"] = track_id_int
                        matched_item["box"] = box.astype(float)
                        matched_item["frame_idx"] = frame_idx
                        if args.map_dedup and map_center is not None:
                            matched_item["map_center"] = map_center
                    if args.debug_dups:
                        reason = "local" if dup_local else "map"
                        action = "relink" if matched_idx is not None else "skip"
                        print(f"[dup-{action}] frame={frame_idx} track={track_id} cls={stable_cls} reason={reason}")
                    continue

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

        save_this_frame = frame_idx == 1 or frame_idx % save_every == 0
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
        f.write("\n# Counts (by class)\n")
        for class_name, count in sorted(count_by_class.items(), key=lambda x: x[0]):
            f.write(f"{class_name}: {count}\n")

    print("Done.")
    print(f"Frames processed: {frame_idx}")
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


if __name__ == "__main__":
    main()
