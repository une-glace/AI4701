import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
from tqdm import tqdm

# ======================
# 配置
# ======================
IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"

OUT_LBL_DIR = "dataset/filtered_labels"
VIS_DIR = "dataset/vis"

IOU_THRESH = 0.2
DIST_THRESH = 80

AREA_NORM_THRESH = 0.7
COMPACTNESS_THRESH = 0.3
KEEP_MARGIN = 2

# 边缘检测阈值（像素）
EDGE_THRESHOLD = 5

os.makedirs(OUT_LBL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ======================
# 工具函数
# ======================
def load_yolo_seg(label_path, img_w, img_h):
    objs = []
    if not os.path.exists(label_path):
        return objs

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            coords = np.array(parts[1:]).reshape(-1, 2)

            coords[:, 0] *= img_w
            coords[:, 1] *= img_h

            objs.append({
                "cls": cls,
                "poly": coords
            })
    return objs


def poly_to_bbox(poly):
    x_min = np.min(poly[:, 0])
    y_min = np.min(poly[:, 1])
    x_max = np.max(poly[:, 0])
    y_max = np.max(poly[:, 1])
    return np.array([x_min, y_min, x_max, y_max])


def bbox_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def center_distance(a, b):
    cx1 = (a[0] + a[2]) / 2
    cy1 = (a[1] + a[3]) / 2
    cx2 = (b[0] + b[2]) / 2
    cy2 = (b[1] + b[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def mask_area(poly):
    return cv2.contourArea(poly.astype(np.float32))


def is_touching_edge(poly, img_w, img_h, edge_threshold=EDGE_THRESHOLD):
    """
    判断多边形是否接触图像边缘
    edge_threshold: 距离边缘多少像素以内算接触
    """
    x_min = np.min(poly[:, 0])
    y_min = np.min(poly[:, 1])
    x_max = np.max(poly[:, 0])
    y_max = np.max(poly[:, 1])
    
    # 检查是否接近任何边缘
    touches_left = x_min <= edge_threshold
    touches_right = x_max >= img_w - edge_threshold
    touches_top = y_min <= edge_threshold
    touches_bottom = y_max >= img_h - edge_threshold
    
    return touches_left or touches_right or touches_top or touches_bottom


# ======================
# 读取数据
# ======================
img_paths = sorted(glob(os.path.join(IMG_DIR, "*")))

frames = []

print("📖 Loading data...")
for idx, img_path in enumerate(tqdm(img_paths, desc='Loading', unit='img')):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    label_path = os.path.join(
        LBL_DIR,
        os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
    )

    objs = load_yolo_seg(label_path, w, h)

    for obj in objs:
        obj["bbox"] = poly_to_bbox(obj["poly"])
        obj["frame"] = idx
        obj["track_id"] = -1
        obj["img_w"] = w
        obj["img_h"] = h
        # 预先计算是否接触边缘
        obj["touching_edge"] = is_touching_edge(obj["poly"], w, h)

    frames.append(objs)


# ======================
# Tracking
# ======================
print("🔗 Tracking objects...")
track_id = 0

for i in tqdm(range(len(frames) - 1), desc='Tracking', unit='frame'):
    curr = frames[i]
    nxt = frames[i + 1]

    for obj in curr:
        best_score = 1e9
        best_j = -1

        for j, obj2 in enumerate(nxt):
            # 只匹配相同类别
            if obj["cls"] != obj2["cls"]:
                continue
                
            iou = bbox_iou(obj["bbox"], obj2["bbox"])
            dist = center_distance(obj["bbox"], obj2["bbox"])

            if iou > IOU_THRESH or dist < DIST_THRESH:
                if dist < best_score:
                    best_score = dist
                    best_j = j

        if best_j != -1:
            if obj["track_id"] == -1:
                obj["track_id"] = track_id
                track_id += 1

            nxt[best_j]["track_id"] = obj["track_id"]


# ======================
# 收集 track
# ======================
tracks = defaultdict(list)

for frame in frames:
    for obj in frame:
        if obj["track_id"] != -1:
            tracks[obj["track_id"]].append(obj)

print(f"📊 Found {len(tracks)} tracks")


# ======================
# 决策保留策略
# ======================
print("🎯 Deciding which objects to keep...")
keep_map = {}

# 首先，标记所有不接触边缘的对象为保留
not_touching_count = 0
for frame in frames:
    for obj in frame:
        if not obj["touching_edge"]:
            keep_map[id(obj)] = True
            not_touching_count += 1

print(f"✨ Objects not touching edges: {not_touching_count}")

# 然后，对有track的对象进行anchor分析
for tid, objs in tqdm(tracks.items(), desc='Analyzing tracks', unit='track'):
    areas = []
    compactness_list = []

    for obj in objs:
        x1, y1, x2, y2 = obj["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        m_area = mask_area(obj["poly"])

        areas.append(bbox_area)
        compactness_list.append(m_area / (bbox_area + 1e-6))

    areas = np.array(areas)
    compactness_list = np.array(compactness_list)

    ref_area = np.percentile(areas, 80)
    area_norm = areas / (ref_area + 1e-6)

    # 找完整帧（anchor frames）
    complete_idx = np.where(
        (area_norm > AREA_NORM_THRESH) &
        (compactness_list > COMPACTNESS_THRESH)
    )[0]

    if len(complete_idx) == 0:
        continue

    start = max(0, complete_idx[0] - KEEP_MARGIN)
    end = min(len(objs) - 1, complete_idx[-1] + KEEP_MARGIN)

    # 在anchor范围内的对象也保留
    for i in range(len(objs)):
        if start <= i <= end:
            obj_id = id(objs[i])
            # 如果还没标记为保留，现在标记
            if obj_id not in keep_map:
                keep_map[obj_id] = True


# ======================
# 输出 + 可视化
# ======================
print("💾 Saving results...")
kept_count = 0
total_count = 0

for idx, img_path in enumerate(tqdm(img_paths, desc='Saving', unit='img')):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    out_lines = []

    for obj in frames[idx]:
        total_count += 1
        obj_id = id(obj)
        keep = keep_map.get(obj_id, False)

        if keep:
            kept_count += 1
            poly = obj["poly"].copy()
            poly[:, 0] /= w
            poly[:, 1] /= h

            line = [obj["cls"]] + poly.flatten().tolist()
            out_lines.append(" ".join(map(str, line)))

        # 可视化
        if keep:
            if not obj["touching_edge"]:
                color = (255, 0, 0)  # 蓝色：不接触边缘
            else:
                color = (0, 255, 0)  # 绿色：在anchor范围内
        else:
            color = (0, 0, 255)  # 红色：丢弃
        
        # 绘制轮廓
        cv2.polylines(img, [obj["poly"].astype(int)], True, color, 2)
        
        # 可选：添加文字标注
        if not obj["touching_edge"] and keep:
            cv2.putText(img, "no_edge", (int(obj["bbox"][0]), int(obj["bbox"][1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out_path = os.path.join(
        OUT_LBL_DIR,
        os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
    )

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))

    vis_path = os.path.join(VIS_DIR, os.path.basename(img_path))
    cv2.imwrite(vis_path, img)

print(f"\n✅ Processing complete!")
print(f"📈 Statistics:")
print(f"   Total objects: {total_count}")
print(f"   Kept objects: {kept_count} ({100*kept_count/total_count:.1f}%)")
print(f"   Discarded objects: {total_count - kept_count}")
print(f"   Objects kept due to not touching edge: {not_touching_count}")