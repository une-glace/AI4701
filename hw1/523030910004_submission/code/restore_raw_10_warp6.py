import argparse
from pathlib import Path

import cv2
import numpy as np


def build_detector(name, nfeatures):
    if name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    if name == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"unknown detector: {name}")


def match_descriptors(desc1, desc2, detector_name, ratio):
    if detector_name in {"orb", "akaze"}:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def filter_matches_grid(matches, keypoints, image_shape, grid_rows, grid_cols, max_per_cell):
    h, w = image_shape[:2]
    cell_w = max(1, w / grid_cols)
    cell_h = max(1, h / grid_rows)
    buckets = {}
    for m in sorted(matches, key=lambda x: x.distance):
        x, y = keypoints[m.trainIdx].pt
        col = min(grid_cols - 1, int(x / cell_w))
        row = min(grid_rows - 1, int(y / cell_h))
        key = (row, col)
        if key not in buckets:
            buckets[key] = []
        if len(buckets[key]) < max_per_cell:
            buckets[key].append(m)
    filtered = []
    for key in buckets:
        filtered.extend(buckets[key])
    return filtered


def estimate_homography(template_gray, image_gray, detector_name, nfeatures, ratio, grid_rows, grid_cols, max_per_cell):
    detector = build_detector(detector_name, nfeatures)
    kp1, des1 = detector.detectAndCompute(template_gray, None)
    kp2, des2 = detector.detectAndCompute(image_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0
    matches = match_descriptors(des1, des2, detector_name, ratio)
    if len(matches) < 4:
        return None, 0
    matches = filter_matches_grid(matches, kp2, image_gray.shape, grid_rows, grid_cols, max_per_cell)
    if len(matches) < 4:
        return None, 0
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=8000, confidence=0.995)
    if mask is None:
        return None, 0
    return H, int(mask.sum())


def warp_to_template(image, H, template_shape):
    h, w = template_shape[:2]
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)


def process_single(template_path, image_path, output_path, nfeatures, ratio, grid_rows, grid_cols, max_per_cell):
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"template not found: {template_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"image not found: {image_path}")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best_H, best_inliers = estimate_homography(
        template_gray, image_gray, "orb", nfeatures, ratio, grid_rows, grid_cols, max_per_cell
    )
    H2, inliers2 = estimate_homography(
        template_gray, image_gray, "akaze", nfeatures, ratio, grid_rows, grid_cols, max_per_cell
    )
    if H2 is not None and inliers2 > best_inliers:
        best_H = H2
        best_inliers = inliers2

    if best_H is None:
        restored = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        restored = warp_to_template(image, best_H, template.shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), restored)
    return best_inliers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("HW1_data/template.png"))
    parser.add_argument("--image", type=Path, default=Path("HW1_data/raw_10_warp6.png"))
    parser.add_argument(
        "--output", type=Path, default=Path("523030910004_submission/restored_images/raw_10_warp6_refined.png")
    )
    parser.add_argument("--nfeatures", type=int, default=8000)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--max_per_cell", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    inliers = process_single(
        args.template, args.image, args.output, args.nfeatures, args.ratio, args.grid_rows, args.grid_cols, args.max_per_cell
    )
    print(f"saved: {args.output} inliers={inliers}")


if __name__ == "__main__":
    main()
