import argparse
from pathlib import Path

import cv2
import numpy as np


def build_detector(detector_name, nfeatures):
    if detector_name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    if detector_name == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"unknown detector: {detector_name}")


def match_descriptors(desc1, desc2, detector_name):
    if detector_name in {"orb", "akaze"}:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def estimate_homography(template_gray, image_gray, detector_name, nfeatures):
    detector = build_detector(detector_name, nfeatures)
    kp1, des1 = detector.detectAndCompute(template_gray, None)
    kp2, des2 = detector.detectAndCompute(image_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0
    good = match_descriptors(des1, des2, detector_name)
    if len(good) < 4:
        return None, 0
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)
    if mask is None:
        return None, 0
    inliers = int(mask.sum())
    return H, inliers


def warp_to_template(image, H, template_shape):
    h, w = template_shape[:2]
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)


def process_images(template_path, input_dir, output_dir, nfeatures, matrix_path):
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"template not found: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(input_dir.glob("raw_*_warp*.png"))

    matrices = []
    for path in input_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable: {path.name}")
            continue
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_H, best_inliers = estimate_homography(template_gray, image_gray, "orb", nfeatures)
        if best_H is None or best_inliers < 10:
            H2, inliers2 = estimate_homography(template_gray, image_gray, "akaze", nfeatures)
            if H2 is not None and inliers2 > best_inliers:
                best_H = H2
                best_inliers = inliers2

        if best_H is None:
            restored = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            restored = warp_to_template(image, best_H, template.shape)
            matrices.append((path.name, best_H))

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), restored)
        print(f"{path.name} -> {out_path.name} inliers={best_inliers}")

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8") as f:
        for name, H in matrices:
            f.write(f"{name}\n")
            f.write(np.array2string(H, formatter={'float_kind': lambda x: f'{x:.6f}'}))
            f.write("\n\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("HW1_data/template.png"))
    parser.add_argument("--input_dir", type=Path, default=Path("HW1_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("523030910004_submission/restored_images"))
    parser.add_argument("--matrix_path", type=Path, default=Path("523030910004_submission/code/h_matrices.txt"))
    parser.add_argument("--nfeatures", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    process_images(args.template, args.input_dir, args.output_dir, args.nfeatures, args.matrix_path)


if __name__ == "__main__":
    main()
