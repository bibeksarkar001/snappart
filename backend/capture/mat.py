"""The printed calibration mat: ArUco layout, world frame and per-frame pose.

World frame (right-handed, mm): origin at the page's top-left corner, x to the right, y UP the page (page y is down in
mat_layout.json, so y_world = page_h - y_page), z up out of the mat. Markers lie at z = 0.
"""
import json
from pathlib import Path

import cv2
import numpy as np

LAYOUT = json.loads(Path(__file__).with_name("mat_layout.json").read_text())
PAGE_W, PAGE_H = LAYOUT["page_mm"]
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_corners_world() -> dict[int, np.ndarray]:
    """id -> (4, 3) world corners in ArUco order (top-left, top-right, bottom-right, bottom-left of the printed marker)."""
    out = {}
    for k, m in LAYOUT["markers"].items():
        x, y, s = m["x_mm"], PAGE_H - m["y_mm"], m["size_mm"]     # (x, y) is the marker's top-left on the page
        out[int(k)] = np.array([[x, y, 0], [x + s, y, 0], [x + s, y - s, 0], [x, y - s, 0]], float)
    return out


WORLD = marker_corners_world()


def make_detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 60
    p.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(DICT, p)


def detect(detector, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ids (n,), corners (n, 4, 2)) of known mat markers only, one detection per id."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return np.zeros(0, int), np.zeros((0, 4, 2))
    seen, keep = set(), []
    for i, (mid, c) in enumerate(zip(ids.ravel(), corners)):
        if int(mid) in WORLD and int(mid) not in seen:
            seen.add(int(mid))
            keep.append(i)
    return ids.ravel()[keep].astype(int), np.array([corners[i].reshape(4, 2) for i in keep]).reshape(-1, 4, 2)


def correspondences(ids, corners):
    """3D world points (4n, 3) and their 2D detections (4n, 2) for the given detections."""
    return (np.concatenate([WORLD[int(i)] for i in ids]) if len(ids) else np.zeros((0, 3)),
            np.asarray(corners, float).reshape(-1, 2))


def solve_pose(ids, corners, K, dist, outlier_px: float = 3.0):
    """World -> camera pose (rvec, tvec) from the marker corners, with LM refinement and one outlier-marker pass.

    Returns (rvec, tvec, rms_px, used_ids) or None. rms is over the corners of the markers kept.
    """
    ids, corners = np.asarray(ids), np.asarray(corners)
    if len(ids) < 1:
        return None
    keep = np.arange(len(ids))
    for _ in range(2):
        P, p = correspondences(ids[keep], corners[keep])
        if len(P) < 4:
            return None
        ok, rv, tv = cv2.solvePnP(P, p, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rv, tv = cv2.solvePnPRefineLM(P, p, K, dist, rv, tv)
        err = np.linalg.norm(cv2.projectPoints(P, rv, tv, K, dist)[0].reshape(-1, 2) - p, axis=1).reshape(-1, 4).max(axis=1)
        good = err <= max(outlier_px, 2 * np.median(err))
        if good.all() or good.sum() < 1:
            break
        keep = keep[good]
    P, p = correspondences(ids[keep], corners[keep])
    e = np.linalg.norm(cv2.projectPoints(P, rv, tv, K, dist)[0].reshape(-1, 2) - p, axis=1)
    return rv, tv, float(np.sqrt((e ** 2).mean())), ids[keep]


def render_mat(px_per_mm: float = 8.0, gray_border: int = 12) -> np.ndarray:
    """Synthetic mat image (page white, black markers) for tests; canvas (u, v) = (x_world, page_h - y_world) * px_per_mm."""
    w, h = int(PAGE_W * px_per_mm), int(PAGE_H * px_per_mm)
    img = np.full((h, w), 255, np.uint8)
    for k, m in LAYOUT["markers"].items():
        s = int(round(m["size_mm"] * px_per_mm))
        x, y = int(round(m["x_mm"] * px_per_mm)), int(round(m["y_mm"] * px_per_mm))
        img[y:y + s, x:x + s] = cv2.aruco.generateImageMarker(DICT, int(k), s)
    return img
