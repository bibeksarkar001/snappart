"""Height profile of a part from its edge-view silhouette (a part stood on its long edge, photographed from above).

Input: the silhouette outline primitives in mm (measure.outline). Output: the extent of the silhouette across the long axis at every
position along it, piecewise-constant steps, and round barrels (arcs of at least 120 degrees). The profile is in the photo's scale
(the reference's plane); a standing part's top is nearer the camera, so its silhouette is magnified relative to that plane.
"""
import math

import cv2
import numpy as np

from measure.outline import sample_primitives

RES = 0.05            # mm per profile sample


def _rot(u):
    return np.array([[u[0], u[1]], [-u[1], u[0]]])      # rows: u, v


def height_profile(prims, direction=None, step_mm=0.7, min_region_mm=1.0, smooth_mm=0.5):
    """direction: unit long-axis vector in the mm frame (default: minimum-area rectangle's long side, pointing +x)."""
    pts = sample_primitives(prims, 0.03)
    if direction is None:
        (_, _), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32))
        a = math.radians(ang if w >= h else ang + 90)
        direction = np.array([math.cos(a), math.sin(a)])
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    R = _rot(u)
    q = pts @ R.T                                         # x along the long axis, y across
    q -= q.min(0)
    n_x, n_y = int(q[:, 0].max() / RES) + 2, int(q[:, 1].max() / RES) + 2
    img = np.zeros((n_y, n_x), np.uint8)
    cv2.fillPoly(img, [np.round(q / RES).astype(np.int32).reshape(-1, 1, 2)], 255)
    cols = img > 0
    has = cols.any(0)
    ymin = np.where(has, cols.argmax(0), 0) * RES
    ymax = np.where(has, n_y - 1 - cols[::-1].argmax(0), 0) * RES
    height = np.where(has, ymax - ymin, 0.0)
    x = np.arange(n_x) * RES
    k = max(int(smooth_mm / RES) | 1, 1)
    hs = np.convolve(np.pad(height, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return {"x_mm": x, "height_mm": height, "height_smooth_mm": hs, "base_mm": ymin, "top_mm": ymax, "length_mm": float(x[has].max() - x[has].min()),
            "max_height_mm": float(height.max()), "x_at_max_mm": float(x[int(height.argmax())]),
            "steps": _steps(x, hs, step_mm, min_region_mm), "barrels": _barrels(prims, R, (pts @ R.T).min(0)),
            "rotation": R, "origin_shift": (pts @ R.T).min(0)}


def _steps(x, h, tol, min_len):
    """Greedy piecewise-constant segmentation of the profile: a region ends when the height leaves its level by more than tol."""
    regions, start = [], 0
    nz = np.nonzero(h > 0)[0]
    if not len(nz):
        return []
    i = nz[0]
    start = i
    level = h[i]
    for j in range(i, nz[-1] + 1):
        if abs(h[j] - level) > tol:
            if (j - start) * RES >= 0.0:
                regions.append([start, j])
            start, level = j, h[j]
        else:
            level = float(np.median(h[start:j + 1]))
    regions.append([start, nz[-1] + 1])
    out = []
    for a, b in regions:
        if (b - a) * RES < min_len:
            continue
        lv = float(np.median(h[a:b]))
        if out and abs(out[-1]["height_mm"] - lv) <= tol:      # merge neighbours at the same level
            out[-1]["x1_mm"] = float(x[min(b, len(x) - 1)])
            out[-1]["height_mm"] = float(np.median(h[int(out[-1]["x0_mm"] / RES):b]))
        else:
            out.append({"x0_mm": float(x[a]), "x1_mm": float(x[min(b, len(x) - 1)]), "height_mm": lv})
    return out


def _barrels(prims, R, shift):
    out = []
    for p in prims:
        if p["type"] == "circle" or (p["type"] == "arc" and p["sweep_deg"] >= 120):
            if not 1.5 <= p["r"] <= 15:
                continue
            c = (np.asarray(p["c"]) @ R.T) - shift
            out.append({"diameter_mm": 2 * p["r"], "sweep_deg": 360.0 if p["type"] == "circle" else p["sweep_deg"], "centre_x_mm": float(c[0]), "centre_y_mm": float(c[1])})
    return out
