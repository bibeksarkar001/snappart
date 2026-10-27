"""Small helpers shared by the phase 1 spikes."""
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
PHOTO = ROOT / "assets" / "golden" / "test_01.jpg"
HAND_PHOTO = ROOT / "assets" / "golden" / "test_02_hand.jpg"
COIN_MM = {"25.0 (Numista catalogue, unverified)": 25.0, "23.0 (Wikipedia, unverified)": 23.0}


def save_overlay(path, img, max_side=1200):
    """Write a preview PNG no larger than max_side px on its long side."""
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img)


def full_mask(box, raster, w, h):
    """Paste a box-local SAM raster into a full-frame 0/255 mask (raster is stretched to the box)."""
    bw, bh = box.right - box.left, box.bottom - box.top
    m = np.zeros((h, w), np.uint8)
    m[box.top:box.bottom, box.left:box.right] = cv2.resize(raster * 255, (bw, bh), interpolation=cv2.INTER_NEAREST)
    return m


def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    out[mask > 0] = ((1 - alpha) * out[mask > 0] + alpha * np.array(color)).astype(np.uint8)
    return out


def unstretch_fn(coin):
    """Fit the coin ellipse; return (mapper, major_px, minor_px, tilt_deg, raw ellipse).

    The mapper stretches the minor-axis direction until the coin is a circle; mapper.inv undoes it.
    """
    cnts, _ = cv2.findContours((coin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (a, b), ang = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
    major, minor = max(a, b), min(a, b)
    th = math.radians((ang if a >= b else ang + 90) + 90)  # direction of the minor axis
    u, k = np.array([math.cos(th), math.sin(th)]), major / minor

    def stretch(pts, factor):
        p = np.asarray(pts, float).reshape(-1, 2)
        return p + (factor - 1) * np.outer((p - [cx, cy]) @ u, u)

    def f(pts):
        return stretch(pts, k)
    f.inv = lambda pts: stretch(pts, 1 / k)
    return f, major, minor, math.degrees(math.acos(min(1.0, minor / major))), ((cx, cy), (a, b), ang)
