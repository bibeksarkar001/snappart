"""Small helpers shared by the phase 1 spikes."""
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
