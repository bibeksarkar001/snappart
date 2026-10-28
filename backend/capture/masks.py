"""Part masks for selected frames: a colour mask (Lab hue/chroma), SAM 3.1 on a crop, and their fusion. Everything works on a crop around the part.

The part is brown plastic on a pink sticky note on a white mat. In Lab, brown has hue (atan2(b*, a*)) about 40-55 deg, the pink note 21-31 deg (also in its
shadow), white and the black markers have almost no chroma. Dark brown faces lose their hue, so dark pixels count only next to lit brown ones.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CENTRE = np.array([148.5, 105.0])
BOX_MM = 35.0            # the part lies within +-35 mm of the page centre in x, y and below 45 mm
BOX_H = 45.0


def crop_box(K, rvec, tvec, size, pad=0.10):
    """(x0, y0, x1, y1) integer pixel box around the projected 70 x 70 x 45 mm region above the page centre."""
    c = np.array([[CENTRE[0] + sx * BOX_MM, CENTRE[1] + sy * BOX_MM, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, BOX_H)])
    p = cv2.projectPoints(c, np.asarray(rvec, float), np.asarray(tvec, float), np.asarray(K, float), None)[0].reshape(-1, 2)
    lo, hi = p.min(0), p.max(0)
    d = (hi - lo) * pad
    x0, y0 = np.floor(np.maximum(lo - d, 0)).astype(int)
    x1, y1 = np.ceil(np.minimum(hi + d, [size[0], size[1]])).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def hue_chroma(img):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    return lab[..., 0], np.degrees(np.arctan2(b, a)), np.hypot(a, b)


def _k(px):
    n = max(int(round(px)), 1) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def colour_mask(img, hue_min=36.0, chroma_min=10.0, dark_L=80.0, min_area=400):
    L, h, c = hue_chroma(img)
    lit = ((h > hue_min) & (h < 80) & (c > chroma_min) & (L > 25)).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, _k(3))
    near = cv2.dilate(lit, _k(9)) > 0
    m = (lit > 0) | ((L < dark_L) & near)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _k(5))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[cc == i] = 255
    inv = (out == 0).astype(np.uint8)                   # fill only tiny holes: the slots are real
    n, cc, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 30:
            out[cc == i] = 255
    return out


def iou(a, b) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def pick_component(masks, ref):
    """Of several SAM masks choose the one agreeing most with `ref` (the colour mask)."""
    if not masks:
        return None
    return max(masks, key=lambda cm: (iou(cm[1], ref), cm[0]))


def refine_sam(sam, img, band_px=2.5):
    """Correct the SAM edge only where colour is decisive: inside a band of +-band_px around it, a pixel that is clearly background (pink note incl. its
    shadow, or white mat) is dropped and one that is clearly brown is kept; everything else keeps SAM's answer. Smoothed so the 0.5 contour is sub-pixel.
    Same idea as the chroma-band step in measure/outline.py, but outline._refine fills holes (it would close the window) and its paper test is a single
    chroma distance, which is only 8 units between this brown and the pink note."""
    L, h, c = hue_chroma(img)
    ring = (cv2.dilate(sam, _k(51)) > 0) & (cv2.dilate(sam, _k(13)) == 0)
    pk = ring & (h > 15) & (h < 33) & (c > 12)
    wh = ring & (c < 9)
    bg = np.zeros(sam.shape, bool)
    if pk.sum() > 500:
        Lp, hp = np.median(L[pk]), np.median(h[pk])
        bg |= (np.abs(h - hp) < 5.5) & (c > 12) & (L > 0.55 * Lp) & (h < 34)
    if wh.sum() > 500:
        bg |= (c < 9) & (L > 0.7 * np.median(L[wh]))
    sure = (h > 38) & (h < 80) & (c > 12) & (L > 25)
    k = _k(2 * band_px + 1)
    band = (cv2.dilate(sam, k) > 0) & (cv2.erode(sam, k) == 0)
    m = (sam > 0)
    m = np.where(band & bg & ~sure, False, np.where(band & sure, True, m))
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) > 0.5).astype(np.uint8) * 255


def agreement(sam, col) -> dict:
    """colour_in_sam: share of the colour mask that lies inside SAM (the colour mask is conservative, so this should be near 1); sam_cover: share of SAM the colour mask explains."""
    c, s_ = col > 0, sam > 0
    return {"iou": iou(sam, col), "colour_in_sam": float((c & s_).sum() / max(c.sum(), 1)), "sam_cover": float((c & s_).sum() / max(s_.sum(), 1))}
