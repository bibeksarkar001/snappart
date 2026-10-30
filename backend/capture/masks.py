"""Part masks for selected frames: a colour mask (Lab hue/chroma), SAM 3.1 on a crop, and their fusion. Everything works on a crop around the part.

The colour model is the part's Lab hue window (hue = atan2(b*, a*)). learn_colour() learns it per job from the pixels inside SAM's own masks;
an achromatic part (grey, black, white) has no reliable hue, and then no colour gating is used. The default window (38-80 deg) is part B's
brown plastic, which sat on a pink note (hue 21-31 deg) on a white mat; with that default every function below behaves exactly as before.
Dark faces lose their hue, so dark pixels count only next to lit part-coloured ones.
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


def learn_colour(crops_and_masks, chroma_min=15.0, erode_px=9):
    """The part's hue window from SAM's masks: only well-inside (eroded), well-lit (upper half of lightness), clearly coloured
    (chroma > chroma_min) pixels, so edges, shadows and dark faces do not drag it toward the background. Window = 10th percentile .. 90th
    percentile + 20 deg (nothing above the part's hue was background on the mat). None when the part has too little colour for hue to mean
    anything (median chroma of its lit pixels under chroma_min). On part B this gives 38.3 deg for the low edge (hand-tuned: 38)."""
    hs, cs = [], []
    for crop, m in crops_and_masks:
        inner = cv2.erode(m, _k(erode_px)) > 0
        if inner.sum() < 50:
            continue
        L, h, c = hue_chroma(crop)
        lit = inner & (L >= np.median(L[inner]))
        hs.append(h[lit])
        cs.append(c[lit])
    if not hs:
        return None
    h, c = np.concatenate(hs), np.concatenate(cs)
    if np.median(c) < chroma_min:
        return None
    hh = h[c > chroma_min]
    return (float(np.percentile(hh, 10)), float(np.percentile(hh, 90) + 20))


def colour_mask(img, hue_min=36.0, chroma_min=10.0, dark_L=80.0, min_area=400, hue_max=80.0):
    L, h, c = hue_chroma(img)
    lit = ((h > hue_min) & (h < hue_max) & (c > chroma_min) & (L > 25)).astype(np.uint8)
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


def refine_sam(sam, img, band_px=2.5, part_hue=None):
    """Correct the SAM edge only where colour is decisive: inside a band of +-band_px around it, a pixel that is clearly background (pink note incl. its
    shadow, or white mat) is dropped and one that is clearly brown is kept; everything else keeps SAM's answer. Smoothed so the 0.5 contour is sub-pixel.
    Same idea as the chroma-band step in measure/outline.py, but outline._refine fills holes (it would close the window) and its paper test is a single
    chroma distance, which is only 8 units between this brown and the pink note."""
    lo, hi = part_hue or (38.0, 80.0)                       # background colour = a hue band just below the part's (part B: pink 15-33)
    L, h, c = hue_chroma(img)
    ring = (cv2.dilate(sam, _k(51)) > 0) & (cv2.dilate(sam, _k(13)) == 0)
    pk = ring & (h > lo - 23) & (h < lo - 5) & (c > 12)
    wh = ring & (c < 9)
    bg = np.zeros(sam.shape, bool)
    if pk.sum() > 500:
        Lp, hp = np.median(L[pk]), np.median(h[pk])
        bg |= (np.abs(h - hp) < 5.5) & (c > 12) & (L > 0.55 * Lp) & (h < lo - 4)
    if wh.sum() > 500:
        bg |= (c < 9) & (L > 0.7 * np.median(L[wh]))
    sure = (h > lo) & (h < hi) & (c > 12) & (L > 25)
    k = _k(2 * band_px + 1)
    band = (cv2.dilate(sam, k) > 0) & (cv2.erode(sam, k) == 0)
    m = (sam > 0)
    m = np.where(band & bg & ~sure, False, np.where(band & sure, True, m))
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) > 0.5).astype(np.uint8) * 255


def agreement(sam, col) -> dict:
    """colour_in_sam: share of the colour mask that lies inside SAM (the colour mask is conservative, so this should be near 1); sam_cover: share of SAM the colour mask explains."""
    c, s_ = col > 0, sam > 0
    return {"iou": iou(sam, col), "colour_in_sam": float((c & s_).sum() / max(c.sum(), 1)), "sam_cover": float((c & s_).sum() / max(s_.sum(), 1))}
