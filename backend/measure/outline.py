"""Millimetre outlines of a flat part from one photo.

Input: the photo, a mask of the part (SAM), a mask of the reference object (coin or bank card) and its size in mm.
Output: the part outline as straight lines and circular arcs in mm, the holes and slots inside it, and a quality score.

How it works:
1. The reference gives the scale. A coin is ellipse-fitted and the tilt is "un-stretched" (the minor axis is stretched until
   the coin is a circle). A card is stretched along its own two sides to its true size.
2. The part mask is smoothed, then only a band around its edge is re-decided by colour: plastic is coloured (Lab a*/b* far
   from the paper), a shadow is grey. Pixels covered by an optional exclusion mask (a hand) never become part.
3. A sub-pixel contour of the refined mask is smoothed and split into lines and arcs, tiny segments are dropped, near-collinear
   lines merged, near-axis lines snapped, and neighbouring primitives closed onto shared vertices.
4. Holes and slots are enclosed regions inside the part whose colour matches the paper sample (not warmer than the paper, which
   rejects glare on coloured plastic) and that are enclosed by the part on every side. They are not inner contours of the mask.

Coordinates are millimetres, x to the right and y DOWN (image orientation), origin at the lower-left-most corner of the parts'
bounding box. Mirror y for a CAD sketch. `quality` is a weakest-link heuristic in 0..1, not a calibrated accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.measure import find_contours

Primitive = dict


@dataclass(frozen=True)
class Params:
    band_mm: float = 1.1                 # the SAM edge is re-decided this far either side
    mask_smooth_mm: float = 0.3          # blur of the (stretched, low-res) SAM mask before the band is cut
    min_part_mm2: float = 32.0           # smaller blobs are not parts
    contour_smooth_mm: float = 0.10      # smoothing along the contour
    fit_tol_mm: float = 0.07             # lines/arcs fit the smoothed contour this closely before closing
    min_segment_mm: float = 0.3          # shorter primitives are dropped
    corner_mm: float = 1.2               # a run of small primitives shorter than this between two long lines becomes a sharp corner
    min_arc_radius_mm: float = 0.5       # smaller arcs are smoothing-rounded corners, not features: dropped (corner stays sharp)
    merge_tol_mm: float = 0.1            # merged lines/arcs stay this close to what they replace
    collinear_deg: float = 3.0           # lines closer than this in direction may merge
    snap_deg: float = 3.0                # lines this close to the main axes are rotated onto them ...
    snap_tol_mm: float = 0.15            # ... if they still lie this close to the contour
    vertex_move_mm: float = 0.3          # a closed vertex may move this far from where the primitives ended
    hole_min_mm2: float = 3.0
    slot_elongation: float = 2.0         # long/short above this = slot
    hole_solidity: float = 0.8           # region area / convex hull area
    hole_open_mm: float = 0.66           # opening that cuts glint speckle away from real openings
    paper_ab_strict: float = 4.5         # certain paper: chroma distance to the paper sample ...
    paper_l_strict: float = 0.5          # ... and lightness relative to it
    paper_ab_loose: float = 8.0          # paper in light shadow, kept only where connected to certain paper
    paper_l_loose: float = 0.3
    hole_min_rel_l: float = 0.6          # core of an accepted hole: lightness relative to the paper sample
    hole_max_da: float = 3.0             # |a* - paper a*|
    hole_max_db: float = 1.5             # b* may not be warmer than the paper's (glare on brown plastic is warm)
    hole_max_sd_l: float = 0.25          # lightness spread inside the hole
    hole_ring_min: float = 0.9           # fraction of the surrounding ring that must lie inside the part, in every sector
    exclude_margin_mm: float = 0.7       # keep-out around the exclusion mask (hand)
    contact_mm: float = 1.1              # outline this close to the exclusion mask counts as hand contact


@dataclass
class Hole:
    kind: str                            # "hole" or "slot"
    long_mm: float
    short_mm: float
    area_mm2: float
    primitives: list[Primitive]


@dataclass
class Part:
    primitives: list[Primitive]
    holes: list[Hole]
    long_mm: float                       # minimum-area rectangle of the outline
    short_mm: float
    area_mm2: float
    max_dev_mm: float                    # primitives vs the smoothed contour, both directions
    p95_dev_mm: float
    hand_contact_mm: float = 0.0


@dataclass
class Frame:
    """Maps between image pixels and the (shifted) mm frame of the result."""
    to_mm_raw: object
    to_px_raw: object
    origin_mm: np.ndarray

    def to_mm(self, px):
        return self.to_mm_raw(px) - self.origin_mm

    def to_px(self, mm):
        return self.to_px_raw(np.asarray(mm, float) + self.origin_mm)


@dataclass
class OutlineResult:
    parts: list[Part]
    mm_per_px: float
    tilt_deg: float | None               # coin only
    ref_kind: str
    quality: float
    quality_parts: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    frame: Frame | None = None

    def to_dict(self) -> dict:
        def part(p: Part) -> dict:
            return {"primitives": p.primitives, "long_mm": p.long_mm, "short_mm": p.short_mm, "area_mm2": p.area_mm2,
                    "max_dev_mm": p.max_dev_mm, "p95_dev_mm": p.p95_dev_mm, "hand_contact_mm": p.hand_contact_mm,
                    "holes": [h.__dict__ for h in p.holes]}
        return {"parts": [part(p) for p in self.parts], "mm_per_px": self.mm_per_px, "tilt_deg": self.tilt_deg,
                "ref_kind": self.ref_kind, "quality": self.quality, "quality_parts": self.quality_parts, "warnings": self.warnings}


# ---------------------------------------------------------------- reference

def _reference_contour(mask: np.ndarray) -> np.ndarray:
    """Sub-pixel outline of the biggest blob of the reference mask. Pixel-centre contours are about 1 px too small across,
    which is 0.3 % of a coin's diameter and passes straight into every size."""
    cs = _contours((mask > 0).astype(np.uint8) * 255, 100, 1.0)
    if not cs:
        raise ValueError("reference mask is empty or too small")
    return max(cs, key=lambda c: cv2.contourArea(c.astype(np.float32)))


def _coin_frame(mask: np.ndarray, diameter_mm: float):
    """Un-stretch mapping from a coin mask. Returns (to_mm, to_px, mm_per_px, tilt_deg, fit_rms_rel)."""
    c = _reference_contour(mask)
    (cx, cy), (a, b), ang = cv2.fitEllipse(c.astype(np.float32))
    major, minor = max(a, b), min(a, b)
    th = math.radians((ang if a >= b else ang + 90) + 90)       # direction of the minor axis
    u, k = np.array([math.cos(th), math.sin(th)]), major / minor
    s = diameter_mm / major

    def stretch(p, f):
        p = np.asarray(p, float).reshape(-1, 2)
        return p + (f - 1) * np.outer((p - [cx, cy]) @ u, u)

    # rms distance of the mask boundary from the fitted ellipse, relative to the diameter
    rel = stretch(c, k) - [cx, cy]
    rms = float(np.sqrt(np.mean((np.linalg.norm(rel, axis=1) - major / 2) ** 2)) / major)
    return (lambda p: stretch(p, k) * s, lambda m: stretch(np.asarray(m, float) / s, 1 / k), s,
            math.degrees(math.acos(min(1.0, minor / major))), rms)


def _card_frame(mask: np.ndarray, size_mm: tuple[float, float]):
    """Stretch along the card's own two sides so it becomes size_mm. Returns the same tuple as _coin_frame (tilt = None)."""
    c = _reference_contour(mask)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c.astype(np.float32))
    long_mm, short_mm = max(size_mm), min(size_mm)
    if w >= h:
        u_long, u_short = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]), None
        long_px, short_px = w, h
    else:
        u_long, u_short = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))]), None
        long_px, short_px = h, w
    u_short = np.array([-u_long[1], u_long[0]])
    sl, ss = long_mm / long_px, short_mm / short_px

    def to_mm(p):
        d = np.asarray(p, float).reshape(-1, 2) - [cx, cy]
        return np.c_[d @ u_long * sl, d @ u_short * ss]

    def to_px(m):
        m = np.asarray(m, float).reshape(-1, 2)
        return [cx, cy] + np.outer(m[:, 0] / sl, u_long) + np.outer(m[:, 1] / ss, u_short)
    area_err = abs(float(cv2.contourArea(c.astype(np.float32))) / (long_px * short_px) - 1)   # how rectangular the mask is
    return to_mm, to_px, math.sqrt(sl * ss), None, area_err


# ---------------------------------------------------------------- mask refinement

def _odd(px: float) -> int:
    n = max(int(round(px)), 1)
    return n if n % 2 else n + 1


def _kernel(px: float, ellipse: bool = False):
    n = _odd(px)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE if ellipse else cv2.MORPH_RECT, (n, n))


def _fill_external(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def _paper_sample(lab, part, ref_mask, exclude, ppm):
    """Median Lab of the paper in a ring 4 to 19 mm around the part, clear of the reference and the exclusion mask."""
    ring = (cv2.dilate(part, _kernel(19 * ppm)) > 0) & (cv2.dilate(part, _kernel(4.4 * ppm)) == 0)
    ring &= cv2.dilate(ref_mask, _kernel(10 * ppm)) == 0
    if exclude is not None:
        ring &= cv2.dilate(exclude, _kernel(10 * ppm)) == 0
    if ring.sum() < 1000:
        raise ValueError("no clear paper around the part to sample")
    return np.median(lab[ring], axis=0)


def _refine(img, part, ref_mask, exclude, ppm, p: Params):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = _paper_sample(lab, part, ref_mask, exclude, ppm)
    blur = cv2.GaussianBlur(lab, (0, 0), max(0.09 * ppm, 0.8))
    d_ab = np.hypot(blur[..., 1] - paper[1], blur[..., 2] - paper[2])
    sm = (cv2.GaussianBlur(part.astype(np.float32) / 255, (0, 0), p.mask_smooth_mm * ppm) > 0.5).astype(np.uint8) * 255
    k = _kernel(2 * p.band_mm * ppm + 1, ellipse=True)
    core, outer = cv2.erode(sm, k), cv2.dilate(sm, k)
    in_band = (outer > 0) & (core == 0)
    plastic = float(np.median(d_ab[cv2.erode(sm, _kernel(4.4 * ppm)) > 0])) if (cv2.erode(sm, _kernel(4.4 * ppm)) > 0).any() else 0.0
    thr = max(6.0, 0.5 * plastic)         # half the plastic's own chroma: where the edge sits
    keep_out = np.zeros(part.shape, bool)
    if exclude is not None:
        keep_out = cv2.dilate(exclude, _kernel(2 * p.exclude_margin_mm * ppm + 1, ellipse=True)) > 0
    mask = (((core > 0) | (in_band & (d_ab > thr))) & ~keep_out).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(0.37 * ppm))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= p.min_part_mm2 * ppm ** 2:
            keep[cc == i] = 255
    return _fill_external(keep), blur, paper, keep_out, thr


def _contours(mask: np.ndarray, min_area_px: float, sigma_px: float):
    """Closed sub-pixel contours (x, y), 1 px spacing, smoothed along the arc."""
    out = []
    for c in find_contours(cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.0), 0.5):
        if len(c) < 20 or np.linalg.norm(c[0] - c[-1]) > 1.5:
            continue
        xy = c[:, ::-1]
        if abs(cv2.contourArea(xy.astype(np.float32))) < min_area_px:
            continue
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        n = max(int(seg[-1]), 8)
        s = np.linspace(0, seg[-1], n, endpoint=False)
        pts = np.c_[np.interp(s, seg, xy[:, 0]), np.interp(s, seg, xy[:, 1])]
        out.append(gaussian_filter1d(pts, sigma_px, axis=0, mode="wrap"))
    return out


# ---------------------------------------------------------------- primitive fitting

def _line_fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], float(np.abs((P - c) @ vt[1]).max())


def _circle_fit(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-12))
    return c, r, float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())


def _sweep(P, c) -> float:
    """Signed swept angle in degrees through P around c (positive = increasing atan2 angle)."""
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    return math.degrees(ang[-1] - ang[0])


def _fit_range(P, tol, depth=0):
    p0, p1 = P[0], P[-1]
    if len(P) <= 3:
        return [{"type": "line", "p0": p0.tolist(), "p1": p1.tolist(), "_n": len(P)}]
    c, d, dev = _line_fit(P)
    if dev <= tol:
        t = (P - c) @ d
        return [{"type": "line", "p0": (c + t[0] * d).tolist(), "p1": (c + t[-1] * d).tolist(), "_n": len(P)}]
    cc, r, devc = _circle_fit(P)
    sw = _sweep(P, cc)
    if devc <= tol and 0.2 < r < 500 and abs(sw) >= 8:
        return [{"type": "arc", "c": cc.tolist(), "r": r, "p0": p0.tolist(), "p1": p1.tolist(), "ccw": sw > 0,
                 "sweep_deg": abs(sw), "_n": len(P)}]
    chord = p1 - p0
    L = np.linalg.norm(chord)
    if depth > 14 or L < 1e-9:
        k = len(P) // 2
    else:
        k = int(np.argmax(np.abs(np.cross(chord / L, P - p0))))
        if k in (0, len(P) - 1):
            k = len(P) // 2
    return _fit_range(P[:k + 1], tol, depth + 1) + _fit_range(P[k:], tol, depth + 1)


def _refit_neighbours(prims, ring, tol):
    """Greedy: join neighbours when one line or arc still fits both runs of contour points within tol."""
    items, a = [], 0
    for p in prims:
        items.append(((a, a + p["_n"] - 1), p))
        a += p["_n"] - 1
    changed = True
    while changed and len(items) > 2:
        changed = False
        for k in range(len(items) - 1):
            cand = _fit_range(ring[items[k][0][0]:items[k + 1][0][1] + 1], tol, depth=99)
            if len(cand) == 1:
                items[k:k + 2] = [((items[k][0][0], items[k + 1][0][1]), cand[0])]
                changed = True
                break
    return [p for _, p in items]


def _fit_closed(P, tol):
    cc, r, dev = _circle_fit(P)
    if dev <= tol and r < 100 and abs(_sweep(np.vstack([P, P[:1]]), cc)) > 350:
        return [{"type": "circle", "c": cc.tolist(), "r": r}]
    P = np.roll(P, -int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1))), axis=0)
    i1 = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    ring = np.vstack([P, P[:1]])
    prims = _fit_range(ring[:i1 + 1], tol) + _fit_range(ring[i1:], tol)
    return [{k: v for k, v in p.items() if k != "_n"} for p in _refit_neighbours(prims, ring, tol)]


def sample_primitives(prims, step: float = 0.02) -> np.ndarray:
    """Points (mm) along the primitives, about `step` mm apart."""
    pts = []
    for p in prims:
        if p["type"] == "line":
            a, b = np.array(p["p0"]), np.array(p["p1"])
            pts.append(a + np.outer(np.linspace(0, 1, max(int(np.linalg.norm(b - a) / step), 2)), b - a))
        elif p["type"] == "arc":
            c, r = np.array(p["c"]), p["r"]
            a0 = math.atan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
            sw = math.radians(p["sweep_deg"]) * (1 if p["ccw"] else -1)
            t = a0 + np.linspace(0, sw, max(int(abs(sw) * r / step), 2))
            pts.append(c + r * np.c_[np.cos(t), np.sin(t)])
        else:
            c, r = np.array(p["c"]), p["r"]
            t = np.linspace(0, 2 * math.pi, max(int(2 * math.pi * r / step), 8), endpoint=False)
            pts.append(c + r * np.c_[np.cos(t), np.sin(t)])
    return np.vstack(pts)


def _length(p) -> float:
    if p["type"] == "line":
        return float(np.linalg.norm(np.subtract(p["p1"], p["p0"])))
    return p["r"] * math.radians(p["sweep_deg"]) if p["type"] == "arc" else 2 * math.pi * p["r"]


def _angle(p) -> float:
    d = np.subtract(p["p1"], p["p0"])
    return math.degrees(math.atan2(d[1], d[0]))


def _angle_diff(a, b) -> float:
    return abs((a - b + 90) % 180 - 90)      # lines have no direction: compare modulo 180


def _intersection(a, b):
    p, r = np.array(a["p0"]), np.subtract(a["p1"], a["p0"])
    q, s = np.array(b["p0"]), np.subtract(b["p1"], b["p0"])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    return p + (((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den) * r


def _circumcircle(a, b, c):
    den = 2 * np.linalg.det(np.array([b - a, c - a]))
    if abs(den) < 1e-9:
        return None
    ba, ca = b - a, c - a
    ctr = a + np.array([(ca[1] * (ba @ ba) - ba[1] * (ca @ ca)) / den, (ba[0] * (ca @ ca) - ca[0] * (ba @ ba)) / den])
    return ctr, float(np.linalg.norm(ctr - a))


def _arc_mid(p):
    c, r = np.array(p["c"]), p["r"]
    a0 = math.atan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
    half = math.radians(p["sweep_deg"]) / 2 * (1 if p["ccw"] else -1)
    return c + r * np.array([math.cos(a0 + half), math.sin(a0 + half)])


def _close_vertices(prims, max_move):
    """Neighbouring primitives share one vertex: line-line = intersection if near, else the midpoint; arcs are re-fitted
    through the shared endpoints and their old midpoint so the chain is exactly closed."""
    mids = {id(p): _arc_mid(p) for p in prims if p["type"] == "arc"}
    for k in range(len(prims)):
        a, b = prims[k - 1], prims[k]
        v = (np.array(a["p1"]) + np.array(b["p0"])) / 2
        if a["type"] == "line" and b["type"] == "line":
            x = _intersection(a, b)
            if x is not None and np.linalg.norm(x - v) < max_move:
                v = x
        a["p1"], b["p0"] = v.tolist(), v.tolist()
    for p in prims:
        if p["type"] != "arc":
            continue
        cc = _circumcircle(np.array(p["p0"]), mids[id(p)], np.array(p["p1"]))
        if cc is not None:
            p["c"], p["r"] = cc[0].tolist(), cc[1]
        c = np.array(p["c"])
        b0, b1 = (math.atan2(p[k][1] - c[1], p[k][0] - c[0]) for k in ("p0", "p1"))
        sw = math.degrees(b1 - b0)
        p["sweep_deg"] = abs(sw % 360 if p["ccw"] else -((-sw) % 360))
    return prims


def _sharpen_corners(prims, p: Params):
    """Replace a short run of small primitives between two long straight lines (a corner rounded by smoothing, or a tiny
    chamfer) by the sharp intersection of the two lines. Fillets and chamfers below corner_mm are not kept."""
    long_idx = [i for i, q in enumerate(prims) if q["type"] == "line" and _length(q) >= 1.5 * p.corner_mm]
    if len(long_idx) < 2:
        return prims
    drop = set()
    for a, b in zip(long_idx, long_idx[1:] + [long_idx[0] + len(prims)]):
        gap = [k % len(prims) for k in range(a + 1, b)]
        if not gap or sum(_length(prims[k]) for k in gap) > p.corner_mm:
            continue
        la, lb = prims[a], prims[b % len(prims)]
        x = _intersection(la, lb)
        mid = (np.array(la["p1"]) + np.array(lb["p0"])) / 2
        if x is None or np.linalg.norm(x - mid) > 0.8 * p.corner_mm:
            continue
        la["p1"], lb["p0"] = x.tolist(), x.tolist()
        drop.update(gap)
    return [q for i, q in enumerate(prims) if i not in drop]


def _drop_short(prims, min_len, min_radius):
    keep = [p for p in prims if p["type"] == "circle" or (_length(p) >= min_len and (p["type"] != "arc" or p["r"] >= min_radius))]
    return keep if len(keep) >= 3 else prims


def _merge_collinear(prims, p: Params):
    """Merge adjacent near-collinear lines and near-identical arcs if the merged primitive stays within merge_tol."""
    changed = True
    while changed and len(prims) > 3:
        changed = False
        for k in range(len(prims)):
            a, b = prims[k], prims[(k + 1) % len(prims)]
            if a["type"] == "line" and b["type"] == "line" and _angle_diff(_angle(a), _angle(b)) <= p.collinear_deg:
                new = {"type": "line", "p0": a["p0"], "p1": b["p1"]}
            elif a["type"] == "arc" and b["type"] == "arc" and a["ccw"] == b["ccw"] \
                    and np.linalg.norm(np.subtract(a["c"], b["c"])) < 0.3 and abs(a["r"] - b["r"]) < 0.3:
                cc = _circumcircle(np.array(a["p0"]), _arc_mid({**a, "p1": b["p1"], "sweep_deg": a["sweep_deg"] + b["sweep_deg"]}),
                                   np.array(b["p1"]))
                if cc is None:
                    continue
                sw = a["sweep_deg"] + b["sweep_deg"]
                new = {"type": "arc", "c": cc[0].tolist(), "r": cc[1], "p0": a["p0"], "p1": b["p1"], "ccw": a["ccw"], "sweep_deg": sw}
            else:
                continue
            old = sample_primitives([a, b], 0.05)
            if cKDTree(sample_primitives([new], 0.02)).query(old)[0].max() <= p.merge_tol_mm:
                j = (k + 1) % len(prims)
                prims[k] = new
                del prims[j]
                changed = True
                break
    return prims


def _snap(prims, axis_deg, contour_tree, p: Params):
    n = 0
    for q in prims:
        if q["type"] != "line":
            continue
        dev = ((_angle(q) - axis_deg + 45) % 90) - 45
        if abs(dev) > p.snap_deg:
            continue
        mid = (np.array(q["p0"]) + np.array(q["p1"])) / 2
        th = math.radians(-dev)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        q0, q1 = (mid + rot @ (np.array(q[k]) - mid) for k in ("p0", "p1"))
        if contour_tree.query(sample_primitives([{"type": "line", "p0": q0, "p1": q1}], 0.05))[0].max() > p.snap_tol_mm:
            continue
        q["p0"], q["p1"] = q0.tolist(), q1.tolist()
        n += 1
    return n


def _densify(P, step=0.02):
    ring = np.vstack([P, P[:1]])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    s = np.arange(0, seg[-1], step)
    return np.c_[np.interp(s, seg, ring[:, 0]), np.interp(s, seg, ring[:, 1])]


def _deviation(prims, contour):
    a, b = sample_primitives(prims), _densify(contour)
    d1, d2 = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    return float(max(d1.max(), d2.max())), float(np.percentile(np.r_[d1, d2], 95))


def _rect(P):
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(P, np.float32))
    return max(w, h), min(w, h)


def _outline_primitives(contour_mm, p: Params):
    prims = _fit_closed(contour_mm, p.fit_tol_mm)
    if prims[0]["type"] == "circle":
        return prims
    tree = cKDTree(_densify(contour_mm))
    prims = _sharpen_corners(prims, p)
    for _ in range(3):
        before = len(prims)
        prims = _close_vertices(_drop_short(prims, p.min_segment_mm, p.min_arc_radius_mm), p.vertex_move_mm + p.min_segment_mm)
        prims = _merge_collinear(prims, p)
        if len(prims) == before:
            break
    axis = cv2.minAreaRect(contour_mm.astype(np.float32))[2]
    _snap(prims, axis, tree, p)
    return _close_vertices(prims, p.vertex_move_mm)


# ---------------------------------------------------------------- holes and slots

def _holes(part_mask, blur, paper, exclude_keep_out, edge_thr, ppm, p: Params):
    """Enclosed paper-coloured regions inside the part. Returns [(contour_px, kind-free stats)] already validated."""
    d_ab = np.hypot(blur[..., 1] - paper[1], blur[..., 2] - paper[2])
    rel_l = blur[..., 0] / paper[0]
    inside = (cv2.erode(part_mask, _kernel(0.5 * ppm)) > 0) & ~exclude_keep_out
    ko = _kernel(p.hole_open_mm * ppm)
    loose = cv2.morphologyEx(((d_ab < p.paper_ab_loose) & (rel_l > p.paper_l_loose) & inside).astype(np.uint8) * 255, cv2.MORPH_OPEN, ko)
    strict = cv2.morphologyEx(((d_ab < p.paper_ab_strict) & (rel_l > p.paper_l_strict) & inside).astype(np.uint8) * 255, cv2.MORPH_OPEN, ko)
    n, cc, _, _ = cv2.connectedComponentsWithStats(loose)
    out = []
    for i in range(1, n):
        comp = (cc == i).astype(np.uint8) * 255
        if not (strict[comp > 0] > 0).any():
            continue                                  # no certain-paper seed: shadowed plastic or glints
        comp = _fill_external(comp)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float((comp > 0).sum())
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        if area * ppm ** -2 < p.hole_min_mm2 or area / max(hull, 1) < p.hole_solidity:
            continue
        if not _enclosed(comp, part_mask, ppm, p):
            continue
        core = cv2.erode(comp, _kernel(0.9 * ppm, ellipse=True)) > 0
        core = core if core.sum() > 50 else comp > 0
        pix = blur[core]
        med, sd = np.median(pix, axis=0), pix.std(axis=0)
        if med[0] / paper[0] < p.hole_min_rel_l or abs(med[1] - paper[1]) > p.hole_max_da \
                or med[2] - paper[2] > p.hole_max_db or sd[0] / paper[0] > p.hole_max_sd_l:
            continue                                  # not the paper's colour: glare, or dark plastic
        # put the edge where the outline's edge rule puts it (half the plastic's chroma), within a few px of the detection
        k = _kernel(0.4 * ppm, ellipse=True)
        band = (cv2.dilate(comp, k) > 0) & (cv2.erode(comp, k) == 0)
        grown = ((cv2.erode(comp, k) > 0) | (band & (d_ab < edge_thr) & inside)).astype(np.uint8) * 255
        out += _contours(_fill_external(grown), 1, 0.1 * ppm)
    return out


def _enclosed(comp, part_mask, ppm, p: Params, sectors: int = 24) -> bool:
    """Every direction around the region must have part just outside it (a ring 0.2 to 0.9 mm wide)."""
    ring = (cv2.dilate(comp, _kernel(0.9 * ppm * 2 + 1, ellipse=True)) > 0) & (cv2.dilate(comp, _kernel(0.22 * ppm * 2 + 1, ellipse=True)) == 0)
    ys, xs = np.nonzero(comp)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = np.nonzero(ring)
    sec = ((np.arctan2(ry - cy, rx - cx) + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors
    for k in range(sectors):
        sel = sec == k
        if sel.sum() > 5 and (part_mask[ry[sel], rx[sel]] > 0).mean() < p.hole_ring_min:
            return False
    return True


# ---------------------------------------------------------------- public API

def extract_outline(image: np.ndarray, part_mask: np.ndarray, ref_mask: np.ndarray, ref_mm: float | tuple[float, float], *,
                    ref_kind: str = "coin", exclude_mask: np.ndarray | None = None, params: Params = Params()) -> OutlineResult:
    """Outline of the part in `part_mask` (0/255, SAM) in mm.

    ref_kind "coin": ref_mm is the diameter. ref_kind "card": ref_mm is (long, short) in mm; card mode is only tested on
    synthetic images. exclude_mask (0/255) marks pixels that must never become part, e.g. a hand.
    Raises ValueError if the reference or the paper around the part cannot be used.
    """
    p = params
    if ref_kind == "coin":
        to_mm, to_px, mm_per_px, tilt, ref_err = _coin_frame(ref_mask, float(ref_mm))
    elif ref_kind == "card":
        to_mm, to_px, mm_per_px, tilt, ref_err = _card_frame(ref_mask, tuple(ref_mm))
    else:
        raise ValueError(f"unknown ref_kind {ref_kind!r}")
    ppm = 1.0 / mm_per_px
    part = (part_mask > 0).astype(np.uint8) * 255
    exclude = None if exclude_mask is None else (exclude_mask > 0).astype(np.uint8) * 255
    if exclude is not None:
        part = cv2.bitwise_and(part, cv2.bitwise_not(cv2.dilate(exclude, _kernel(2 * p.exclude_margin_mm * ppm + 1, ellipse=True))))
    warnings: list[str] = []
    if not part.any():
        return OutlineResult([], mm_per_px, tilt, ref_kind, 0.0, {"part": 0.0}, ["no part mask"])
    refined, blur, paper, keep_out, edge_thr = _refine(image, part, (ref_mask > 0).astype(np.uint8) * 255, exclude, ppm, p)
    contours = _contours(refined, p.min_part_mm2 * ppm ** 2, p.contour_smooth_mm * ppm)
    if not contours:
        return OutlineResult([], mm_per_px, tilt, ref_kind, 0.0, {"part": 0.0}, ["no outline found after refinement"])
    mm_contours = [to_mm(c) for c in contours]
    origin = np.vstack(mm_contours).min(0)
    shifted = [c - origin for c in mm_contours]
    frame = Frame(to_mm, to_px, origin)

    parts = []
    for c_px, c in zip(contours, shifted):
        prims = _outline_primitives(c, p)
        dmax, d95 = _deviation(prims, c)
        L, S = _rect(c)
        contact = 0.0
        if exclude is not None:
            dist = cv2.distanceTransform((exclude == 0).astype(np.uint8), cv2.DIST_L2, 3)
            near = dist[np.clip(np.round(c_px[:, 1]).astype(int), 0, dist.shape[0] - 1),
                        np.clip(np.round(c_px[:, 0]).astype(int), 0, dist.shape[1] - 1)] <= p.contact_mm * ppm
            contact = float(near.sum()) * mm_per_px
        parts.append(Part(prims, [], L, S, float(cv2.contourArea(c.astype(np.float32))), dmax, d95, contact))
    for hc in _holes(refined, blur, paper, keep_out, edge_thr, ppm, p):
        hm = to_mm(hc) - origin
        owner = next((i for i, c in enumerate(contours) if cv2.pointPolygonTest(c.astype(np.float32), tuple(hc.mean(0)), False) >= 0), None)
        if owner is None:
            continue
        L, S = _rect(hm)
        parts[owner].holes.append(Hole("slot" if L / max(S, 1e-6) > p.slot_elongation else "hole", L, S,
                                       float(cv2.contourArea(hm.astype(np.float32))), _fit_closed(hm, p.fit_tol_mm)))
    for part_ in parts:
        part_.holes.sort(key=lambda h: -h.area_mm2)

    comps = _quality(parts, refined, part, tilt, ref_err, exclude is not None, ref_kind, warnings)
    return OutlineResult(parts, mm_per_px, tilt, ref_kind, min(comps.values()), comps, warnings, frame)


def _quality(parts, refined, part_mask, tilt, ref_err, has_exclude, ref_kind, warnings) -> dict[str, float]:
    clip = lambda v: float(min(1.0, max(0.0, v)))
    inter = float(((refined > 0) & (_fill_external(part_mask) > 0)).sum())
    union = float(((refined > 0) | (_fill_external(part_mask) > 0)).sum())
    agree = inter / max(union, 1)
    comps = {
        "mask_agreement": clip((agree - 0.85) / 0.12),                       # IoU 0.97 and up = 1, 0.85 and below = 0
        "fit": clip(1 - (max(p.max_dev_mm for p in parts) - 0.1) / 0.2),    # max deviation 0.1 mm or less = 1, 0.3 mm = 0
        "reference": clip(1 - ref_err / 0.01) if ref_kind == "coin" else clip(1 - ref_err / 0.05),
    }
    if tilt is not None:
        comps["tilt"] = clip(1 - max(0.0, tilt - 3.0) / 17.0)                # 3 degrees or less = 1, 20 degrees = 0
        if tilt > 5.0:
            warnings.append(f"coin tilt {tilt:.1f} deg: scale is stretched along one direction")
    if has_exclude:
        total = sum(_perimeter(p) for p in parts)
        contact = sum(p.hand_contact_mm for p in parts)
        comps["hand"] = clip(1 - (contact / max(total, 1e-6)) / 0.3)         # no contact = 1, 30% of the outline = 0
        if contact > 0:
            warnings.append(f"hand contact along {contact:.1f} mm of the outline: that stretch follows the hand, not the part")
    if comps["fit"] < 1:
        warnings.append(f"max deviation {max(p.max_dev_mm for p in parts):.2f} mm is above 0.1 mm")
    return comps


def _perimeter(part: Part) -> float:
    return float(sum(_length(p) for p in part.primitives))
