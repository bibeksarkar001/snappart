"""Phase 1 spike: CAD-ready outlines from the cached SAM masks of the private lock_set photos. No API calls.

1. smooth the cached SAM mask (it is a stretched low-res raster), then refine only a band around its edge by chroma
   (Lab a*/b* distance from the paper: plastic is coloured, shadow is grey)
2. sub-pixel contour (skimage find_contours), smoothed, un-stretched with the coin ellipse, scaled to mm
3. split into straight lines and circular arcs (fit tolerance 0.07 mm, aiming at 0.1 mm after closing), snap near-axis lines, close the vertices
4. holes and slots = enclosed regions inside the part whose colour matches the paper (not inner contours of the mask)
Runs for coin = 25.0 mm and 23.0 mm. Reads out/lock_set/*, writes only out/lock_set/refined/ (git-ignored).

Usage: python backend/spikes/outline_refine.py [--only lock_single_top_03 ...]
"""
import argparse
import json
import math
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from spike_common import OUT, ROOT, unstretch_fn

SRC = OUT / "lock_set"
R = SRC / "refined"
SEND = R / "SEND_TO_CLAUDE"
PHOTOS = ROOT / "assets" / "private" / "lock_set"
COINS = {"25.0": 25.0, "23.0": 23.0}
COMPARE = ["lock_pair_sep_front_01", "lock_pair_clipped_top_01", "lock_pair_clipped_top_04",
           "lock_single_top_03", "lock_edge_01", "lock_edge_03"]
PAIRS = [("lock_single_top_03", "lock_single_top_04"), ("lock_single_top_05", "lock_single_top_06")]

BAND_PX = 15              # refine this far either side of the SAM edge
SM_SIGMA_PX = 4           # smoothing of the stretched SAM raster before the band is cut
MIN_PART_PX = 6000
FIT_TOL_MM = 0.07         # fit tolerance; the 0.1 mm asked for is the budget AFTER snapping and vertex closing, which add up to ~0.05
CONTOUR_SIGMA_PX = 2.5    # contour smoothing along the arc (1 px spacing)
SNAP_DEG = 2.0            # lines this close to the main axis (or its perpendicular) are snapped to it
MIN_HOLE_MM2 = 3.0   # judged once at the 25.0 scale, so both coin runs report the same openings
SLOT_ELONGATION = 2.0
STRICT_AB, STRICT_L = 4.5, 0.5   # certain paper: chroma within this of the paper's and lightness above this fraction of it
LOOSE_AB, LOOSE_L = 8.0, 0.3    # paper in light shadow; only kept where it connects to certain paper
OPEN_PX = 9                      # opening that cuts glint speckle on the glossy faces away from real openings
SOLIDITY_MIN = 0.8               # area / convex hull area; ragged glint clusters fail this


# ---------- mask refinement ----------

def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)  # OpenCV 8-bit Lab: a and b are offset by 128


def paper_ref(lab, part, coin):
    """Median Lab of the paper in a ring 60 to 260 px around the part (and clear of the coin)."""
    ring = (cv2.dilate(part, np.ones((521, 521), np.uint8)) > 0) & (cv2.dilate(part, np.ones((121, 121), np.uint8)) == 0)
    ring &= cv2.dilate(coin, np.ones((141, 141), np.uint8)) == 0
    return np.median(lab[ring], axis=0)


def refine_mask(lab, sam, band=BAND_PX):
    """Keep the SAM interior, decide only the +-band px around its edge by chroma. Returns (mask, chroma threshold)."""
    sm = (cv2.GaussianBlur(sam.astype(np.float32) / 255, (0, 0), SM_SIGMA_PX) > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    core, outer = cv2.erode(sm, k), cv2.dilate(sm, k)
    in_band = (outer > 0) & (core == 0)
    return sm, core, in_band


def chroma_distance(lab, ref):
    blur = cv2.GaussianBlur(lab, (0, 0), 1.2)
    return np.hypot(blur[..., 1] - ref[1], blur[..., 2] - ref[2]), blur


def build_part_mask(img, sam, coin):
    lab = to_lab(img)
    ref = paper_ref(lab, sam, coin)
    d_ab, blur = chroma_distance(lab, ref)
    sm, core, in_band = refine_mask(lab, sam)
    plastic = float(np.median(d_ab[cv2.erode(sm, np.ones((61, 61), np.uint8)) > 0]))
    thr = max(6.0, 0.5 * plastic)  # half of the plastic's own chroma = where the edge sits
    mask = ((core > 0) | (in_band & (d_ab > thr))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_PART_PX:
            keep[lab_cc == i] = 255
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(keep)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)  # interior holes are found by colour later, not kept from the mask
    return out, {"lab": lab, "blur": blur, "ref": ref, "chroma_thr": thr, "plastic_chroma": plastic}


def filled(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def iou(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


# ---------- contours and primitive fitting ----------

def subpixel_contours(mask, min_area=MIN_PART_PX):
    """Closed sub-pixel contours (x, y) of a 0/255 mask, resampled to 1 px spacing and smoothed along the arc."""
    out = []
    for c in find_contours(cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.5), 0.5):
        if len(c) < 20 or np.linalg.norm(c[0] - c[-1]) > 1.5:
            continue
        xy = c[:, ::-1]
        if abs(cv2.contourArea(xy.astype(np.float32))) < min_area:
            continue
        out.append(smooth_closed(xy))
    return out


def smooth_closed(xy, step=1.0, sigma=CONTOUR_SIGMA_PX):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = max(int(seg[-1] / step), 8)
    s = np.linspace(0, seg[-1], n, endpoint=False)
    pts = np.c_[np.interp(s, seg, xy[:, 0]), np.interp(s, seg, xy[:, 1])]
    return gaussian_filter1d(pts, sigma / step, axis=0, mode="wrap")


def line_fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d, n = vt[0], vt[1]
    return c, d, float(np.abs((P - c) @ n).max())


def circle_fit(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-12))
    return c, r, float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())


def sweep(P, c):
    """Signed angle (degrees, + = counter-clockwise in a y-up frame) swept going through P around centre c."""
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    return math.degrees(ang[-1] - ang[0])


def fit_range(P, tol, depth=0):
    """Split an open run of points into lines and arcs. Returns [primitive dicts]."""
    p0, p1 = P[0], P[-1]
    if len(P) <= 3:
        return [{"type": "line", "p0": p0.tolist(), "p1": p1.tolist(), "_n": len(P)}]
    c, d, dev = line_fit(P)
    if dev <= tol:
        t = (P - c) @ d
        return [{"type": "line", "p0": (c + t[0] * d).tolist(), "p1": (c + t[-1] * d).tolist(), "_n": len(P)}]
    cc, r, devc = circle_fit(P)
    sw = sweep(P, cc)
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
    return fit_range(P[:k + 1], tol, depth + 1) + fit_range(P[k:], tol, depth + 1)


def fit_closed(P, tol):
    """Closed contour -> primitives. A contour that is one circle becomes a circle primitive."""
    cc, r, dev = circle_fit(P)
    if dev <= tol and r < 100 and abs(sweep(np.vstack([P, P[:1]]), cc)) > 350:
        return [{"type": "circle", "c": cc.tolist(), "r": r, "_n": len(P)}]
    i0 = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
    P = np.roll(P, -i0, axis=0)
    i1 = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    ring = np.vstack([P, P[:1]])
    prims = fit_range(ring[:i1 + 1], tol) + fit_range(ring[i1:], tol)
    return merge(prims, ring, tol)


def merge(prims, ring, tol):
    """Greedy: join neighbours when one line/arc still fits both runs within tol."""
    ranges, a = [], 0
    for p in prims:
        ranges.append((a, a + p["_n"] - 1))
        a += p["_n"] - 1
    items = list(zip(ranges, prims))
    changed = True
    while changed and len(items) > 2:
        changed = False
        for k in range(len(items) - 1):
            (a0, _), _ = items[k]
            (_, b1), _ = items[k + 1]
            cand = fit_range(ring[a0:b1 + 1], tol, depth=99)
            if len(cand) == 1:
                items[k:k + 2] = [((a0, b1), cand[0])]
                changed = True
                break
    return [p for _, p in items]


def seg_angle(p):
    d = np.subtract(p["p1"], p["p0"])
    return math.degrees(math.atan2(d[1], d[0]))


def snap_lines(prims, axis_deg, contour_mm, snap=SNAP_DEG):
    """Rotate lines within snap degrees of the main axis (or its perpendicular) onto it, but only where the rotated line
    still lies within FIT_TOL_MM of the contour. Returns (snapped, rejected)."""
    n, rejected = 0, 0
    tree = cKDTree(densify(contour_mm))
    for p in prims:
        if p["type"] != "line":
            continue
        dev = ((seg_angle(p) - axis_deg + 45) % 90) - 45
        if abs(dev) <= snap:
            mid = (np.array(p["p0"]) + np.array(p["p1"])) / 2
            th = math.radians(-dev)
            rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            q0 = (mid + rot @ (np.array(p["p0"]) - mid)).tolist()
            q1 = (mid + rot @ (np.array(p["p1"]) - mid)).tolist()
            if tree.query(sample_prims([{"type": "line", "p0": q0, "p1": q1}], 0.05))[0].max() > FIT_TOL_MM:
                rejected += 1
                continue
            p["p0"], p["p1"], p["snapped"] = q0, q1, True
            n += 1
    return n, rejected


def line_intersection(a, b):
    p, r = np.array(a["p0"]), np.subtract(a["p1"], a["p0"])
    q, s = np.array(b["p0"]), np.subtract(b["p1"], b["p0"])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    return p + t * r


def close_vertices(prims):
    """Make neighbouring primitives share one vertex: line-line = intersection (if near), otherwise the midpoint."""
    n = len(prims)
    for k in range(n):
        a, b = prims[k - 1], prims[k]
        mid = (np.array(a["p1"]) + np.array(b["p0"])) / 2
        v = mid
        if a["type"] == "line" and b["type"] == "line":
            x = line_intersection(a, b)
            if x is not None and np.linalg.norm(x - mid) < 0.3:
                v = x
        a["p1"], b["p0"] = v.tolist(), v.tolist()
    for p in prims:
        if p["type"] == "arc":
            # re-fit the circle through the moved endpoints and the old arc's midpoint, so the chain is exactly closed
            c, r = np.array(p["c"]), p["r"]
            a0 = math.atan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
            half = math.radians(p["sweep_deg"]) / 2 * (1 if p["ccw"] else -1)
            mid = c + r * np.array([math.cos(a0 + half), math.sin(a0 + half)])
            cc = circumcircle(np.array(p["p0"]), mid, np.array(p["p1"]))
            if cc is not None:
                p["c"], p["r"] = cc[0].tolist(), cc[1]
                c = cc[0]
            b0, b1 = (math.atan2(p[k][1] - c[1], p[k][0] - c[0]) for k in ("p0", "p1"))
            sw = math.degrees(b1 - b0)
            p["sweep_deg"] = abs(sw % 360 if p["ccw"] else -((-sw) % 360))
    return prims


def circumcircle(a, b, c):
    m = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]])
    den = 2 * np.linalg.det(m)
    if abs(den) < 1e-9:
        return None
    ba, ca = b - a, c - a
    ux = (ca[1] * (ba @ ba) - ba[1] * (ca @ ca)) / den
    uy = (ba[0] * (ca @ ca) - ca[0] * (ba @ ba)) / den
    ctr = a + np.array([ux, uy])
    return ctr, float(np.linalg.norm(ctr - a))


def sample_prims(prims, step=0.02):
    pts = []
    for p in prims:
        if p["type"] == "line":
            a, b = np.array(p["p0"]), np.array(p["p1"])
            m = max(int(np.linalg.norm(b - a) / step), 2)
            pts.append(a + np.outer(np.linspace(0, 1, m), b - a))
        elif p["type"] == "arc":
            c, r = np.array(p["c"]), p["r"]
            a0 = math.atan2(p["p0"][1] - c[1], p["p0"][0] - c[0])
            sw = math.radians(p["sweep_deg"]) * (1 if p["ccw"] else -1)
            m = max(int(abs(sw) * r / step), 2)
            t = a0 + np.linspace(0, sw, m)
            pts.append(c + r * np.c_[np.cos(t), np.sin(t)])
        else:
            c, r = np.array(p["c"]), p["r"]
            m = max(int(2 * math.pi * r / step), 8)
            t = np.linspace(0, 2 * math.pi, m, endpoint=False)
            pts.append(c + r * np.c_[np.cos(t), np.sin(t)])
    return np.vstack(pts)


def densify(P, step=0.02):
    ring = np.vstack([P, P[:1]])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    s = np.arange(0, seg[-1], step)
    return np.c_[np.interp(s, seg, ring[:, 0]), np.interp(s, seg, ring[:, 1])]


def deviation(prims, contour_mm):
    """Max distance (mm) between the primitive chain and the smoothed contour, both directions."""
    a, b = sample_prims(prims), densify(contour_mm)
    d1 = cKDTree(b).query(a)[0]
    d2 = cKDTree(a).query(b)[0]
    return float(max(d1.max(), d2.max())), float(np.percentile(np.r_[d1, d2], 95))


def rect_dims(P):
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(P, np.float32))
    return max(w, h), min(w, h)


def clean(prims):
    return [{k: v for k, v in p.items() if k != "_n"} for p in prims]


# ---------- holes and slots by paper colour ----------

def find_holes(part_mask, info, mm_per_px_25):
    """Enclosed paper-coloured regions inside the part (hysteresis on Lab). Returns closed sub-pixel contours in px (x, y)."""
    blur, ref = info["blur"], info["ref"]
    d_ab = np.hypot(blur[..., 1] - ref[1], blur[..., 2] - ref[2])
    rel_l = blur[..., 0] / ref[0]
    inside = cv2.erode(part_mask, np.ones((7, 7), np.uint8)) > 0
    k_open = np.ones((OPEN_PX, OPEN_PX), np.uint8)
    loose = cv2.morphologyEx(((d_ab < LOOSE_AB) & (rel_l > LOOSE_L) & inside).astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open)
    strict = cv2.morphologyEx(((d_ab < STRICT_AB) & (rel_l > STRICT_L) & inside).astype(np.uint8) * 255, cv2.MORPH_OPEN, k_open)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(loose)
    out = []
    for i in range(1, n):
        comp = (cc == i).astype(np.uint8) * 255
        if not (strict[comp > 0] > 0).any():
            continue  # no certain-paper seed: shadowed plastic or glints, not an opening
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp = np.zeros_like(comp)
        cv2.drawContours(comp, cnts, -1, 255, cv2.FILLED)
        area_px = float((comp > 0).sum())
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        if area_px * mm_per_px_25 ** 2 < MIN_HOLE_MM2 or area_px / max(hull, 1) < SOLIDITY_MIN:
            continue
        out += subpixel_contours(comp, min_area=1)
    return out


# ---------- per image ----------

def process(stem):
    t0 = time.perf_counter()
    img = cv2.imread(str(PHOTOS / f"{stem}.jpg"))
    sam = cv2.imread(str(SRC / f"{stem}_part_sam.png"), 0)
    coin = cv2.imread(str(SRC / f"{stem}_coin.png"), 0)
    fix, major, minor, tilt, ell = unstretch_fn(coin)
    mask, info = build_part_mask(img, sam, coin)
    contours = subpixel_contours(mask)
    holes_px = find_holes(mask, info, 25.0 / major)
    rec = {"image": stem, "tilt_deg": tilt, "coin_major_px": major, "chroma_thr": info["chroma_thr"],
           "plastic_chroma": info["plastic_chroma"], "iou_refined_vs_sam": iou(mask, filled(sam)), "by_coin": {}}
    fit_px = None
    for name, mm in COINS.items():
        s = mm / major
        parts, devs, nprim, snapped, rejected = [], [], 0, 0, 0
        outlines_mm = [fix(c) * s for c in contours]
        for oc in outlines_mm:
            prims = fit_closed(oc, FIT_TOL_MM)
            axis = cv2.minAreaRect(oc.astype(np.float32))[2]
            sn, rj = snap_lines(prims, axis, oc)
            snapped, rejected = snapped + sn, rejected + rj
            prims = close_vertices(prims)
            dmax, d95 = deviation(prims, oc)
            L, S = rect_dims(oc)
            devs.append((dmax, d95))
            nprim += len(prims)
            parts.append({"primitives": clean(prims), "long_mm": L, "short_mm": S, "area_mm2": float(cv2.contourArea(oc.astype(np.float32))),
                          "max_dev_mm": dmax, "p95_dev_mm": d95, "holes": []})
        for hc in holes_px:
            hm = fix(hc) * s
            owner = next((i for i, c in enumerate(contours) if cv2.pointPolygonTest(c.astype(np.float32), tuple(hc.mean(0)), False) >= 0), None)
            if owner is None:
                continue
            L, S = rect_dims(hm)
            prims = fit_closed(hm, FIT_TOL_MM)
            parts[owner]["holes"].append({"kind": "slot" if L / max(S, 1e-6) > SLOT_ELONGATION else "hole", "long_mm": L, "short_mm": S,
                                          "area_mm2": float(cv2.contourArea(hm.astype(np.float32))),
                                          "primitives": clean(prims)})
        allpts = np.vstack(outlines_mm)
        L, S = rect_dims(cv2.convexHull(allpts.astype(np.float32)).reshape(-1, 2))
        holes = [h for p in parts for h in p["holes"]]
        rec["by_coin"][name] = {"px_per_mm": major / mm, "n_parts": len(parts), "long_mm": L, "short_mm": S, "n_primitives": nprim,
                                "n_snapped": snapped, "n_snap_rejected": rejected, "max_dev_mm": max(d[0] for d in devs), "p95_dev_mm": max(d[1] for d in devs),
                                "n_holes": sum(h["kind"] == "hole" for h in holes), "n_slots": sum(h["kind"] == "slot" for h in holes),
                                "parts": parts}
        if name == "25.0":
            fit_px = (fix, s)
    # IoU of the fitted outline (primitives rasterised) vs SAM
    fix25, s25 = fit_px
    ras = np.zeros_like(sam)
    for p in rec["by_coin"]["25.0"]["parts"]:
        poly = to_px(sample_prims(p["primitives"], 0.05), fix25, s25)
        cv2.fillPoly(ras, [np.round(poly).astype(np.int32)], 255)
    rec["iou_fit_vs_sam"] = iou(ras, filled(sam))
    rec["runtime_s"] = time.perf_counter() - t0
    return rec, img, (fix25, s25), coin


def to_px(pts_mm, fix, s):
    return fix.inv(np.asarray(pts_mm) / s)


# ---------- drawing ----------

COL = {"line": (0, 0, 255), "arc": (0, 200, 0), "circle": (255, 0, 255), "hole": (255, 200, 0)}


def draw(img, rec, fix, s, coin_mask, width):
    """Fitted primitives (and holes) over the photo, cropped to the part + coin and scaled to `width` px wide."""
    pts = [to_px(sample_prims(p["primitives"], 0.2), fix, s) for p in rec["by_coin"]["25.0"]["parts"]]
    cy, cx = np.nonzero(coin_mask)
    allp = np.vstack(pts + [np.array([[cx.min(), cy.min()], [cx.max(), cy.max()]])])
    h, w = img.shape[:2]
    x0, y0 = np.maximum(allp.min(0) - 120, 0).astype(int)
    x1, y1 = np.minimum(allp.max(0) + 120, [w, h]).astype(int)
    left = img[y0:y1, x0:x1].copy()
    right = left.copy()
    scale = width / (x1 - x0)
    th = max(2, int(round(2.5 / scale)))
    def poly(P, col, t=th):
        cv2.polylines(right, [np.round((P - [x0, y0]) * 16).astype(np.int32)], False, col, t, cv2.LINE_AA, shift=4)
    for part in rec["by_coin"]["25.0"]["parts"]:
        for p in part["primitives"]:
            poly(to_px(sample_prims([p], 0.05), fix, s), COL[p["type"]])
            for k in ("p0", "p1"):
                if k in p:
                    cv2.circle(right, tuple(np.round(to_px([p[k]], fix, s)[0] - [x0, y0]).astype(int)), th * 2, (255, 255, 255), -1)
        for hole in part["holes"]:
            for p in hole["primitives"]:
                poly(to_px(sample_prims([p], 0.03), fix, s), COL["hole"], th)
    left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return left, right


def hole_list(rec, coin):
    return sorted((h for p in rec["by_coin"][coin]["parts"] for h in p["holes"]), key=lambda h: -h["area_mm2"])


def pair_check(a, b, rec_a, rec_b, lines):
    lines.append(f"{a.replace('lock_', '')} vs {b.replace('lock_', '')}")
    for coin in COINS:
        ha, hb = hole_list(rec_a, coin), hole_list(rec_b, coin)
        A, B = rec_a["by_coin"][coin], rec_b["by_coin"][coin]
        same = (A["n_holes"], A["n_slots"]) == (B["n_holes"], B["n_slots"])
        diffs = [max(abs(x["long_mm"] - y["long_mm"]), abs(x["short_mm"] - y["short_mm"])) for x, y in zip(ha, hb)]
        worst = max(diffs) if diffs else float("nan")
        ok = same and diffs and worst <= 0.15
        lines.append(f"  coin {coin}: counts {A['n_holes']}h/{A['n_slots']}s vs {B['n_holes']}h/{B['n_slots']}s "
                     f"({'identical' if same else 'DIFFERENT'}); sizes (long x short mm): "
                     + " | ".join(f"{x['long_mm']:.2f}x{x['short_mm']:.2f} vs {y['long_mm']:.2f}x{y['short_mm']:.2f}" for x, y in zip(ha, hb))
                     + f"; worst size diff {worst:.2f} mm -> {'within 0.15 mm' if ok else 'NOT within 0.15 mm'}"
                     + f"; outline {A['long_mm']:.2f}x{A['short_mm']:.2f} vs {B['long_mm']:.2f}x{B['short_mm']:.2f}")


def finalize(recs):
    old = json.loads((SRC / "results.json").read_text())
    L = []
    P = L.append
    P(f"{'image':<26}{'prims':>6}{'maxdev':>8}{'p95dev':>8}{'IoU ref/fit':>13}{'h/s':>7}  {'LxS mm @25':<13}{'LxS mm @23':<13}{'snap ok/rej':>13}{'s':>6}")
    for k, r in recs.items():
        a, b = r["by_coin"]["25.0"], r["by_coin"]["23.0"]
        P(f"{k:<26}{a['n_primitives']:>6}{a['max_dev_mm']:>8.3f}{a['p95_dev_mm']:>8.3f}{r['iou_refined_vs_sam']:>7.3f}/{r['iou_fit_vs_sam']:<5.3f}"
          f"{a['n_holes']:>3}h/{a['n_slots']}s  {a['long_mm']:>5.1f}x{a['short_mm']:<6.1f} {b['long_mm']:>5.1f}x{b['short_mm']:<6.1f}{a['n_snapped']:>7}/{a['n_snap_rejected']:<5}{r['runtime_s']:>6.1f}")
    P("\nmax dev = largest distance (mm, both directions) between the final primitives (after snapping and vertex closing) and the")
    P("smoothed sub-pixel contour they were fitted to. Asked for 0.1 mm; fitting uses 0.07 mm so the closed chain lands near 0.1. p95 = 95th percentile. IoU ref = refined mask vs SAM mask")
    P("(both with interior holes filled, so it compares outlines); IoU fit = fitted outline rasterised vs the same SAM mask. holes/slots = enclosed")
    P("paper-coloured regions (>= 2 mm2, solid shape), slot = long/short > 2. Counts and sizes below are at coin 25.0 unless stated.")
    P("\n== repeat-shot consistency (holes/slots must match, sizes within 0.15 mm) ==")
    for a, b in PAIRS:
        if a in recs and b in recs:
            pair_check(a, b, recs[a], recs[b], L)
    P("\n== what changed vs the first run (lock_set_test.py: HSV colour mask, inner contours of the mask) ==")
    P(f"{'image':<26}{'first LxS @25':<16}{'now LxS @25':<14}{'first h/s':>10}{'now h/s':>9}{'first IoU raw/filled':>22}{'now IoU ref/fit':>17}")
    for k, r in recs.items():
        o = old[k]["by_coin"]["25.0"]
        a = r["by_coin"]["25.0"]
        P(f"{k:<26}{o['long_mm']:>6.1f}x{o['short_mm']:<8.1f}{a['long_mm']:>6.1f}x{a['short_mm']:<7.1f}{o['holes']:>6}h/{o['slots']}s{a['n_holes']:>5}h/{a['n_slots']}s"
          f"{old[k]['iou_raw']:>14.3f}/{old[k]['iou_filled']:<6.3f}{r['iou_refined_vs_sam']:>8.3f}/{r['iou_fit_vs_sam']:<5.3f}")
    P("\nmethod changes: outline now comes from the SAM edge refined by Lab chroma in a +-15 px band (shadow is grey, plastic is coloured), as a")
    P("sub-pixel smoothed contour fitted with lines/arcs, not the pixel mask; holes/slots now come from paper colour inside the part, not inner")
    P("contours of the mask. Not fixed / not verified: no calipers comparison; perspective and part height are not corrected; hand_* photos have")
    P("no cached coin mask, so they are not processed here.")
    text = "\n".join(L)
    (SEND / "summary.txt").write_text(text)
    (SEND / "results.json").write_text(json.dumps(recs, indent=1, default=float))
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    R.mkdir(exist_ok=True)
    SEND.mkdir(exist_ok=True)
    stems = args.only or sorted(p.stem for p in PHOTOS.glob("lock_*.jpg") if "hand" not in p.stem)
    recs = {}
    for stem in stems:
        rec, img, (fix, s), coin = process(stem)
        recs[stem] = rec
        left, right = draw(img, rec, fix, s, coin, 1200)
        cv2.imwrite(str(R / f"{stem}_fit.png"), right)
        if stem in COMPARE:
            l2, r2 = draw(img, rec, fix, s, coin, 800)
            cv2.imwrite(str(SEND / f"{stem}_compare.png"), np.hstack([l2, r2]))
        b = rec["by_coin"]["25.0"]
        print(f"{stem}: prims={b['n_primitives']} dev={b['max_dev_mm']:.3f} iou={rec['iou_refined_vs_sam']:.3f}/{rec['iou_fit_vs_sam']:.3f} "
              f"h/s={b['n_holes']}/{b['n_slots']} {b['long_mm']:.1f}x{b['short_mm']:.1f} t={rec['runtime_s']:.1f}s", flush=True)
    if not args.only:
        finalize(recs)
        print(f"\nSEND_TO_CLAUDE: {SEND}")
        for f in sorted(SEND.iterdir()):
            print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
