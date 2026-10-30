"""Small-detail candidates: holes, slots, bosses, chamfers / fillets, steps, and engraved or embossed text / marks.

Two sources:
  mesh   plane and cylinder RANSAC on the carved mesh (snap.py): steps (parallel faces at different heights), chamfers (narrow faces at
         30-60 deg between two large ones), holes (concave cylinders), bosses and fillets (convex cylinders; a fillet covers less than half
         a turn and has a small radius). Only what silhouettes can carve shows here (through-holes, outer shapes).
  image  views rectified onto a face (features.rectify): dark openings (holes if roundish, slots if long) and edge-dense areas that are
         not openings (text or a mark). These see blind pockets and engravings that silhouettes cannot carve, but not their depth.
Every candidate gets a size, a count of the evidence (views or surface points) and a status: measured (size >= 1 mm, seen in >= 3 views
or by >= 120 of 40k surface samples), question (above 1 mm but uncertain: asked with a crop, "is this a hole or a mark?"), or dropped (under 1 mm,
learnings rule 6). Depths of image-only openings are not_detected (the camera cannot see them). Nothing is added to the model here."""
import cv2
import numpy as np

from . import learnings as L, snap

MIN_POINTS = 120                 # surface samples (of 40k) on a primitive before it counts as measured
MIN_VIEWS = 3


def _status(size_mm, evidence, kind_min):
    if size_mm < L.MIN_FEATURE_MM:
        return None
    return "measured" if evidence >= kind_min else "question"


def mesh_candidates(verts, faces, n_samples=40000, seed=0) -> list[dict]:
    P, N = snap.oriented_points(verts, faces, n_samples, seed)
    tri = verts[faces]
    area_total = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    per_pt = area_total / len(P)
    pls = snap.planes(P, N, tol=0.15, max_planes=14, min_frac=0.004, iters=600, seed=seed)
    pls = [(n, d, m) if N[m].mean(0) @ n >= 0 else (-n, -d, m) for n, d, m in pls]     # outward: RANSAC's sign is arbitrary
    out = []
    # steps: faces with the same outward normal at different offsets
    used = set()
    for i, (n1, d1, m1) in enumerate(pls):
        for j, (n2, d2, m2) in enumerate(pls):
            if j <= i or (i, j) in used or n1 @ n2 < 0.985:
                continue
            h = abs(d1 - d2)
            st = _status(h, min(m1.sum(), m2.sum()), MIN_POINTS)
            if st and h > 2 * 0.15:
                c = (P[m1].mean(0) + P[m2].mean(0)) / 2
                out.append({"kind": "step", "source": "mesh", "size_mm": {"height": round(float(h), 2)}, "centre": c.round(2).tolist(),
                            "normal": n1.round(3).tolist(), "evidence": {"points": int(min(m1.sum(), m2.sum()))}, "status": st})
                used.add((i, j))
    # chamfers: a narrow face at 30-60 deg to two faces at least twice its size that are themselves roughly perpendicular
    tilt = lambda a, b: np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))
    for n, d, m in pls:
        bigger = [nb for nb, _, mb in pls if mb.sum() >= 2 * m.sum()]
        pair = any(30 <= tilt(n, a) <= 60 and 30 <= tilt(n, b) <= 60 and 70 <= tilt(a, b) <= 110 for i, a in enumerate(bigger) for b in bigger[i + 1:])
        if pair:
            q = P[m] - P[m].mean(0)
            w = float(np.sqrt(12 * np.linalg.eigvalsh(q.T @ q / len(q))[1]))       # width across the narrow direction
            st = _status(w, m.sum(), MIN_POINTS)
            if st:
                out.append({"kind": "chamfer", "source": "mesh", "size_mm": {"width": round(w, 2)}, "centre": P[m].mean(0).round(2).tolist(),
                            "normal": n.round(3).tolist(), "evidence": {"points": int(m.sum())}, "status": st})
    # cylinders on what the planes left
    avail = np.ones(len(P), bool)
    for _, _, m in pls:
        avail &= ~m
    for cyl in snap.cylinders(P, N, rmin=0.5, rmax=15.0, tol=0.12, max_cyl=6, iters=6000, min_inliers=MIN_POINTS, seed=seed, avail=avail):
        m = cyl["inliers"]
        v = P[m] - cyl["centre"]
        rad = v - np.outer(v @ cyl["axis"], cyl["axis"])
        rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-12
        convex = float((N[m] * rad).sum(1).mean()) > 0                            # normals point away from the axis
        e1 = np.cross(cyl["axis"], [1.0, 0, 0] if abs(cyl["axis"][0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        ang = np.sort(np.degrees(np.arctan2(rad @ np.cross(cyl["axis"], e1), rad @ e1)) % 360)
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        cover = 360 - float(gaps.max())
        dia = 2 * cyl["r"]
        kind = "hole" if not convex else ("fillet" if cover < 180 and cyl["r"] < 3.0 else "boss")
        size = {"diameter": round(dia, 2), "length": round(cyl["length"], 2)} if kind != "fillet" else {"radius": round(cyl["r"], 2)}
        st = _status(dia if kind != "fillet" else cyl["r"] * 2, m.sum(), MIN_POINTS)
        if st:
            out.append({"kind": kind, "source": "mesh", "size_mm": size, "centre": cyl["centre"].round(2).tolist(), "axis": cyl["axis"].round(3).tolist(),
                        "arc_deg": round(cover, 0), "evidence": {"points": int(m.sum()), "rms_mm": round(cyl["rms"], 3)}, "status": st})
    return out


def dark_openings(gray, ppm, min_mm=0.5) -> list[dict]:
    """Dark blobs in a rectified face image (pixels/mm = ppm): centre (s, t mm), long and short size (mm), roundness."""
    g = gray.astype(np.float32)
    cand = (g < cv2.GaussianBlur(g, (0, 0), 3 * ppm) * 0.7).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    H = g.shape[0]
    for i in range(1, n):
        if st[i, 4] < (min_mm * ppm) ** 2:
            continue
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (w, h), _ = cv2.minAreaRect(cs[0])
        lo, sh = max(w, h) / ppm, min(w, h) / ppm
        fill = st[i, 4] / max(w * h, 1)
        out.append({"s": cx / ppm, "t": (H - 1 - cy) / ppm, "long_mm": lo, "short_mm": sh, "fill": float(fill),
                    "kind": "hole" if lo / max(sh, 1e-6) < 1.4 and fill < 0.9 else "slot"})
    return out


def text_areas(gray, ppm, openings, cell_mm=3.0, factor=4.0) -> list[dict]:
    """Edge-dense cells of a rectified face (gradient energy > factor x the face's median) not covered by an opening: text or a mark."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.7)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    c = max(int(cell_mm * ppm), 4)
    H, W = e.shape
    cells = e[:H // c * c, :W // c * c].reshape(H // c, c, W // c, c).mean((1, 3))
    hot = (cells > factor * max(float(np.median(cells)), 1.0)).astype(np.uint8)
    yc, xc = np.mgrid[0:hot.shape[0], 0:hot.shape[1]] * c + c / 2                # cell centres (px)
    for o in openings:                                                          # edges of an opening are not text
        reach = o["long_mm"] * ppm / 2 + c / 2
        hot[(np.abs(yc - (H - 1 - o["t"] * ppm)) <= reach) & (np.abs(xc - o["s"] * ppm) <= reach)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 2:
            continue
        out.append({"kind": "text_or_mark", "s": (x + w / 2) * c / ppm, "t": (H - 1 - (y + h / 2) * c) / ppm,
                    "size_mm": {"width": round(w * c / ppm, 1), "height": round(h * c / ppm, 1)}})
    return out


def cluster_views(per_view: list[list[dict]], match_mm=1.5) -> list[dict]:
    """Merge one face's per-view detections (same s, t within match_mm) into candidates with a view count and median size."""
    groups = []
    for dets in per_view:
        for d in dets:
            g = next((g for g in groups if abs(g["s"] - d["s"]) < match_mm and abs(g["t"] - d["t"]) < match_mm and g["kind"] == d["kind"]), None)
            if g is None:
                groups.append({**d, "items": [d]})
            else:
                g["items"].append(d)
                g["s"], g["t"] = float(np.mean([x["s"] for x in g["items"]])), float(np.mean([x["t"] for x in g["items"]]))
    out = []
    for g in groups:
        views = len(g["items"])
        if g["kind"] == "text_or_mark":
            size = {k: float(np.median([x["size_mm"][k] for x in g["items"]])) for k in ("width", "height")}
            st = _status(max(size.values()), views, 10 ** 9)                   # text is always asked about: its depth is not visible
        else:
            size = {"long": round(float(np.median([x["long_mm"] for x in g["items"]])), 2), "short": round(float(np.median([x["short_mm"] for x in g["items"]])), 2)}
            st = _status(size["short"], views, MIN_VIEWS)
        if st:
            out.append({"kind": g["kind"], "source": "image", "s": round(g["s"], 2), "t": round(g["t"], 2), "size_mm": size,
                        "evidence": {"views": views}, "status": st,
                        "not_detected": {"depth": "the camera cannot see how deep it goes"}})
    return out
