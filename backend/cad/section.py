"""Horizontal section of a binary STL: the holes (inner loops) at height z. Used to check the solid against the detected holes."""
import cv2
import numpy as np


def read_stl(path) -> np.ndarray:
    """Triangles as an (n, 3, 3) array, from a binary STL."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    rec = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    return rec["v"].astype(float)


def _loops(tris, z):
    segs = []
    for t in tris:
        d = t[:, 2] - z
        pts = []
        for i in range(3):
            a, b, da, db = t[i], t[(i + 1) % 3], d[i], d[(i + 1) % 3]
            if (da < 0) != (db < 0):
                pts.append(a + (b - a) * (da / (da - db)))
        if len(pts) == 2:
            segs.append((pts[0][:2], pts[1][:2]))
    key = lambda p: tuple(np.round(p, 4))
    nxt = {}
    for a, b in segs:
        nxt.setdefault(key(a), []).append(b)
        nxt.setdefault(key(b), []).append(a)
    seen, loops = set(), []
    for a, b in segs:
        if key(a) in seen:
            continue
        loop, cur, prev = [a], b, key(a)
        seen.add(key(a))
        while key(cur) != key(a) and key(cur) not in seen:
            seen.add(key(cur))
            loop.append(cur)
            cands = [q for q in nxt[key(cur)] if key(q) != prev]
            if not cands:
                break
            prev, cur = key(cur), cands[0]
        loops.append(np.array(loop))
    return loops


def section_holes(path, z) -> list[dict]:
    """Inner loops of the section at z: centre (x, y up), long/short side of the min-area rectangle, area. Needs a watertight STL."""
    loops = [l for l in _loops(read_stl(path), z) if len(l) >= 3]
    holes = []
    for i, l in enumerate(loops):
        depth = sum(cv2.pointPolygonTest(o.astype(np.float32), tuple(map(float, l[0])), False) > 0 for j, o in enumerate(loops) if j != i)
        if depth % 2 == 1:
            (_, _), (w, h), _ = cv2.minAreaRect(l.astype(np.float32))
            m = cv2.moments(l.astype(np.float32))
            holes.append({"cx": m["m10"] / m["m00"], "cy": m["m01"] / m["m00"], "long": max(w, h), "short": min(w, h),
                          "area": float(cv2.contourArea(l.astype(np.float32)))})
    return holes
