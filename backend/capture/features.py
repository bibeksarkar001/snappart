"""Small features the visual hull cannot carve (openings in a face): find the face (RANSAC plane), rectify every view onto it with the known
camera poses, detect dark rectangular openings, triangulate them per video, fuse across videos, and cut pockets into the carve field.

Face frame: origin O on the plane, e1 horizontal along the face, e2 up the face, n outward. (s, t) are mm along e1, e2.
"""
import numpy as np
import cv2

from . import carve, snap


def end_faces(verts, faces, max_tilt_deg=15.0, near_mm=3.0, seed=0):
    """Near-vertical RANSAC planes at both ends of the footprint's long axis: the largest one per end, normal pointing outward."""
    P, N = snap.oriented_points(verts, faces, 200000, seed=seed)
    pls = snap.planes(P, N, tol=0.12, max_planes=14, min_frac=0.005)
    box = cv2.boxPoints(cv2.minAreaRect(verts[:, :2].astype(np.float32)))
    sides = [box[1] - box[0], box[2] - box[1]]
    a2 = max(sides, key=np.linalg.norm)
    ax = np.r_[a2 / np.linalg.norm(a2), 0.0]                   # long axis of the footprint's min-area rectangle
    proj = verts @ ax
    out = []
    for sign, ext in ((1, proj.max()), (-1, proj.min())):
        best = None
        for n, d, m in pls:
            n = n * (1 if n @ ax * sign > 0 else -1)
            d = -n @ P[m].mean(0)
            if np.degrees(np.arccos(min(abs(n @ ax), 1))) > max_tilt_deg or abs(P[m].mean(0) @ ax - ext) > near_mm:
                continue
            if best is None or m.sum() > best[2].sum():
                best = (n, d, m)
        if best is None:
            continue
        n, d, m = best
        e1 = np.cross([0, 0, 1.0], n)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        q = P[m]
        span = q @ e1
        O = q.mean(0) - (q.mean(0) @ e1 - span.min() + 2) * e1
        O = O - (O[2] - verts[:, 2].min() + 2) * e2 / e2[2]     # t = 0 two mm below the part's lowest point
        out.append({"n": n, "d": float(d), "O": O, "e1": e1, "e2": e2, "len_s": float(span.max() - span.min() + 4), "len_t": float(verts[:, 2].max() - verts[:, 2].min() + 4),
                    "inliers": int(m.sum()), "x_extreme": float(ext), "plane_rms": float(np.sqrt(((q @ n + d) ** 2).mean()))})
    return out


def facing(view, face):
    """Cosine between the face normal and the direction to the camera, at the face centre."""
    C = -view.R.T @ view.t
    P = face["O"] + face["len_s"] / 2 * face["e1"] + face["len_t"] / 2 * face["e2"]
    return float((C - P) @ face["n"] / np.linalg.norm(C - P))


def grid(face, delta, ppm):
    W, H = int(face["len_s"] * ppm), int(face["len_t"] * ppm)
    s, t = np.meshgrid(np.arange(W) / ppm, (H - 1 - np.arange(H)) / ppm)
    return face["O"] + delta * face["n"] + s[..., None] * face["e1"] + t[..., None] * face["e2"]


def rectify(view, img, face, delta=0.0, ppm=20):
    """The view's image resampled onto the face plane moved by delta mm along its normal: row 0 = top (largest t)."""
    X = grid(face, delta, ppm)
    H, W = X.shape[:2]
    u, v, _ = carve.project(view, X.reshape(-1, 3))
    return cv2.remap(img, u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32), cv2.INTER_LINEAR)


def openings(g, ppm, min_len=3.0, max_len=15.0, min_w=1.0, max_w=5.0, t_tol=0.6):
    """Dark rectangles in a rectified gray image: (s, t, long, short) in mm. Candidates darker than 0.8x the local background, then each one's
    edge is re-thresholded (Otsu in a window 1 mm around it) so the size is taken at the dark/light transition. Detections that are not in one
    row (|t - median t| > t_tol) are dropped."""
    g = g.astype(np.float32)
    H, W = g.shape
    cand = (g < cv2.GaussianBlur(g, (0, 0), 3 * ppm) * 0.8).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, _, st, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    for i in range(1, nlab):
        x, y, w, h, a = st[i]
        if not (min_len * min_w * ppm * ppm * 0.5 < a < max_len * max_w * ppm * ppm and w > 1.5 * h):
            continue
        x0, y0, x1, y1 = max(x - ppm, 0), max(y - ppm, 0), min(x + w + ppm, W), min(y + h + ppm, H)
        win = cv2.GaussianBlur(g[y0:y1, x0:x1], (0, 0), 1).astype(np.uint8)
        th, _ = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nl, lb, st2, _ = cv2.connectedComponentsWithStats((win < th).astype(np.uint8))
        if nl < 2:
            continue
        j = 1 + int(np.argmax(st2[1:, 4]))
        cs, _ = cv2.findContours((lb == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(cs[0])
        L, S = max(rw, rh) / ppm, min(rw, rh) / ppm
        if min_len <= L <= max_len and min_w <= S <= max_w:
            out.append((( x0 + cx) / ppm, (H - 1 - (y0 + cy)) / ppm, L, S))
    if len(out) > 1:
        tm = np.median([o[1] for o in out])
        out = [o for o in out if abs(o[1] - tm) <= t_tol]
    return sorted(out)


def triangulate(rays):
    """Least-squares intersection of rays [(C, unit d)]: point and per-ray distances (mm)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for C, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    p = np.linalg.solve(A, b)
    return p, [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - C))) for C, d in rays]


def ray_plane(C, d, n, off):
    """Point where ray C + l d meets the plane n.x + off = 0."""
    return C + (-(off + n @ C) / (n @ d)) * d


def detect_on_face(views, images, face, min_cos=0.3, ppm=20, cluster_mm=2.0, max_ray_mm=0.6, min_views=3):
    """Openings on one face. views: [(video, view in the part frame)], images: matching gray crops.

    Per video: detect on the rectified image (delta 0), cluster detections by s, triangulate each cluster from the rays through the detected
    centres (dropping rays > max_ray_mm from the point, then re-fitting); a cluster needs min_views views. Returns {video: [{"p": 3D, "rays": n, "res": [...], "views": [...], "cos": mean}]}."""
    per = {}
    for (vid, v), img in zip(views, images):
        c = facing(v, face)
        if c < min_cos:
            continue
        g = rectify(v, img, face, 0.0, ppm)
        C = -v.R.T @ v.t
        for s, t, L, S in openings(g, ppm):
            P = face["O"] + s * face["e1"] + t * face["e2"]
            per.setdefault(vid, []).append({"s": s, "t": t, "L": L, "S": S, "view": v, "C": C, "d": (P - C) / np.linalg.norm(P - C), "cos": c})
    out = {}
    for vid, dets in per.items():
        dets.sort(key=lambda q: q["s"])
        groups = []
        for q in dets:
            if groups and q["s"] - np.mean([x["s"] for x in groups[-1]]) < cluster_mm:
                groups[-1].append(q)
            else:
                groups.append([q])
        res = []
        for gr in groups:
            if len({x["view"].name for x in gr}) < min_views:
                continue
            rays = [(x["C"], x["d"]) for x in gr]
            p, r = triangulate(rays)
            keep = [x for x, ri in zip(gr, r) if ri <= max_ray_mm]
            if len({x["view"].name for x in keep}) >= min_views and len(keep) < len(gr):
                p, r = triangulate([(x["C"], x["d"]) for x in keep])
                gr = keep
            res.append({"p": p, "res": r, "views": sorted({x["view"].name for x in gr}), "dets": gr, "cos": float(np.mean([x["cos"] for x in gr])),
                        "baseline_deg": float(np.degrees(max(np.arccos(np.clip(a["d"] @ b["d"], -1, 1)) for a in gr for b in gr)))})
        out[vid] = res
    return out


def fuse(per_video, face, match_mm=1.5):
    """One record per opening seen in several videos. Lateral position (e1): mean over videos. Depth along the normal: from the video with the
    widest ray baseline. Height (e2): rays of the most frontal video met with the plane at that depth. Sizes from the views of that video,
    rectified at that depth (done by the caller). Returns a list sorted by e1 with every video's own numbers kept."""
    recs = []
    for vid, ops in per_video.items():
        for o in ops:
            e1 = float(o["p"] @ face["e1"])
            hit = next((r for r in recs if abs(r["e1"] - e1) < match_mm), None)
            if hit is None:
                hit = {"e1": e1, "by_video": {}}
                recs.append(hit)
            hit["by_video"][vid] = o
            hit["e1"] = float(np.mean([x["p"] @ face["e1"] for x in hit["by_video"].values()]))
    for r in recs:
        bv = r["by_video"]
        wide = max(bv, key=lambda k: bv[k]["baseline_deg"])
        front = max(bv, key=lambda k: bv[k]["cos"])
        depth = float(bv[wide]["p"] @ face["n"] + face["d"])           # signed distance of the opening plane from the RANSAC face (outward +)
        off = face["d"] - depth
        hits = [ray_plane(x["C"], x["d"], face["n"], off) for x in bv[front]["dets"]]
        r.update({"depth_from_face_mm": depth, "depth_video": wide, "height_video": front, "centre": np.mean(hits, 0),
                  "e2_by_video": {k: float(np.mean([ray_plane(x["C"], x["d"], face["n"], off) @ face["e2"] for x in o["dets"]])) for k, o in bv.items()}})
    return sorted(recs, key=lambda r: r["e1"])


def box_sdf(X, c, axes, half):
    """Signed distance (mm, positive outside) of points X to the oriented box centre c, unit axes (3x3 rows), half extents."""
    q = np.abs((X - c) @ axes.T) - half
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def cut_pockets(field, g, boxes):
    """Remove material inside each box from an inside-positive carve field: f = min(f, sdf_box)."""
    X = g.points()
    f = field.ravel().copy()
    for c, axes, half in boxes:
        f = np.minimum(f, box_sdf(X, c, axes, half).astype(np.float32))
    return f.reshape(field.shape)


def render_shaded(verts, faces, view, light=(0.3, -0.4, -1.0)):
    """Painter's-algorithm shaded render of a mesh in a view (part frame): BGR image and coverage mask. Lambert shading in camera space."""
    c = verts @ view.R.T + view.t
    z = np.maximum(c[:, 2], 1e-6)
    uv = np.stack([view.K[0, 0] * c[:, 0] / z + view.K[0, 2], view.K[1, 1] * c[:, 1] / z + view.K[1, 2]], 1)
    w, h = view.size
    tri = c[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    front = nrm[:, 2] < 0                                     # facing the camera (camera looks along +z)
    L = np.asarray(light, float) / np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * np.abs(nrm @ L), 0, 1)
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    pts = np.round(uv[faces] * 16).astype(np.int32)
    for i in np.argsort(-tri[:, :, 2].mean(1)):
        if not front[i]:
            continue
        col = (np.array([90, 140, 230]) * shade[i]).astype(int).tolist()
        cv2.fillConvexPoly(img, pts[i], col, cv2.LINE_8, 4)
        cv2.fillConvexPoly(mask, pts[i], 255, cv2.LINE_8, 4)
    return img, mask
