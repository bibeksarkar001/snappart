"""Small features the visual hull cannot carve (openings in a face): find the face (RANSAC plane), rectify every view onto it with the known
camera poses, detect dark rectangular openings, triangulate them per video, fuse across videos, and cut pockets into the carve field.

Face frame: origin O on the plane, e1 horizontal along the face, e2 up the face, n outward. (s, t) are mm along e1, e2.
"""
import numpy as np
import cv2

from . import carve, snap


def candidate_faces(verts, faces, max_tilt_deg=15.0, near_mm=3.0, end_mm=20.0, seed=0, n_samples=40000):
    """Faces where openings may sit, at both ends of the footprint's long axis: per end, the largest near-vertical RANSAC plane (normal along
    the axis, outward) and every near-horizontal RANSAC plane, cropped to the last end_mm of the part. e1 runs along the end edge in every case."""
    P, N = snap.oriented_points(verts, faces, n_samples, seed=seed)       # 40k samples: same faces as 200k (v3 stage 3), ~5x faster
    pls = snap.planes(P, N, tol=0.12, max_planes=14, min_frac=0.005)
    box = cv2.boxPoints(cv2.minAreaRect(verts[:, :2].astype(np.float32)))
    sides = [box[1] - box[0], box[2] - box[1]]
    a2 = max(sides, key=np.linalg.norm)
    ax = np.r_[a2 / np.linalg.norm(a2), 0.0]                   # long axis of the footprint's min-area rectangle
    proj = verts @ ax
    out = []
    for sign, ext in ((1, proj.max()), (-1, proj.min())):
        out_ax = sign * ax
        vert = [(n * np.sign(n @ out_ax), m) for n, d, m in pls
                if np.degrees(np.arccos(min(abs(n @ ax), 1))) <= max_tilt_deg and abs(P[m].mean(0) @ ax - ext) <= near_mm]
        horiz = [(n, m & (np.abs(P @ ax - ext) <= end_mm)) for n, d, m in pls if np.degrees(np.arccos(min(abs(n[2]), 1))) <= max_tilt_deg]
        cands = ([max(vert, key=lambda q: q[1].sum())] if vert else []) + [h for h in horiz if h[1].sum() > 0.001 * len(P)]
        for n, m in cands:
            q = P[m]
            if np.degrees(np.arccos(min(abs(n[2]), 1))) <= max_tilt_deg:     # horizontal: outward = away from the part's mid-height
                n = n * (1 if (q.mean(0)[2] - verts[:, 2].mean()) * n[2] > 0 else -1)
                e1 = np.cross(n, out_ax)
            else:
                e1 = np.cross([0, 0, 1.0], n)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            d = -float(n @ q.mean(0))
            vt = verts @ e2 if abs(n[2]) < 0.5 else q @ e2
            s0, t0 = (q @ e1).min() - 2, vt.min() - 2
            O = q.mean(0) + (s0 - q.mean(0) @ e1) * e1 + (t0 - q.mean(0) @ e2) * e2
            out.append({"n": n, "d": d, "O": O, "e1": e1, "e2": e2, "len_s": float((q @ e1).max() + 2 - s0), "len_t": float(vt.max() + 2 - t0),
                        "inliers": int(m.sum()), "kind": "horizontal" if abs(n[2]) > 0.5 else "vertical", "end": sign,
                        "plane_rms": float(np.sqrt(((q @ n + d) ** 2).mean()))})
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


def detect_on_face(views, images, face, min_cos=0.3, ppm=20, cluster_mm=2.0, max_ray_mm=0.6, min_views=3):
    """Openings on one face. views: [(video, view in the part frame)], images: matching gray crops.

    Per video: detect on the rectified image (delta 0), cluster detections by s, triangulate each cluster from the rays through the detected
    centre (rays > max_ray_mm from the point dropped, then re-fitted) and through its 4 corners (axis-aligned rectangle in the face frame, corner
    k in the same face-frame quadrant in every view). The corners give the opening's own plane and size without assuming the face plane.
    Returns {video: [{"p", "corners" (4,3), "normal", "L", "S", "views", "res", "cos", "baseline_deg"}]}; a cluster needs min_views views."""
    quad = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    per = {}
    for (vid, v), img in zip(views, images):
        c = facing(v, face)
        if c < min_cos:
            continue
        g = rectify(v, img, face, 0.0, ppm)
        C = -v.R.T @ v.t
        for s, t, L, S in openings(g, ppm):
            pts = [face["O"] + (s + a * L / 2) * face["e1"] + (t + b * S / 2) * face["e2"] for a, b in ((0, 0), *quad)]
            dirs = [(P - C) / np.linalg.norm(P - C) for P in pts]
            per.setdefault(vid, []).append({"s": s, "view": v, "C": C, "d": dirs[0], "dc": dirs[1:], "cos": c})
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
            p, r = triangulate([(x["C"], x["d"]) for x in gr])
            keep = [x for x, ri in zip(gr, r) if ri <= max_ray_mm]
            if len({x["view"].name for x in keep}) >= min_views and len(keep) < len(gr):
                gr = keep
                p, r = triangulate([(x["C"], x["d"]) for x in gr])
            cor = np.array([triangulate([(x["C"], x["dc"][k]) for x in gr])[0] for k in range(4)])
            nrm = np.cross(cor[1] - cor[0], cor[3] - cor[0])
            nrm /= np.linalg.norm(nrm)
            res.append({"p": p, "res": r, "corners": cor, "normal": nrm * np.sign(nrm @ face["n"]),
                        "L": float((np.linalg.norm(cor[1] - cor[0]) + np.linalg.norm(cor[2] - cor[3])) / 2),
                        "S": float((np.linalg.norm(cor[3] - cor[0]) + np.linalg.norm(cor[2] - cor[1])) / 2),
                        "views": sorted({x["view"].name for x in gr}), "cos": float(np.mean([x["cos"] for x in gr])),
                        "baseline_deg": float(np.degrees(max(np.arccos(np.clip(a["d"] @ b["d"], -1, 1)) for a in gr for b in gr)))})
        out[vid] = res
    return out


def fuse(per_video, face, match_mm=1.5):
    """One record per opening, matched across videos by position along the face (e1). The primary video (most views) gives the centre, corners
    and size; every video's own numbers are kept in by_video. Sorted along e1."""
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
        prim = max(bv, key=lambda k: len(bv[k]["views"]))
        o = bv[prim]
        r.update({"primary": prim, "centre": o["corners"].mean(0), "corners": o["corners"], "L": o["L"], "S": o["S"], "normal": o["normal"],
                  "dist_from_face_mm": float(o["corners"].mean(0) @ face["n"] + face["d"]), "n_views": sum(len(x["views"]) for x in bv.values())})
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


def step_band(g, ppm, t_lo, t_hi, s_lo, s_hi, min_len=8.0, min_step=0.25, w_lo=None, w_hi=None):
    """A raised rail along e1 in a rectified gray image: per column, the row of the strongest vertical-gradient edge inside the band
    t_lo..t_hi gives an edge line t(s); a 3-segment piecewise-constant fit (least absolute deviation, breakpoints on a 0.25 mm grid) finds
    where the edge steps. The middle segment is the rail if it differs from BOTH sides by >= min_step mm in the same direction.
    Returns {"s0", "s1" (ends, mm), "t_edge" (middle segment), "step" (mm, middle minus sides), "width" (mm, between the two strongest
    opposite-polarity edges across the middle segment, rows w_lo..w_hi, default t_lo..t_hi)}. The ends are refined to the strongest
    horizontal-gradient column within 1 mm of each breakpoint (gradient averaged over the band rows). None if no such step."""
    H = g.shape[0]
    r0, r1 = int(H - 1 - t_hi * ppm), int(H - 1 - t_lo * ppm)
    c0, c1 = int(s_lo * ppm), int(s_hi * ppm)
    if r0 < 0 or r1 >= H or c1 - c0 < 4 * ppm:
        return None
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)[r0:r1, c0:c1]
    rows = np.abs(gy).argmax(0)
    t = (H - 1 - (r0 + rows)) / ppm
    from scipy.ndimage import median_filter
    t = median_filter(t, int(ppm) | 1)
    s = (c0 + np.arange(c1 - c0)) / ppm
    step = int(0.25 * ppm)
    cuts = np.arange(step, len(s) - step, step)
    cs = np.r_[0, np.cumsum(t)]
    best = None
    for i in cuts:
        for j in cuts[cuts > i + min_len * ppm]:
            segs = [t[:i], t[i:j], t[j:]]
            if min(len(x) for x in segs) < ppm:
                continue
            meds = [np.median(x) for x in segs]
            cost = sum(np.abs(x - m).sum() for x, m in zip(segs, meds))
            if best is None or cost < best[0]:
                best = (cost, i, j, meds)
    if best is None:
        return None
    _, i, j, (ma, mb, mc) = best
    if not ((mb - ma) * (mb - mc) > 0 and min(abs(mb - ma), abs(mb - mc)) >= min_step):
        return None
    gx = np.abs(cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)[r0:r1, c0:c1]).mean(0)
    win = int(ppm)
    i = max(i - win, 0) + int(np.argmax(gx[max(i - win, 0):i + win]))
    j = max(j - win, 0) + int(np.argmax(gx[max(j - win, 0):j + win]))
    q0, q1 = (r0, r1) if w_lo is None else (int(H - 1 - w_hi * ppm), int(H - 1 - w_lo * ppm))
    prof = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)[q0:q1, c0 + i:c0 + j].mean(1)
    a, b = int(np.argmax(prof)), int(np.argmin(prof))
    return {"s0": float(s[i]), "s1": float(s[j]), "t_edge": float(mb), "step": float(mb - (ma + mc) / 2), "width": abs(a - b) / ppm,
            "t_a": float((H - 1 - (q0 + a)) / ppm), "t_b": float((H - 1 - (q0 + b)) / ppm), "fit_mad": float(best[0] / len(t))}


def find_grooves(v, fc, vv, imgs, log=lambda m: None, max_angle_deg=15.0, faces=None, pers=None):
    """Openings on the candidate face whose triangulated corner planes agree with it (within max_angle_deg) and that has the most view detections.
    faces / pers: candidate faces and their detect_on_face results when computed elsewhere (in parallel).
    Returns (face, fused openings, per-video detections, per-face table)."""
    table, found = [], None
    faces = candidate_faces(v, fc) if faces is None else faces
    for fi, face in enumerate(faces):
        per = detect_on_face(vv, imgs, face) if pers is None else pers[fi]
        ops = fuse(per, face)
        nview = sum(o["n_views"] for o in ops)
        ang = float(np.degrees(np.arccos(np.clip(np.mean([o["normal"] for o in ops], 0) @ face["n"] / np.linalg.norm(np.mean([o["normal"] for o in ops], 0)), -1, 1)))) if ops else None
        row = {"face": fi, "kind": face["kind"], "n": face["n"].tolist(), "centre_x": float(face["O"][0] + face["len_s"] / 2 * face["e1"][0] + face["len_t"] / 2 * face["e2"][0]),
               "inliers": face["inliers"], "rms": face["plane_rms"], "openings_per_video": {k: len(o) for k, o in per.items()}, "fused": len(ops), "view_detections": nview, "corner_plane_angle_deg": ang}
        table.append(row)
        log(f"face #{fi} ({face['kind']}, near x={row['centre_x']:.0f}): n=({face['n'][0]:+.3f},{face['n'][1]:+.3f},{face['n'][2]:+.3f}), {face['inliers']} samples, rms {face['plane_rms']:.3f} mm, "
            f"{sum(facing(x, face) >= 0.3 for _, x in vv)} views facing it; openings (>= 3 views in one video): " + (", ".join(f"{k[-2:]}: {len(o)}" for k, o in per.items()) or "none")
            + (f"; fused {len(ops)}, {nview} view detections, triangulated corner planes {ang:.1f} deg from this plane" if ops else ""))
        if ops and ang < max_angle_deg and (found is None or nview > found[3]):
            found = (face, ops, per, nview)
    return (found[:3] if found else (None, [], {})) + (table,)


def groove_boxes(ops, face, depth_mm, out_mm=3.0):
    """Pocket boxes: the triangulated opening rectangle (centre, long axis from its corners), axis = the face normal, depth_mm into the part
    from the opening plane and out_mm outward so the pocket breaks through whatever hull material lies outside the opening plane."""
    out = []
    for o in ops:
        c = o["corners"]
        lng = (c[1] - c[0]) + (c[2] - c[3])
        lng -= face["n"] * (lng @ face["n"])
        lng /= np.linalg.norm(lng)
        axes = np.stack([lng, np.cross(face["n"], lng), face["n"]])
        out.append((o["centre"] + face["n"] * (out_mm - depth_mm) / 2, axes, np.array([o["L"] / 2, o["S"] / 2, (depth_mm + out_mm) / 2])))
    return out


def plane_sweep(views, images, face_at, zs, ppm=10, blur=4.0, boot=20, seed=0):
    """Photo-consistency plane sweep: for each offset z, rectify every view onto face_at(z) and score the mean per-pixel variance of the
    high-passed, normalised images (low = the views agree = a textured surface at that plane). Returns (best z, bootstrap sd, scores)."""
    stacks = []
    for z in zs:
        R = []
        for v, im in zip(views, images):
            r = rectify(v, im, face_at(z), 0, ppm).astype(np.float32)
            hp = r - cv2.GaussianBlur(r, (0, 0), blur)
            R.append(hp / (hp.std() + 1e-6))
        stacks.append(np.stack(R))
    sc = np.array([s.var(0).mean() for s in stacks])
    rng = np.random.default_rng(seed)
    best = [zs[int(np.argmin([s[k].var(0).mean() for s in stacks]))] for k in (rng.choice(len(views), len(views)) for _ in range(boot))]
    return float(zs[int(np.argmin(sc))]), float(np.std(best)), sc


def groove_features(ops, face):
    """Measured openings as model features: centre, axes rows (long, short, outward normal), opening size, view count."""
    out = []
    for (c, axes, half), o in zip(groove_boxes(ops, face, 0.0, 0.0), ops):
        out.append({"centre": o["centre"].tolist(), "axes": axes.tolist(), "long_mm": float(o["L"]), "short_mm": float(o["S"]), "views": int(o["n_views"])})
    return out


def detect_bump(vv, imgs, face, delta, min_cos=0.5, ppm=20):
    """Raised block above the openings (v2 stage 2 method): per view, a 3-segment step on the strongest edge 0.3..3.2 mm above the openings' top
    edge; views agree if their length is within 1 mm of their video's median; the ends are triangulated in the video with the most agreeing views.
    Returns None or {"ends", "length_mm", "video", "agree", "found", "ridge_to_opening_mm"}."""
    per = {}
    base = face["O"] + delta * face["n"]
    for (vid, v), im in zip(vv, imgs):
        if facing(v, face) < min_cos:
            continue
        g = rectify(v, im, face, delta, ppm)
        oo = openings(g, ppm)
        if len(oo) < 3:
            continue
        top = float(np.median([o[1] + o[3] / 2 for o in oo]))
        r = step_band(g, ppm, top + 0.3, top + 3.2, min(o[0] - o[2] / 2 for o in oo), max(o[0] + o[2] / 2 for o in oo), w_lo=top - 0.3, w_hi=top + 3.2)
        if r is not None:
            r.update(top=top, len=r["s1"] - r["s0"], C=-v.R.T @ v.t, name=v.name, cos=facing(v, face))
            per.setdefault(vid, []).append(r)
    best = None
    for vid, lst in per.items():
        med = float(np.median([r["len"] for r in lst]))
        ok = [r for r in lst if abs(r["len"] - med) <= 1.0]
        if len(ok) >= 3 and (best is None or len(ok) > len(best[1])):
            best = (vid, ok, len(lst))
    if best is None:
        return None
    vid, ok, nf = best
    ends = []
    for key in ("s0", "s1"):
        rays = []
        for r in ok:
            P = base + r[key] * face["e1"] + r["t_edge"] * face["e2"]
            rays.append((r["C"], (P - r["C"]) / np.linalg.norm(P - r["C"])))
        ends.append(triangulate(rays)[0])
    return {"ends": [e.tolist() for e in ends], "length_mm": float(np.linalg.norm(ends[1] - ends[0])), "video": vid, "agree": len(ok), "found": nf,
            "ridge_to_opening_mm": float(np.median([r["t_edge"] - r["top"] for r in ok])),
            "per_view": [{"name": r["name"], "cos": r["cos"], "step_mm": r["step"], "edge_to_opening_mm": r["t_edge"] - r["top"], "len_mm": r["len"]} for r in ok]}


def ridge_from_steps(bump, face, min_cos=0.9):
    """Where the ridge stands, from the edge steps in near-head-on views (cos >= min_cos, so the rectified step is the ridge's extent
    projected along the face normal): within the detected span the visible edge lies `step` mm further from the openings (along e2) than
    beside it. So the face beside the span ends at the far edge minus step*e2 (the wall the ridge stands on) and the ridge reaches step mm
    out of that wall along e2. Returns {"wall_line" (2 points), "extent_mm", "extent_sd_mm", "views", "normal" (= e2)} or None."""
    st = [p["step_mm"] for p in bump["per_view"] if p["cos"] >= min_cos]
    if len(st) < 3:
        return None
    e2 = np.asarray(face["e2"], float)
    ext = float(np.median(st))
    return {"wall_line": [(np.asarray(e) - ext * e2).tolist() for e in bump["ends"]], "extent_mm": ext, "extent_sd_mm": float(np.std(st)),
            "extent_range_mm": [float(min(st)), float(max(st))], "views": len(st), "normal": e2.tolist()}


def bump_feature(bump, grooves):
    """Model feature for the bump: detected far-edge line (midpoint, axis), direction toward the openings, outward normal of the groove face."""
    e0, e1 = np.array(bump["ends"][0]), np.array(bump["ends"][1])
    n = np.array(grooves[0]["axes"][2])
    a = e1 - e0
    a -= n * (a @ n)
    a /= np.linalg.norm(a)
    mid = (e0 + e1) / 2
    w = np.mean([g["centre"] for g in grooves], 0) - mid
    w -= a * (w @ a) + n * (w @ n)
    w /= np.linalg.norm(w)
    return {"edge_mid": mid.tolist(), "axis": a.tolist(), "toward_openings": w.tolist(), "normal": n.tolist(), "detected_length_mm": bump["length_mm"],
            "ends": bump["ends"], "source_video": bump["video"], "agreeing_views": bump["agree"]}


def plate_region(field, g, bump_ends):
    """Plate footprint from the carve: x from the window's far edge (material along y = part centre at z = 6) to the bump's detected far edge;
    y between the rails' inner faces (gap at 5 mm before the window edge)."""
    ix = lambda v, a: int(round((v - g.lo[a]) / g.vs))
    yc = float(np.mean([e[1] for e in bump_ends]))
    zc = ix(6.0, 2)
    xs = g.lo[0] + np.where(field[:, ix(yc, 1), zc] > 0)[0] * g.vs
    x1 = min(e[0] for e in bump_ends)
    gaps = np.where(np.diff(xs) > 2 * g.vs)[0]
    xw = xs[gaps[-1] + 1] if len(gaps) else xs.min()               # first material after the last gap (the window) before the bump
    ys = g.lo[1] + np.where(field[ix(xw - 5, 0), :, zc] > 0)[0] * g.vs
    y0, y1 = ys[ys < yc].max() + g.vs, ys[ys > yc].min() - g.vs
    return float(xw), float(x1), float(y0), float(y1)


def plate_faces(vv, imgs, region, videos_by_side, zs=None, min_cos=0.6):
    """Both faces of a horizontal plate by photo-consistency plane sweeps: {side: {"z", "boot_sd", "views", "video"}}. videos_by_side maps
    "lower" / "upper" to the video that sees that face."""
    x0, x1, y0, y1 = region
    zs = np.arange(6.0, 12.01, 0.1) if zs is None else zs

    def face_z(z, up):
        return {"O": np.array([x0 + 1, y0 + 1.5, z]), "e1": np.array([0, 1.0, 0]), "e2": np.array([1.0, 0, 0]), "n": np.array([0, 0, 1.0 if up else -1.0]),
                "len_s": y1 - y0 - 3, "len_t": x1 - x0 - 3}
    out = {}
    for side, vid in videos_by_side.items():
        up = side == "upper"
        sel = [(v, im) for (n, v), im in zip(vv, imgs) if n == vid and facing(v, face_z(10, up)) > min_cos]
        if len(sel) < 5:
            continue
        z, sd, _ = plane_sweep([v for v, _ in sel], [im for _, im in sel], lambda z: face_z(z, up), zs)
        out[side] = {"z": z, "boot_sd": sd, "views": len(sel), "video": vid}
    return out


def face_rms(v, fc, box, up, n=200000):
    """Roughness: RMS distance (mm) of the mesh's surface samples in box (x0, x1, y0, y1; inset 0.5 mm), facing up (+z) or down, to their own
    least-squares plane. Returns (rms, samples, mean z, tilt of the plane from horizontal in degrees)."""
    P, N = snap.oriented_points(v, fc, n)
    m = (P[:, 0] > box[0] + 0.5) & (P[:, 0] < box[1] - 0.5) & (P[:, 1] > box[2] + 0.5) & (P[:, 1] < box[3] - 0.5) & ((N[:, 2] > 0.5) if up else (N[:, 2] < -0.5))
    q = P[m]
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][2]
    r = (q - c) @ n
    return float(np.sqrt((r ** 2).mean())), int(m.sum()), float(c[2]), float(np.degrees(np.arccos(abs(n[2]))))
