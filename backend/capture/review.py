""""Review your model": find regions of a carved model a user should look at, and turn each into a review item with a photo crop, a render
of the same region from the same camera, and a question with choices.

Flags (per mesh vertex, then grouped into connected surface regions):
  lump        deviates from the plane fitted to its neighbourhood (same-facing samples within LUMP_RADIUS) by more than LUMP_RMS (rms)
  few_views   fixed by fewer than MIN_VIEWS silhouettes (a silhouette edge within 0.3 mm)
  disagree    one video's silhouettes put the surface more than DISAGREE_MM further in than the model (its 3rd strongest view)
  one_video   a detected feature that only one video found
Thresholds live in config.py (REVIEW_*). Nothing here calls a model; the user's answers become corrections (corrections.py).
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import carve, config as C, render as Rn

QUESTIONS = {
    "lump": ("This area looks bumpy. On the real part, is it a flat surface?", ["Keep", "Remove", "Make flat"]),
    "few_views": ("The camera barely saw this area. Is the shape here right?", ["Keep", "Remove", "Make flat"]),
    "disagree": ("Two videos disagree about this area. Is the shape here right?", ["Keep", "Remove", "Make flat"]),
    "one_video": ("Only one video found this feature. Is it on this side of the part?", ["Current side", "Opposite side", "Remove"]),
}


def vertex_normals(v, fc):
    n = np.zeros_like(v)
    fn = np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]])
    for k in range(3):
        np.add.at(n, fc[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def lumpiness(v, nv, idx, radius, cos_min=0.8):
    """Per sample vertex: rms distance of its same-facing neighbours (normal within acos(cos_min), within radius) to their least-squares plane."""
    tree = cKDTree(v)
    out = np.zeros(len(idx))
    for j, i in enumerate(idx):
        nb = tree.query_ball_point(v[i], radius)
        nb = np.asarray(nb)[nv[nb] @ nv[i] > cos_min]
        if len(nb) < 12:
            continue
        q = v[nb] - v[nb].mean(0)
        out[j] = np.sqrt(np.linalg.eigvalsh(q.T @ q)[0] / len(nb))
    return out


def view_counts(views, X, tol=0.3):
    cnt = np.zeros(len(X), int)
    for _, vw in views:
        cnt += np.abs(carve.view_values(vw, X)) < tol
    return cnt


def video_disagreement(views, X, k=3):
    """Per point: how far inside the model each video's silhouettes put the surface (mm): the k-th most negative view value per video
    (positive = that video says the point is outside by that much). Returns the max over videos and the arg video."""
    vids = sorted({vid for vid, _ in views})
    out = np.zeros((len(vids), len(X)))
    for j, vid in enumerate(vids):
        vals = np.stack([carve.view_values(vw, X) for vv, vw in views if vv == vid])
        vals = np.where(vals > 1e8, np.inf, vals)
        out[j] = -np.sort(vals, 0)[min(k, len(vals)) - 1]
    j = out.argmax(0)
    return out.max(0), [vids[i] for i in j]


def regions(fc, flag_v, v, min_area):
    """Connected triangle regions whose 3 vertices are flagged: [(triangle indices, area mm2)], largest first."""
    ft = flag_v[fc].all(1)
    t = np.nonzero(ft)[0]
    if not len(t):
        return []
    e = np.sort(np.concatenate([fc[t][:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))]), 1)
    owner = np.tile(np.arange(len(t)), 3)
    _, inv = np.unique(e, axis=0, return_inverse=True)
    o = np.argsort(inv.ravel(), kind="stable")
    same = inv.ravel()[o][1:] == inv.ravel()[o][:-1]                   # consecutive equal edges = two triangles sharing it
    rows, cols = owner[o][1:][same], owner[o][:-1][same]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(t), len(t)))
    n, lab = connected_components(A, directed=False)
    tri = v[fc[t]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = []
    for i in range(n):
        m = lab == i
        if area[m].sum() >= min_area:
            out.append((t[m], float(area[m].sum())))
    return sorted(out, key=lambda r: -r[1])


def feature_triangles(v, fc, box):
    """Triangles with a vertex inside an oriented box (centre, axes rows, half extents)."""
    c, axes, half = box
    inside = (np.abs((v - c) @ np.asarray(axes).T) <= np.asarray(half) + 1e-6).all(1)
    return np.nonzero(inside[fc].any(1))[0]


def flags(v, fc, views, n_sample=60000, seed=0, th=None):
    """Per-vertex measurements (sampled, then spread to all vertices by nearest sample) and boolean flags."""
    th = {**default_thresholds(), **(th or {})}
    nv = vertex_normals(v, fc)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), min(n_sample, len(v)), replace=False)
    X = v[idx]
    meas = {"lump_rms": lumpiness(v, nv, idx, th["lump_radius_mm"]), "views": view_counts(views, X).astype(float)}
    meas["disagree_mm"], _ = video_disagreement(views, X)
    near = cKDTree(X).query(v)[1]
    full = {k: m[near] for k, m in meas.items()}
    return full, {"lump": full["lump_rms"] > th["lump_rms_mm"], "few_views": full["views"] < th["min_views"],
                  "disagree": full["disagree_mm"] > th["disagree_mm"]}


def default_thresholds():
    return {"lump_radius_mm": C.REVIEW_LUMP_RADIUS_MM, "lump_rms_mm": C.REVIEW_LUMP_RMS_MM, "min_views": C.REVIEW_MIN_VIEWS,
            "disagree_mm": C.REVIEW_DISAGREE_MM, "min_area_mm2": C.REVIEW_MIN_AREA_MM2}


def find(v, fc, views, feature_boxes=(), th=None, measured=None, nv=None):
    """Review items (without pictures): [{"id", "kind", "reason", "triangles", "area_mm2", "centre", "normal", "polygon_xy", "question", "choices"}].
    feature_boxes: [(feature id, oriented box, {video: views that detected it})]. measured: reuse a flags() result (thresholds then only
    change which vertices count as flagged)."""
    th = {**default_thresholds(), **(th or {})}
    full, fl = measured if measured is not None else flags(v, fc, views, th=th)
    if measured is not None:
        fl = {"lump": full["lump_rms"] > th["lump_rms_mm"], "few_views": full["views"] < th["min_views"], "disagree": full["disagree_mm"] > th["disagree_mm"]}
    nv = vertex_normals(v, fc) if nv is None else nv
    items = []
    for kind, f in fl.items():
        for tri, area in regions(fc, f, v, th["min_area_mm2"]):
            items.append({"kind": kind, "triangles": tri, "area_mm2": area})
    for fid, box, by_video in feature_boxes:
        found = {k: n for k, n in by_video.items() if n >= 3}
        if len(found) == 1:
            items.append({"kind": "one_video", "triangles": feature_triangles(v, fc, box), "area_mm2": None, "feature_id": fid,
                          "videos": by_video})
    for i, it in enumerate(items):
        pv = np.unique(fc[it["triangles"]])
        it["id"] = f"R{i + 1:02d}"
        it["centre"] = v[pv].mean(0).round(2).tolist()
        n = nv[pv].mean(0)
        it["normal"] = (n / (np.linalg.norm(n) + 1e-12)).round(3).tolist()
        it["polygon_xy"] = cv2.convexHull(v[pv][:, :2].astype(np.float32)).reshape(-1, 2).round(2).tolist()
        it["question"], it["choices"] = QUESTIONS[it["kind"]]
        it["reason"] = {
            "lump": f"surface deviates from its local plane (rms {np.median(full['lump_rms'][pv]):.2f} mm > {th['lump_rms_mm']} mm)",
            "few_views": f"fixed by {np.median(full['views'][pv]):.0f} silhouettes (< {th['min_views']})",
            "disagree": f"one video puts the surface {np.median(full['disagree_mm'][pv]):.2f} mm further in (> {th['disagree_mm']} mm)",
            "one_video": f"{it.get('feature_id')} detected in one video only: {it.get('videos')}"}[it["kind"]]
    return items


def visible_fraction(vw, pts, occluder, tol=0.4):
    """Share of the points not hidden by the occluder mesh (verts, faces) in a view: a point is visible if no occluder triangle covering its
    pixel is more than tol mm nearer the camera (painter's id buffer of the occluder; depth = that triangle's mean depth)."""
    ov, of = occluder
    _, ids = Rn.render(ov, of, vw, ids=True)
    u, w, z = carve.project(vw, pts)
    H, W = ids.shape
    ui, wi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(w).astype(int), 0, H - 1)
    t = ids[wi, ui]
    dz = ((ov[of[np.maximum(t, 0)]] @ vw.R.T + vw.t)[:, :, 2]).mean(1)
    return float(((t < 0) | (dz > z - tol)).mean())


def best_view(views, images, centre, normal, region_pts, min_cos=0.3, occluder=None, top=6):
    """The view that shows a region best: score = facing (cos between its normal and the direction to the camera) x sharpness (log variance
    of the Laplacian in the crop) x size of the crop x visible share of the region (occlusion test against `occluder`, a coarse mesh, for the
    `top` best views by the other terms). Returns (index, score, crop box, cos, sharpness, visible share) or None."""
    cands = []
    for i, ((_, vw), img) in enumerate(zip(views, images)):
        C_ = -vw.R.T @ vw.t
        d = C_ - centre
        cos = float(d @ normal / np.linalg.norm(d))
        if cos < min_cos:
            continue
        u, w, z = carve.project(vw, region_pts)
        H, W = img.shape[:2]
        if not ((u > 0) & (u < W - 1) & (w > 0) & (w < H - 1) & (z > 0)).all():
            continue
        pad = 40
        x0, x1 = int(max(u.min() - pad, 0)), int(min(u.max() + pad, W))
        y0, y1 = int(max(w.min() - pad, 0)), int(min(w.max() + pad, H))
        sharp = float(cv2.Laplacian(img[y0:y1, x0:x1] if img.ndim == 2 else cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        cands.append((i, cos * np.log1p(sharp) * min((x1 - x0) * (y1 - y0) / 4e4, 1.0), (x0, y0, x1, y1), cos, sharp, 1.0))
    cands.sort(key=lambda c: -c[1])
    if occluder is not None:
        cands = [(i, s * (vis := visible_fraction(views[i][1], region_pts, occluder)), box, cos, sh, vis) for i, s, box, cos, sh, _ in cands[:top]]
        cands.sort(key=lambda c: -c[1])
    return cands[0] if cands else None


def picture(item, v, fc, views, images_bgr, occluder=None, scale_px=360):
    """Photo crop | render of the same model region from the same camera, region outlined; and the view used."""
    pv = np.unique(fc[item["triangles"]])
    pts = v[pv][:: max(len(pv) // 400, 1)]
    b = best_view(views, [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images_bgr], np.asarray(item["centre"]),
                  np.asarray(item["normal"]), pts, occluder=occluder)
    if b is None:
        return None, None
    i, score, (x0, y0, x1, y1), cos, sharp, vis = b
    vw = views[i][1]
    ren, ids = Rn.render(v, fc, vw, ids=True)
    reg = np.isin(ids, item["triangles"]).astype(np.uint8)
    photo = images_bgr[i] if images_bgr[i].ndim == 3 else cv2.cvtColor(images_bgr[i], cv2.COLOR_GRAY2BGR)
    tiles = []
    for im in (photo, ren):
        t = im[y0:y1, x0:x1].copy()
        cs, _ = cv2.findContours(reg[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.drawContours(t, cs, -1, (255, 0, 255), 2, cv2.LINE_AA)
        s = scale_px / max(t.shape[:2])
        tiles.append(cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    return np.hstack(tiles), {"view": vw.name, "cos": round(cos, 2), "sharpness": round(sharp, 1), "score": round(float(score), 2),
                              "visible_share": round(vis, 2), "crop": [x0, y0, x1, y1], "visible_px": int(reg.sum())}
