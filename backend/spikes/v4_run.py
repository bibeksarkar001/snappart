"""v4: owner review of the v3 model (assets/private/lock_set/labels/labels_B_review_v3.json, review_1..5): lock ridge, flat regions and plate,
anomaly finder, corrections engine, question cards. Usage: python backend/spikes/v4_run.py --stage views|1|2|3|4|5

Reads the v1-v3 caches; writes out/v4/SEND_TO_CLAUDE/ (summary.txt first, measured numbers only)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import v2_run as V2R  # noqa: E402
import v3_run as V3R  # noqa: E402
from spike_common import OUT, ROOT  # noqa: E402

V4 = OUT / "v4"
SEND = V4 / "SEND_TO_CLAUDE"
SEND.mkdir(parents=True, exist_ok=True)
LABELS = ROOT / "assets" / "private" / "lock_set" / "labels"
REVIEWS = {k: LABELS / f"review_{k}_{s}.png" for k, s in ((1, "bump_wrong_side"), (2, "bump_underside_convex"), (3, "bump_underside_strip"),
                                                        (4, "underside_should_be_flat"), (5, "window_edge_region"))}
NAMES = V2R.NAMES
date = V2R.date


def save_stage(key, text, results):
    (SEND / f"stage_{key}.txt").write_text(text)
    head = SEND / "header.txt"
    parts = ([head.read_text()] if head.exists() else []) + [p.read_text() for p in sorted(SEND.glob("stage_*.txt"))]
    (SEND / "summary.txt").write_text("\n\n".join(parts))
    f = SEND / "results.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[key] = results
    f.write_text(json.dumps(cur, indent=1, default=float))


# ---------- owner viewpoints: fit a camera to each review screenshot's model silhouette ----------

def owner_mask(img):
    """Model pixels of an owner screenshot: gray (low saturation), not black background, not the light-gray speech bubbles, not the inset
    thumbnail (bottom-right box); annotation strokes are closed over."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = ((g > 45) & (g < 222) & (hsv[..., 1] < 40)).astype(np.uint8)
    h, w = g.shape
    m[int(h * 0.86):, int(w * 0.87):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8) if n > 1 else m


def owner_strokes(img):
    """Annotation strokes: blue (owner's correct place) and orange/red (wrong place) pixel masks."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v, hh = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    blue = (s > 60) & (v > 80) & (hh > 90) & (hh < 115)
    red = (s > 90) & (v > 120) & ((hh < 12) | (hh > 170))
    return blue, red


def sil(verts, faces, view):
    import cv2
    import numpy as np
    c = verts @ view.R.T + view.t
    z = np.maximum(c[:, 2], 1e-6)
    uv = np.stack([view.K[0, 0] * c[:, 0] / z + view.K[0, 2], view.K[1, 1] * c[:, 1] / z + view.K[1, 2]], 1)
    w, h = view.size
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, list(np.round(uv[faces] * 8).astype(np.int32)), 1, cv2.LINE_8, 3)
    return m


def fit_view(verts, faces, target, init, scale):
    """Camera (az, el, roll, dist, cx, cy; f = image width) maximising silhouette IoU with the target mask. init: list of (az, el, roll)."""
    import numpy as np
    from scipy.optimize import minimize
    from capture import render as Rn
    h, w = target.shape
    c = (verts.max(0) + verts.min(0)) / 2
    f = float(w)
    ta = target.sum()
    ty, tx = np.nonzero(target)
    tcen = np.array([tx.mean(), ty.mean()])

    def cam(p):
        az, el, roll, dist, cx, cy = p
        return Rn.orbit(c, az, el, dist, roll, f=f, size=(w, h), centre=(cx, cy))

    def iou(p):
        s = sil(verts, faces, cam(p))
        return (s & target).sum() / max((s | target).sum(), 1)
    best = None
    for az, el, roll in init:
        p = np.array([az, el, roll, 250.0, w / 2, h / 2])
        for _ in range(3):                                # match area and centroid
            s = sil(verts, faces, cam(p))
            if s.sum() < 50:
                break
            sy, sx = np.nonzero(s)
            p[3] *= np.sqrt(s.sum() / ta)
            p[4:] += tcen - np.array([sx.mean(), sy.mean()])
        q = iou(p)
        if best is None or q > best[0]:
            best = (q, p.copy())
    r = minimize(lambda p: -iou(p), best[1], method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 600})
    p = r.x
    return {"az": p[0], "el": p[1], "roll": p[2], "dist": p[3], "cx": p[4] / scale, "cy": p[5] / scale, "f": f / scale,
            "w": int(w / scale), "h": int(h / scale), "iou": float(-r.fun), "iou_init": float(best[0]), "target": c.tolist()}


def owner_camera(rec, size_scale=1.0):
    from capture import render as Rn
    s = size_scale
    return Rn.orbit(rec["target"], rec["az"], rec["el"], rec["dist"], rec["roll"], f=rec["f"] * s, size=(int(rec["w"] * s), int(rec["h"] * s)),
                    centre=(rec["cx"] * s, rec["cy"] * s))


def coarse_mesh(field, g, step=3):
    from capture import carve
    gc = carve.Grid(g.lo, g.lo + g.vs * step, g.vs * step)
    gc.shape = field[::step, ::step, ::step].shape
    return carve.to_mesh(field[::step, ::step, ::step], gc)


def stage_views():
    """Fit the five owner viewpoints to the v3 model (B_plate.stl) by silhouette IoU; owner_views.json + overlay check pictures."""
    import cv2
    import numpy as np
    from capture import model as Mo
    t0, d0 = time.time(), date()
    m = Mo.load(V3R.V3 / "model_B")
    fv3 = Mo.apply(m["field"], m["grid"], m["features"], Mo.default_answers())
    vc, fcc = coarse_mesh(fv3, m["grid"])
    scale = 0.25
    out, L = {}, [f"OWNER VIEWPOINTS. start {d0}. Camera fitted to each review screenshot by silhouette IoU against the v3 model (B_plate), "
                  f"coarse mesh ({len(fcc)} triangles), screenshots at 1/4 scale, f = image width; search over azimuth x elevation x roll, then Powell."]
    grid = [(az, el, roll) for az in range(0, 360, 30) for el in (-75, -50, -25, 0, 25, 50, 75) for roll in range(0, 360, 45)]
    for k, p in REVIEWS.items():
        img = cv2.imread(str(p))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tgt = owner_mask(small)
        rec = fit_view(vc, fcc, tgt, grid, scale)
        out[k] = rec
        cam = owner_camera(rec, scale)
        s = sil(vc, fcc, cam)
        pic = small.copy()
        cs, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(pic, cs, -1, (0, 255, 0), 1)
        cv2.imwrite(str(V4 / f"owner_view_fit_{k}.png"), pic)
        L.append(f"  review_{k}: IoU {rec['iou']:.3f} (grid best {rec['iou_init']:.3f}); az {rec['az']:.1f}, el {rec['el']:.1f}, roll {rec['roll']:.1f}, dist {rec['dist']:.0f} mm")
    (V4 / "owner_views.json").write_text(json.dumps(out, indent=1, default=float))
    L.append(f"end {date()}; elapsed {time.time() - t0:.0f} s")
    print("\n".join(L))
    return L


def owner_region_mask(img, colour="blue"):
    """The area the owner outlined in one colour: strokes closed, holes filled, the speech bubble (light interior) and leader lines dropped
    (leader lines enclose nothing). Returns a 0/1 mask at the screenshot's size."""
    import cv2
    import numpy as np
    blue, red = owner_strokes(img)
    s = (blue if colour == "blue" else red).astype(np.uint8)
    s = cv2.dilate(s, np.ones((3, 3), np.uint8))
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    h, w = s.shape
    ff = s.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = (s | (1 - ff)).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled)
    out = np.zeros_like(filled)
    for i in range(1, n):
        comp = lab == i
        inner = comp & (s == 0)
        if inner.sum() < 400 or np.median(g[inner]) > 215:               # a bare line, or the speech bubble
            continue
        out[comp] = 1
    return out


def owner_regions(verts, faces, scale=0.5):
    """3D triangles the owner marked: {"review_1": orange box, "review_4": blue outline, "review_5": blue outline, "review_2"/"review_3": blue
    boxes (where the ridge really is)}, each {"tri": triangle indices visible inside the outline in the fitted owner camera, "polygon_xy":
    convex hull of their xy}."""
    import cv2
    import numpy as np
    from capture import render as Rn
    out = {}
    for k, col in ((1, "red"), (2, "blue"), (3, "blue"), (4, "blue"), (5, "blue")):
        img = cv2.imread(str(REVIEWS[k]))
        msk = cv2.resize(owner_region_mask(img, col), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        _, ids = Rn.render(verts, faces, owner_camera(owner_views()[k], scale), ids=True)
        tri = np.unique(ids[(msk > 0) & (ids >= 0)])
        xy = verts[faces[tri]].reshape(-1, 3)[:, :2].astype(np.float32)
        hull = cv2.convexHull(xy).reshape(-1, 2).tolist() if len(tri) else []
        out[f"review_{k}"] = {"tri": tri, "polygon_xy": hull, "pixels": int(msk.sum())}
    return out


def owner_views():
    return {int(k): v for k, v in json.loads((V4 / "owner_views.json").read_text()).items()}


def render_owner(verts, faces, k, scale=0.5, label=None):
    """The model from the owner's review_k viewpoint (fitted camera), at scale x the screenshot size."""
    import cv2
    from capture import render as Rn
    img = Rn.render(verts, faces, owner_camera(owner_views()[k], scale))
    if label:
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def facing_counts(vv, n, p):
    """Views per video facing a face (outward normal n at point p) with cos > 0.5, and their median cos."""
    import numpy as np
    out = {}
    for vid, v in vv:
        C = -v.R.T @ v.t
        c = float((C - p) @ n / np.linalg.norm(C - p))
        out.setdefault(vid, []).append(c)
    return {vid: {"views_cos_gt_0.5": int(sum(c > 0.5 for c in cs)), "median_cos": round(float(np.median(cs)), 2)} for vid, cs in out.items()}


def depth_sweep(vv, imgs, vid, box, zs, ppm=10, min_cos=0.5):
    """Per-pixel photo-consistency sweep of horizontal planes (normal -z) over box (x0, x1, y0, y1) with one video's views: robust spread
    (median absolute deviation from the median) of high-passed, contrast-normalised rectified images; returns (cost [z, row, col], views)."""
    import cv2
    import numpy as np
    from capture import features as F
    x0, x1, y0, y1 = box
    face = lambda z: {"O": np.array([x0, y0, z]), "e1": np.array([0, 1.0, 0]), "e2": np.array([1.0, 0, 0]), "n": np.array([0, 0, -1.0]),
                      "len_s": y1 - y0, "len_t": x1 - x0}
    sel = [(v, im) for (n, v), im in zip(vv, imgs) if n == vid and F.facing(v, face(5)) > min_cos]
    cost = []
    for z in zs:
        R = []
        for v, im in sel:
            r = F.rectify(v, im, face(z), 0, ppm).astype(np.float32)
            hp = r - cv2.GaussianBlur(r, (0, 0), 3)
            R.append(hp / (cv2.GaussianBlur(np.abs(hp), (0, 0), 6) + 2))
        R = np.stack(R)
        cost.append(cv2.GaussianBlur(np.median(np.abs(R - np.median(R, 0)), 0), (0, 0), 1.5))
    return np.stack(cost), len(sel)


def overlay(frame_bgr, view, verts, faces, tri_mask, title):
    """Frame | frame with the model's outline (green) and the ridge's visible surface (magenta) drawn on it."""
    import cv2
    import numpy as np
    from capture import render as Rn
    w, h = view.size
    _, ids = Rn.render(verts, faces, view, ids=True)
    vis = np.zeros((h, w), np.uint8)
    hit = ids >= 0
    vis[hit] = tri_mask[faces[ids[hit]]].all(1)
    out = frame_bgr.copy()
    out[vis > 0] = (0.45 * out[vis > 0] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    cs, _ = cv2.findContours((ids >= 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 220, 0), 1, cv2.LINE_AA)
    cs, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (255, 0, 255), 1, cv2.LINE_AA)
    pic = np.hstack([frame_bgr, out])
    cv2.putText(pic, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return pic


def stage1():
    """Lock ridge: remove the v3 block, detect the ridge from video, model it as a hill on the face it stands on."""
    import cv2
    import numpy as np
    from capture import carve, config as C, dataset, features as F, model as Mo, pipeline as P
    t0, d0 = time.time(), date()
    tm = {}
    tick = lambda k, t=[time.time()]: (tm.__setitem__(k, round(time.time() - t[0], 1)), t.__setitem__(0, time.time()))
    m = Mo.load(V3R.V3 / "model_B")
    field, g, feats = m["field"], m["grid"], m["features"]
    v0, f0 = Mo.mesh(field, g)
    vv, imgs = V2R.all_views()
    tick("load")
    face, ops, _, _ = P.find_grooves_parallel(v0, f0, vv, imgs)
    delta = float(np.mean([o["dist_from_face_mm"] for o in ops]))
    tick("groove face")
    b = F.detect_bump(vv, imgs, face, delta)
    rs = F.ridge_from_steps(b, face)
    tick("ridge detection")
    # which video sees which face best: the flat +z face (the face opposite the rim the v3 block sat on) and the rim/pocket side
    plate_c = np.array([158.0, 101.0, 10.0])
    cnt = {"+z face (opposite the rim face; flat, video 3 rests on it)": facing_counts(vv, np.array([0, 0, 1.0]), plate_c),
           "-z side (rim face with the slot openings, pocket, ridge)": facing_counts(vv, np.array([0, 0, -1.0]), plate_c),
           "slot column inner wall (faces the plate, -x)": facing_counts(vv, np.array(rs["normal"]), np.mean(rs["wall_line"], 0))}
    # depth of the ridge's surface from video 3 (horizontal-plane sweep, per pixel): rim, ridge, pocket floor along x
    zs = np.arange(-1, 11.01, 0.1)
    box = (160.0, 170.0, 94.0, 110.0)
    cost, nsel = depth_sweep(vv, imgs, "A_orbit_03", box, zs)
    H = cost.shape[1]
    prof = []
    for x in np.arange(162.0, 169.01, 0.2):
        r = int(round(H - 1 - (x - box[0]) * 10))
        best = zs[cost[:, r, 10:-10].argmin(0)]
        prof.append((round(float(x), 1), round(float(np.median(best)), 2), round(float(np.percentile(best, 25)), 2), round(float(np.percentile(best, 75)), 2)))
    tick("depth sweep")
    # the ridge feature: base on the wall beside the span, from the rim inward by the base width; outward normal = measured direction
    ends = np.array(b["ends"])
    axis = ends[1] - ends[0]
    axis /= np.linalg.norm(axis)
    nrm = np.array(rs["normal"])
    nrm -= axis * (axis @ nrm)
    nrm /= np.linalg.norm(nrm)
    across = np.cross(nrm, axis)
    across *= np.sign(across @ -np.asarray(face["n"]))              # into the part from the rim
    W = C.BUMP_WIDTH_USER_MM
    wall_mid = np.mean(rs["wall_line"], 0)
    ridge = {"kind": "ridge", "shape": "hill", "base_mid": (wall_mid + across * W / 2).tolist(), "axis": axis.tolist(), "across": across.tolist(),
             "normal": nrm.tolist(), "detected_length_mm": b["length_mm"], "ends": b["ends"], "wall_line": rs["wall_line"],
             "extent_mm": rs["extent_mm"], "extent_views": rs["views"], "source_video": b["video"], "agreeing_views": b["agree"],
             "face": "inner wall of the slot column (pocket side), below the rim"}
    # open the pocket up to the wall the ridge stands on (the v3 plate stopped at the ridge's far edge); v3 plate plane kept until stage 2
    x0, _, y0, y1 = feats["plate"]["region_xy"]
    xw = float(wall_mid[0])
    plate = {k: v for k, v in feats["plate"].items() if k != "mask"}
    plate.update(polygon_xy=[[x0, y0], [xw, y0], [xw, y1], [x0, y1]], region_xy=[x0, xw, y0, y1])
    feats1 = {**feats, "bump": ridge, "plate": plate}
    ans = Mo.default_answers()
    f_no = Mo.apply(field, g, {**feats1, "bump": None}, ans)
    f1 = Mo.apply(field, g, feats1, ans)
    v1, fc1 = Mo.mesh(f1, g)
    ca = carve.edge_census(v1, fc1)
    carve.write_stl(SEND / "B_ridge.stl", v1, fc1)
    Mo.save(V4 / "model_B_stage1", field, g, feats1)
    tick("model + mesh")
    poly = Mo.profile("hill", W, C.BUMP_HEIGHT_USER_MM, C.BUMP_TOP_RADIUS_MM)
    peak = np.asarray(ridge["base_mid"]) + nrm * C.BUMP_HEIGHT_USER_MM
    # overlays on two real frames: one used for the detection (video 3, head-on), one not used (video 2)
    tri = Mo.field_at(f_no, g, v1) < -0.05
    used = sorted(b["per_view"], key=lambda p: -p["cos"])[0]["name"]
    agreeing = {p["name"] for p in b["per_view"]}
    T = V2R.load_T()
    pics = []
    for name in (used, "A_orbit_02_1204", "A_orbit_03_762"):
        vid = name.rsplit("_", 1)[0]
        v = next(x for x in dataset.load_views(vid) if x.name == name).moved(T[vid])
        fr = cv2.imread(str(dataset.CACHE / "crops" / f"{name}.jpg"))
        tag = "used for the detection" if name in agreeing else "NOT used for the detection"
        pic = overlay(fr, v, v1, fc1, tri, f"{name} ({tag}): frame | model outline (green), ridge (magenta)")
        cv2.imwrite(str(SEND / f"ridge_overlay_{name}.png"), pic)
        pics.append((name, tag))
    # renders from the owner's review_1 and review_2 viewpoints, v3 vs stage 1
    v3m = V2R.load_mesh(V3R.SEND / "B_plate.stl")
    for k in (1, 2):
        a = render_owner(*v3m, k, label=f"v3 (review_{k} viewpoint)")
        c = render_owner(v1, fc1, k, label="v4 stage 1: ridge on the slot column's inner wall")
        cv2.imwrite(str(SEND / f"stage1_review{k}_view.png"), np.hstack([a, c]))
    tick("overlays + renders")
    L = [f"STAGE 1: lock ridge. start {d0}. Baseline v3 B_plate.stl (block on the rim face the slot openings sit in, standing out of it along -z).",
         "WHICH FACE. The owner's screenshots were matched to the model by fitting a camera to each (owner_views.json): review_1 and review_2 look at the "
         "-z side (the rim face with the slot openings and the deep pocket over the plate); review_3, review_4 and review_5 also look at the -z side, from "
         "below the cross-bar end (3, 4) and from the slot end (5). The face opposite the v3 block (the +z face) is the flat face: video 3 rests on it and "
         "video 1 sees it from above. Views facing each (cos > 0.5, median cos):"]
    L += [f"  {k}: " + ", ".join(f"{vid[-2:]} {c['views_cos_gt_0.5']} views (median cos {c['median_cos']})" for vid, c in d.items()) for k, d in cnt.items()]
    L += ["  On the +z face there is no ridge in the video: video 1's frames show it flat (frames 3324, 776, 2274) and video 3 lies flat on it. A 2.3 mm ridge "
          "there would stand 2.3 mm out of the face the part rests on in video 3.",
          "  The v3 direction (out of the rim face, -z) is also impossible at 2.3 mm: the rim face is at z 1.93 and video 1 rests on the plane z = 0, so a "
          "2.3 mm block would reach z = -0.37, through the mat.",
          "  The ridge the video does show is on the -z side, inside the pocket: a rounded rail along the slot column's inner wall, just below the rim "
          "(frames A_orbit_03_762, A_orbit_03_1304, A_orbit_02_1204). Best views: video 3 head-on (top-down) for its outline, video 2 for the oblique view.",
          "  So the ridge is modelled there; its outward normal points from the wall toward the plate. This differs from the brief's reading of "
          "'underside' (the +z face); the owner should confirm.",
          f"DETECTION (rectify views onto the rim plane, 3-segment edge-step fit per view, cluster views whose length is within 1 mm of their video's "
          f"median, triangulate the ends): {b['agree']} of {b['found']} views agree ({b['video']}).",
          f"  ends {np.round(ends[0], 2).tolist()} .. {np.round(ends[1], 2).tolist()}, detected length {b['length_mm']:.2f} mm (owner 19.7).",
          f"  plate-side edge of the ridge (head-on outline) at x {ends[:, 0].mean():.2f}; it lies {b['ridge_to_opening_mm']:.2f} mm from the openings' edge.",
          f"  extent beyond the wall, from head-on views (cos >= 0.9): {rs['extent_mm']:.2f} mm (sd {rs['extent_sd_mm']:.2f}, range "
          f"{rs['extent_range_mm'][0]:.2f}..{rs['extent_range_mm'][1]:.2f}, {rs['views']} views): MEASURED height of the ridge out of the wall in projection. "
          f"Owner's height 2.3 mm: difference {rs['extent_mm'] - 2.3:+.2f} mm.",
          f"  wall line (where the rim edge runs beside the span): x {np.mean([p[0] for p in rs['wall_line']]):.2f}; direction (outward normal) "
          f"({nrm[0]:+.3f}, {nrm[1]:+.3f}, {nrm[2]:+.3f}), toward the plate: MEASURED.",
          f"  depth check (video 3, {nsel} views, per-pixel plane sweep of horizontal planes; median z over y 95..109 along x; IQR):"]
    L += ["    " + "; ".join(f"x {x}: z {z} ({a}..{c})" for x, z, a, c in prof[i:i + 6]) for i in range(0, len(prof), 6)]
    L += ["    reading: the rim at z ~1.8-2.2 for x >= 166.2, the ridge surface at z ~3.0-3.6 for x 165.2-165.6, then the pocket floor (plate) at z ~9-10 "
          "for x 164.2-164.8. A hill whose base runs from the rim (z 1.93) 2.3 mm into the part peaks at z 3.08: consistent with the ridge surface at "
          "z 3.0-3.6.",
          "MODEL (B_ridge.stl): v3 block removed; ridge = hill extruded along the edge. Values:",
          f"  position along the edge: centred on the detected ends' midpoint ({np.round(np.mean(ends, 0), 2).tolist()}): MEASURED",
          f"  face and direction: inner wall of the slot column, base line x {xw:.2f}, outward normal toward the plate: MEASURED",
          f"  length {C.BUMP_LENGTH_USER_MM} mm: USER-SUPPLIED (detected {b['length_mm']:.2f})",
          f"  base width {W} mm (from the rim edge into the part): USER-SUPPLIED",
          f"  height {C.BUMP_HEIGHT_USER_MM} mm out of the wall: USER-SUPPLIED (measured extent {rs['extent_mm']:.2f}); peak line at "
          f"({peak[0]:.2f}, -, {peak[2]:.2f})",
          f"  top radius {C.BUMP_TOP_RADIUS_MM} mm and side slope {Mo.side_slope_deg(poly, W):.1f} deg: ESTIMATED. No frame shows the profile edge-on: "
          "the ridge is inside the pocket, between the rails, so it is never on a silhouette.",
          f"  pocket opened up to the wall (x {xw:.2f}); the v3 plate stopped at the ridge's far edge (x {x0:.2f}..{feats['plate']['region_xy'][1]:.2f}). "
          "The plate itself is re-placed in stage 2.",
          f"  watertight {ca['watertight']} (boundary {ca['boundary_edges']}, non-manifold {ca['non_manifold_edges']}), volume {ca['volume_mm3']:.0f} mm3",
          "  owner's position check: the owner put it ~1-1.5 mm closer to the plate than the v3 block, roughly centred along the edge. The v3 block spanned "
          f"x 165.67..167.97 (centre 166.82); the ridge spans x {peak[0]:.2f}..{xw:.2f} (centre {(peak[0] + xw) / 2:.2f}, "
          f"{166.82 - (peak[0] + xw) / 2:.2f} mm closer); along the edge it is centred at y {np.mean(ends[:, 1]):.2f} (part centre y 101.5).",
          "VERIFY: overlays " + ", ".join(f"ridge_overlay_{n}.png ({t})" for n, t in pics) + "; owner viewpoints stage1_review1_view.png, "
          "stage1_review2_view.png (left v3, right stage 1).",
          "timings (s): " + ", ".join(f"{k} {v}" for k, v in tm.items()),
          f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("1_ridge", "\n".join(L), {"bump": {k: v for k, v in b.items() if k != "per_view"}, "per_view": b["per_view"], "ridge_steps": rs,
                                         "facing": cnt, "depth_profile": prof, "ridge": ridge, "census": ca, "timings": tm})
    print("\n".join(L))


def bottom_plane(v, fc, rail_y=((80.8, 83.6), (119.6, 122.2)), x_range=(128.0, 160.0)):
    """The flat +z face (the face video 3 rests on) fitted to the mesh's up-facing samples on the two rails (clean: convex, on the
    silhouette in many views): outward normal n, d (n.x + d = 0), rms and samples."""
    import numpy as np
    from capture import snap
    P, N = snap.oriented_points(v, fc, 200000)
    m = (N[:, 2] > 0.9) & (P[:, 0] > x_range[0]) & (P[:, 0] < x_range[1]) & np.any([(P[:, 1] > a) & (P[:, 1] < b) for a, b in rail_y], 0)
    q = P[m]
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][2]
    n *= np.sign(n[2])
    r = (q - c) @ n
    return {"n": n.tolist(), "d": float(-n @ c), "rms_mm": float(np.sqrt((r ** 2).mean())), "samples": int(m.sum()),
            "tilt_deg": float(np.degrees(np.arccos(n[2])))}


def window_edge(field, g, plane, y0, y1, x_lo=140.0, x_hi=160.0, depth=1.3):
    """The window's far edge (where the plate starts), from the carve: along each column y, the first material going +x from the window at
    `depth` mm under the bottom plane (the window is a through-hole, so silhouettes see this edge). Line x = a + b*y fitted by least absolute
    deviation (IRLS). Returns (a, b, per-y edge points, rms of the points to the line)."""
    import numpy as np
    from capture import model as Mo
    n, d = np.asarray(plane["n"]), plane["d"]
    ys = np.arange(y0 + 1.0, y1 - 1.0, 0.25)
    xs = np.arange(x_lo, x_hi, 0.05)
    pts = []
    for y in ys:
        X = np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], 1)
        X[:, 2] = -(d + depth + X[:, :2] @ n[:2]) / n[2]                  # on the plane `depth` under the bottom plane
        f = Mo.field_at(field, g, X)
        k = np.argmax(f > 0)
        if f[k] > 0 and k > 0:
            pts.append((xs[k - 1] + 0.05 * (-f[k - 1]) / (f[k] - f[k - 1]), y))
    P = np.array(pts)
    w = np.ones(len(P))
    for _ in range(20):
        A = np.stack([np.ones(len(P)), P[:, 1]], 1) * w[:, None]
        a, b = np.linalg.lstsq(A, P[:, 0] * w, rcond=None)[0]
        w = 1 / np.sqrt(np.maximum(np.abs(P[:, 0] - a - b * P[:, 1]), 1e-3))
    r = P[:, 0] - a - b * P[:, 1]
    return float(a), float(b), P, float(np.sqrt((r ** 2).mean()))


def region_stats(v, fc, poly, plane, n_samples=300000):
    """Surface samples of a mesh inside an xy polygon: the -z-facing ones (the side the owner looked at) -> RMS to their own plane; the
    +z-facing ones -> RMS distance to the bottom plane."""
    import numpy as np
    from matplotlib.path import Path as MPath
    from capture import snap
    P, N = snap.oriented_points(v, fc, n_samples)
    inside = MPath(np.asarray(poly)).contains_points(P[:, :2])
    out = {}
    for side, m in (("down", inside & (N[:, 2] < -0.5)), ("up", inside & (N[:, 2] > 0.5))):
        q = P[m]
        if len(q) < 10:
            out[side] = None
            continue
        c = q.mean(0)
        nn = np.linalg.svd(q - c)[2][2]
        r = (q - c) @ nn
        dp = q @ np.asarray(plane["n"]) + plane["d"]
        out[side] = {"rms_own_plane_mm": float(np.sqrt((r ** 2).mean())), "rms_to_bottom_plane_mm": float(np.sqrt((dp ** 2).mean())),
                     "mean_offset_from_bottom_plane_mm": float(dp.mean()), "samples": int(m.sum())}
    return out


def stage2():
    """Flat regions and the plate: bottom plane, review_4 region and review_5 strip flat, window edge straight, plate on the bottom plane."""
    import cv2
    import numpy as np
    from capture import carve, config as C, features as F, model as Mo
    t0, d0 = time.time(), date()
    tm = {}
    tick = lambda k, t=[time.time()]: (tm.__setitem__(k, round(time.time() - t[0], 1)), t.__setitem__(0, time.time()))
    m3 = Mo.load(V3R.V3 / "model_B")
    field, g, feats3 = m3["field"], m3["grid"], m3["features"]
    m1 = Mo.load(V4 / "model_B_stage1")
    ridge = m1["features"]["bump"]
    v3m = V2R.load_mesh(V3R.SEND / "B_plate.stl")
    v0, f0 = Mo.mesh(field, g)
    tick("load")
    bp = bottom_plane(v0, f0)
    regs = owner_regions(*v3m)
    tick("bottom plane + owner regions")
    _, _, y0, y1 = feats3["plate"]["region_xy"]
    a, b, epts, erms = window_edge(field, g, bp, y0, y1)
    xw = float(np.mean([p[0] for p in ridge["wall_line"]]))
    edge = lambda y: a + b * y
    plate_poly = [[edge(y0), y0], [xw, y0], [xw, y1], [edge(y1), y1]]
    win_poly = [[edge(y0) - 4.0, y0], [edge(y0), y0], [edge(y1), y1], [edge(y1) - 4.0, y1]]       # 4 mm strip of the window before the edge
    clip = lambda poly: [[min(max(x, edge(y)), xw), min(max(y, y0), y1)] for x, y in poly]
    plane = {"n": bp["n"], "d": bp["d"], "name": "bottom"}
    th = C.PLATE_THICKNESS_USER_MM
    corr = [{"op": "add_feature", "feature_id": "bump", "feature": ridge, "reason": "lock ridge measured from video (stage 1); replaces the v3 block",
             "source": "owner review_1/2/3 + video", "review_item": "review_1"},
            {"op": "flatten_region", "feature_id": "plate", "region": {"polygon_xy": plate_poly}, "to_plane": {"n": bp["n"], "d": bp["d"]},
             "reason": "plate: outer face on the bottom plane (owner: flat like the bottom), from the straightened window edge to the slot column wall",
             "source": "owner + video"},
            {"op": "flatten_region", "region": {"polygon_xy": clip(regs["review_4"]["polygon_xy"])}, "to_plane": {"n": bp["n"], "d": bp["d"]},
             "thickness_answer": "plate_thickness_mm", "reason": "owner: this region is flat like the bottom (review_4)", "source": "owner", "review_item": "review_4"},
            {"op": "flatten_region", "region": {"polygon_xy": clip(regs["review_5"]["polygon_xy"])}, "to_plane": {"n": bp["n"], "d": bp["d"]},
             "thickness_answer": "plate_thickness_mm", "reason": "strip between the window and the plate is flat (review_5)", "source": "owner", "review_item": "review_5"},
            {"op": "remove_region", "region": {"polygon_xy": win_poly, "z_mm": None}, "reason": "straight window edge (line fitted to the silhouette-carved edge)",
             "source": "video", "review_item": "review_5"}]
    ans = Mo.default_answers()
    f_v3 = Mo.apply(field, g, feats3, ans)
    f2 = Mo.apply(field, g, feats3, ans, corr)
    v2, fc2 = Mo.mesh(f2, g)
    ca = carve.edge_census(v2, fc2)
    carve.write_stl(SEND / "B_v4.stl", v2, fc2)
    tick("apply + mesh")
    vol = lambda f: float((f > 0).sum() * g.vs ** 3)
    # volume removed / added per correction (each applied alone on top of v3 + ridge, so the numbers do not overlap)
    f_r = Mo.apply(field, g, feats3, ans, corr[:1])
    dv = {}
    for c in corr[1:]:
        fc_ = Mo.apply(field, g, feats3, ans, corr[:1] + [c])
        dv[c["reason"][:60]] = {"removed_mm3": float(((f_r > 0) & (fc_ <= 0)).sum() * g.vs ** 3), "added_mm3": float(((f_r <= 0) & (fc_ > 0)).sum() * g.vs ** 3)}
    tot = {"removed_mm3": float(((f_v3 > 0) & (f2 <= 0)).sum() * g.vs ** 3), "added_mm3": float(((f_v3 <= 0) & (f2 > 0)).sum() * g.vs ** 3)}
    tick("volumes")
    # plate faces re-measured with planes parallel to the bottom plane (offset inward from it)
    n = np.asarray(bp["n"])
    e1 = np.array([0, 1.0, 0]) - n * n[1]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    px0, px1 = edge(101) + 1.5, xw - 3.0
    vv, imgs = V2R.all_views()

    def face_off(off, up):
        c0 = np.array([px0, y0 + 1.5, 0.0])
        c0[2] = -(bp["d"] + c0[:2] @ n[:2]) / n[2]
        O = c0 - off * n
        return {"O": O, "e1": e1, "e2": e2 * np.sign(e2[0]), "n": n if up else -n, "len_s": y1 - y0 - 3, "len_t": px1 - px0}
    offs = np.arange(-1.0, 5.01, 0.1)
    meas = {}
    for side, vid, up in (("outer (+z) face", "A_orbit_01", True), ("inner (pocket floor) face", "A_orbit_03", False)):
        sel = [(v, im) for (nm, v), im in zip(vv, imgs) if nm == vid and F.facing(v, face_off(1.0, up)) > 0.6]
        z, sd, sc = F.plane_sweep([v for v, _ in sel], [im for _, im in sel], lambda o: face_off(o, up), offs)
        meas[side] = {"offset_mm": z, "boot_sd": sd, "views": len(sel), "video": vid, "contrast": float(sc.max() / sc.min())}
    sep = meas["inner (pocket floor) face"]["offset_mm"] - meas["outer (+z) face"]["offset_mm"]
    tick("plate sweep")
    # flatness of the owner's regions before (v3) and after
    flat = {}
    for k in ("review_4", "review_5"):
        poly = clip(regs[k]["polygon_xy"])
        flat[k] = {"before": region_stats(*v3m, poly, bp), "after": region_stats(v2, fc2, poly, bp), "polygon_xy": poly}
    a2, b2, epts2, erms2 = window_edge(f2, g, bp, y0, y1)
    tick("flatness")
    for k in range(1, 6):
        cv2.imwrite(str(SEND / f"before_after_review{k}.png"),
                    np.hstack([render_owner(*v3m, k, label=f"v3, review_{k} viewpoint"), render_owner(v2, fc2, k, label="v4 (stages 1+2)")]))
    tick("renders")
    Mo.save(V4 / "model_B_v4", field, g, {**feats3, "planes": {"bottom": {"n": bp["n"], "d": bp["d"]}}})
    (V4 / "corrections_v4.json").write_text(json.dumps(corr, indent=1, default=lambda o: np.asarray(o).tolist()))
    L = [f"STAGE 2: flat regions and the plate. start {d0}. Baseline v3 B_plate + the stage 1 ridge.",
         f"(a) bottom plane (the flat +z face video 3 rests on), fitted to the up-facing samples on both rails, x 128..160: normal "
         f"({bp['n'][0]:+.4f}, {bp['n'][1]:+.4f}, {bp['n'][2]:+.4f}) (tilt {bp['tilt_deg']:.2f} deg from the video 1 mat plane), rms {bp['rms_mm']:.3f} mm, "
         f"{bp['samples']} samples.",
         "  owner's regions in 3D: each outline mapped through the fitted owner camera onto the v3 mesh (triangles visible inside it): "
         + ", ".join(f"{k} {len(r['tri'])} triangles" for k, r in regs.items()) + ".",
         f"  review_4 region (xy polygon, clipped to the plate area between the window edge and the rails' inner faces): {np.round(flat['review_4']['polygon_xy'], 1).tolist()}",
         "  made a single flat face: its outer face on the bottom plane, its pocket-side face flat, plate thickness from config.py (both from flatten_region)."]
    for k in ("review_4", "review_5"):
        for when in ("before", "after"):
            s = flat[k][when]
            L.append(f"  {k} {when}: " + "; ".join(f"{side}-facing {s[side]['samples']} samples, RMS to own plane {s[side]['rms_own_plane_mm']:.3f} mm, "
                                                    f"RMS to bottom plane {s[side]['rms_to_bottom_plane_mm']:.3f} mm" for side in ("down", "up") if s[side]))
    L += ["  volume per correction (each alone on v3 + ridge):"] + [f"    {k}: removed {x['removed_mm3']:.1f} mm3, added {x['added_mm3']:.1f} mm3" for k, x in dv.items()]
    L += [f"  all corrections vs v3: removed {tot['removed_mm3']:.1f} mm3, added {tot['added_mm3']:.1f} mm3",
          f"(b) window edge: line x = {a:.2f} + {b:+.4f} y fitted (least absolute deviation) to {len(epts)} edge points of the carve at 1.3 mm under the bottom "
          f"plane; points' rms to the line before {erms:.3f} mm (jaggedness), after {erms2:.3f} mm; the edge moved from mean x {epts[:, 0].mean():.2f} to "
          f"{epts2[:, 0].mean():.2f}. Material in the window on the near side of the line removed (4 mm strip).",
          f"(c) plate: outer face on the bottom plane, thickness {th} mm (config.py, USER-SUPPLIED), from the window edge to the slot column wall (x {xw:.2f}).",
          "  re-measured (photo-consistency sweeps of planes parallel to the bottom plane, offset inward; 20 bootstrap resamples):"]
    L += [f"    {k}: offset {r['offset_mm']:.2f} mm (sd {r['boot_sd']:.2f}), {r['views']} views of {r['video']}, contrast {r['contrast']:.2f}" for k, r in meas.items()]
    L += [f"    MEASURED SEPARATION {sep:.2f} mm vs owner 2.6 ({sep - 2.6:+.2f}). " + ("Within 0.3 mm." if abs(sep - 2.6) <= 0.3 else "Still in conflict; 2.6 is used (user-supplied)."),
          f"B_v4.stl: watertight {ca['watertight']} (boundary {ca['boundary_edges']}, non-manifold {ca['non_manifold_edges']}), volume {ca['volume_mm3']:.0f} mm3, {ca['triangles']} triangles",
          "renders from the owner's five viewpoints: before_after_review1..5.png (left v3, right v4)",
          "corrections that turn v3 into v4: out/v4/corrections_v4.json (also in report.json, stage 4)",
          "timings (s): " + ", ".join(f"{k} {v}" for k, v in tm.items()),
          f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("2_flat_plate", "\n".join(L), {"bottom_plane": bp, "window_edge": {"a": a, "b": b, "rms_before": erms, "rms_after": erms2},
                                              "flatness": flat, "volumes": dv, "volume_total": tot, "plate_sweep": meas, "separation_mm": sep, "census": ca,
                                              "owner_regions": {k: {"triangles": len(r["tri"]), "polygon_xy": r["polygon_xy"]} for k, r in regs.items()}, "timings": tm})
    print("\n".join(L))


def color_crops(vv):
    import cv2
    from capture import dataset
    return [cv2.imread(str(dataset.CACHE / "crops" / f"{v.name}.jpg")) for _, v in vv]


def match(items, owner, v, tol=1.0, share=0.2):
    """Which owner regions each item hits: >= share of the item's vertices within tol mm of the owner region's vertices, or >= share of the
    owner region's vertices within tol mm of the item's."""
    import numpy as np
    from scipy.spatial import cKDTree
    hits = {}
    for it in items:
        pi = v[np.unique(it["fc_tri"])]
        ti = cKDTree(pi)
        hits[it["id"]] = []
        for k, o in owner.items():
            po = v[o["verts"]]
            a = (cKDTree(po).query(pi, distance_upper_bound=tol)[0] < tol).mean()
            b = (ti.query(po, distance_upper_bound=tol)[0] < tol).mean()
            if a >= share or b >= share:
                hits[it["id"]].append(k)
    return hits


def score(items, owner, v):
    h = match(items, owner, v)
    caught = sorted({k for ks in h.values() for k in ks})
    false = [i for i, ks in h.items() if not ks]
    return caught, false, h


def stage3():
    """Anomaly finder on the UNFIXED v3 model: recall of the owner's marks (review_1, review_4, review_5), false alarms, threshold tuning."""
    import cv2
    import numpy as np
    from capture import config as C, model as Mo, review as Rv
    t0, d0 = time.time(), date()
    tm = {}
    tick = lambda k, t=[time.time()]: (tm.__setitem__(k, round(time.time() - t[0], 1)), t.__setitem__(0, time.time()))
    v, fc = V2R.load_mesh(V3R.SEND / "B_plate.stl")
    m3 = Mo.load(V3R.V3 / "model_B")
    vv, _ = V2R.all_views()
    tick("load")
    regs = owner_regions(v, fc)
    owner = {k: {"verts": np.unique(fc[regs[k]["tri"]])} for k in ("review_1", "review_4", "review_5")}
    tick("owner regions")
    v2res = json.loads((V2R.SEND / "results.json").read_text())
    bump_by_video = {vid: r["agree"] for vid, r in v2res["2_bump"]["per_video"].items()}
    boxes = [("lock_bump", Mo.bump_op(m3["features"]["bump"], *(Mo.default_answers()[k] for k in ("bump_length_mm", "bump_width_mm", "bump_height_mm"))), bump_by_video)]
    for i, o in enumerate(v2res["1_grooves"]["openings"]):
        gfe = m3["features"]["grooves"][i]
        ax = np.asarray(gfe["axes"])
        boxes.append((f"slot_{i + 1}", (np.asarray(gfe["centre"]), ax, np.array([gfe["long_mm"] / 2, gfe["short_mm"] / 2, 1.0])),
                      {vid: len(r["views"]) if isinstance(r, dict) and "views" in r else int(r) for vid, r in o["by_video"].items()}))
    meas = {}
    for rad in (1.5, 2.5, 4.0):
        meas[rad] = Rv.flags(v, fc, vv, th={"lump_radius_mm": rad})
    tick("measurements (3 radii)")
    rows = []
    for rad, ms in meas.items():
        for lr in (0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4):
            for mv in (0, 1, 2, 3):
                for dg in (0.5, 0.75, 1.0, 1.5, 2.0, 1e9):
                    for ma in (2.0, 4.0, 8.0, 16.0):
                        th = {"lump_radius_mm": rad, "lump_rms_mm": lr, "min_views": mv, "disagree_mm": dg, "min_area_mm2": ma}
                        items = Rv.find(v, fc, vv, boxes, th=th, measured=ms)
                        for it in items:
                            it["fc_tri"] = fc[it["triangles"]]
                        caught, false, _ = score(items, owner, v)
                        rows.append((len(caught), -len(false), lr, ma, dg, mv, rad, caught, len(items)))
    tick("tuning grid")
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], -r[5]), reverse=True)
    best = rows[0]
    th = {"lump_radius_mm": best[6], "lump_rms_mm": best[2], "min_views": best[5], "disagree_mm": best[4], "min_area_mm2": best[3]}
    # defaults (before tuning) for comparison
    items0 = Rv.find(v, fc, vv, boxes, measured=meas[C.REVIEW_LUMP_RADIUS_MM])
    for it in items0:
        it["fc_tri"] = fc[it["triangles"]]
    c0, f0_, _ = score(items0, owner, v)
    items = Rv.find(v, fc, vv, boxes, th=th, measured=meas[th["lump_radius_mm"]])
    for it in items:
        it["fc_tri"] = fc[it["triangles"]]
    caught, false, hits = score(items, owner, v)
    tick("final items")
    # pictures: best frame crop | render of the same region, for every item
    imgs = color_crops(vv)
    vcm, fcm = coarse_mesh(Mo.apply(m3["field"], m3["grid"], m3["features"], Mo.default_answers()), m3["grid"])
    out_dir = SEND / "review_items"
    out_dir.mkdir(exist_ok=True)
    recs = []
    for it in items:
        pic, fr = Rv.picture(it, v, fc, vv, imgs, occluder=(vcm, fcm))
        if pic is not None:
            hdr = np.full((70, pic.shape[1], 3), 255, np.uint8)
            cv2.putText(hdr, f"{it['id']} {it['kind']}: {it['reason'][:70]}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(hdr, f"{it['question']}  [{' / '.join(it['choices'])}]", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{it['id']}.png"), np.vstack([hdr, pic]))
        recs.append({k: it[k] for k in ("id", "kind", "reason", "centre", "normal", "polygon_xy", "area_mm2", "question", "choices") if k in it}
                    | {"feature_id": it.get("feature_id"), "owner_marks_hit": hits[it["id"]], "picture": f"review_items/{it['id']}.png" if pic is not None else None,
                       "frame": fr, "triangles": len(it["triangles"])})
    (SEND / "review_items.json").write_text(json.dumps(recs, indent=1, default=float))
    np.savez(V4 / "review_items_tri.npz", **{it["id"]: it["triangles"] for it in items})
    (V4 / "review_thresholds.json").write_text(json.dumps(th, indent=1))
    tick("pictures")
    L = [f"STAGE 3: 'Review your model' anomaly finder (backend/capture/review.py). start {d0}. Tested on the UNFIXED v3 B_plate.stl ({len(fc)} triangles).",
         "  flags: lump (rms to the local plane of same-facing samples within R), few_views (silhouette edges within 0.3 mm), disagree (one video's 3rd "
         "strongest view puts the surface further in), one_video (a feature found in one video only); flagged vertices grouped into connected regions >= min area.",
         "  owner's marks = triangles visible inside the owner's outline in the fitted owner camera: " + ", ".join(f"{k} {len(o['verts'])} vertices" for k, o in owner.items()),
         "  hit = >= 20 % of an item's vertices within 1 mm of the mark, or >= 20 % of the mark's vertices within 1 mm of the item.",
         f"  before tuning (config placeholders): recall {len(c0)}/3 ({', '.join(c0) or 'none'}), false alarms {len(f0_)} of {len(items0)} items",
         f"  tuning grid: {len(rows)} threshold sets (R 1.5/2.5/4 mm x lump rms 0.08..0.4 x min views 0..3 x disagreement 0.5..2 mm or off x min area 2..16 mm2); "
         "ranked by recall, then fewest false alarms, then the least sensitive thresholds.",
         f"  FINAL thresholds: {th}",
         f"  FINAL: recall {len(caught)}/3 ({', '.join(caught)}), false alarms {len(false)} of {len(items)} items.",
         "  items (id, kind, area, marks hit, reason, frame):"]
    L += [f"    {r['id']} {r['kind']}: {r['area_mm2'] if r['area_mm2'] is None else round(r['area_mm2'], 1)} mm2, hits {r['owner_marks_hit'] or '-'}; {r['reason']}; "
          f"frame {r['frame']['view'] if r['frame'] else None}" for r in recs]
    L += ["  caveat: tuned and tested on the same three marks of one part, so the recall is not a held-out number; regions the owner did not mark "
          "count as false alarms even where the model is in fact wrong (e.g. the lumpy cross-bar).",
          "  pictures: review_items/<id>.png (best frame crop | render of the same region from the same camera, region outlined); review_items.json",
          "timings (s): " + ", ".join(f"{k} {v}" for k, v in tm.items()),
          f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("3_review", "\n".join(L), {"thresholds": th, "recall": caught, "false_alarms": false, "n_items": len(items), "before_tuning": {"recall": c0, "false": len(f0_), "items": len(items0)},
                                          "top_rows": [r[:7] + (r[8],) for r in rows[:15]], "items": recs, "timings": tm})
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"views": stage_views, "1": stage1, "2": stage2, "3": stage3}[a.stage]()
