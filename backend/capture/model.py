"""Editable part model: the carve field plus feature operations whose sizes are answers (measured or user-supplied).

rebuild(model, answers, corrections) re-applies every feature and correction to the stored carve field and re-meshes: no model call, no
images. Features (all in the part frame):
  grooves: pockets (centre, axes rows = long / short / outward normal, opening size measured), depth = answers["groove_depth_mm"]
  bump:    v3: block on the groove face, anchored on its detected far edge line.
           v4 (kind "ridge"): extruded profile (box | rounded | hill) standing on a face: base_mid on the face, axis along its length,
           across along the base width, normal outward. Length / base width / height / top radius from answers.
  plate:   flat slab over a footprint (2D mask, or polygon_xy) between its outer plane (normal, d) and the plane `thickness` inside it
Corrections (a list of dicts, see corrections.py for the JSON schema) are applied after the plate and before grooves and bumps.
Field convention: carve field, inside positive (mm), zero level = surface.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from . import carve, config as C
from .features import box_sdf


def _sub(g, shape, lo, hi):
    """Index slices and points of the grid's sub-box covering [lo, hi] (mm)."""
    i0 = np.clip(np.floor((np.asarray(lo) - g.lo) / g.vs).astype(int), 0, np.array(shape) - 1)
    i1 = np.clip(np.ceil((np.asarray(hi) - g.lo) / g.vs).astype(int) + 1, 1, np.array(shape))
    sl = tuple(slice(a, b) for a, b in zip(i0, i1))
    ax = [g.lo[k] + g.vs * np.arange(i0[k], i1[k]) for k in range(3)]
    return sl, np.stack(np.meshgrid(*ax, indexing="ij"), -1)


def oriented_box(field, g, c, axes, half, add, margin=3.0):
    """Union (add) or subtraction of an oriented box with the field, evaluated only within margin mm of the box (the field is clipped at
    +-3 mm, so nothing changes further away). An added box must fit inside the grid (it would be clipped otherwise)."""
    sg = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    P = c + (sg * half) @ axes
    if add:
        hi = g.lo + (np.array(field.shape) - 1) * g.vs
        if (P < g.lo).any() or (P > hi).any():
            raise ValueError(f"added feature reaches {P.min(0).round(2).tolist()}..{P.max(0).round(2).tolist()}, outside the model grid {g.lo.round(2).tolist()}..{hi.round(2).tolist()}")
    sl, X = _sub(g, field.shape, P.min(0) - margin, P.max(0) + margin)
    d = box_sdf(X.reshape(-1, 3), c, axes, half).astype(np.float32).reshape(X.shape[:3])
    out = field.copy()
    out[sl] = np.maximum(field[sl], -d) if add else np.minimum(field[sl], d)
    return out


def groove_ops(grooves, depth, out_mm=3.0):
    """(centre, axes, half) of each pocket: opening rectangle on its plane, depth into the part, out_mm outward."""
    out = []
    for gv in grooves:
        axes, n = np.asarray(gv["axes"], float), np.asarray(gv["axes"], float)[2]
        out.append((np.asarray(gv["centre"], float) + n * (out_mm - depth) / 2, axes, np.array([gv["long_mm"] / 2, gv["short_mm"] / 2, (depth + out_mm) / 2])))
    return out


def bump_op(b, length, width, height, embed_mm=1.0):
    """Block centred on the detected edge's midpoint along its length; across: from the detected far edge toward the openings by `width`;
    normal: from embed_mm inside the surface to `height` outside it."""
    a = np.asarray(b["axis"], float)
    w = np.asarray(b["toward_openings"], float)
    n = np.asarray(b["normal"], float)
    c = np.asarray(b["edge_mid"], float) + w * width / 2 + n * (height - embed_mm) / 2
    return c, np.stack([a, w, n]), np.array([length / 2, width / 2, (height + embed_mm) / 2])


def poly_mask(g, shape, poly):
    """(nx, ny) bool: grid columns whose (x, y) lies inside the polygon [[x, y], ...] (mm)."""
    m = np.zeros((shape[1], shape[0]), np.uint8)
    pts = np.round((np.asarray(poly, float) - g.lo[:2]) / g.vs * 16).astype(np.int32)
    cv2.fillPoly(m, [pts], 1, cv2.LINE_8, 4)
    return m.T.astype(bool)


def plate_op(field, g, p, thickness):
    """Inside the plate's footprint region (2D mask over the grid's x, y, or polygon_xy), replace the field by the slab between the outer
    plane and the plane `thickness` inside it (signed distance to the slab, clipped like the carve field)."""
    m = np.asarray(p["mask"], bool) if "mask" in p else poly_mask(g, field.shape, p["polygon_xy"])
    ix, iy = np.nonzero(m)
    Z = g.lo[2] + g.vs * np.arange(field.shape[2])
    X = np.stack([g.lo[0] + g.vs * ix, g.lo[1] + g.vs * iy], 1)
    n = np.asarray(p["normal"], float)                 # outward (from the plate into the air on the outer side)
    s = (X @ n[:2])[:, None] + Z[None, :] * n[2] + p["d"]   # signed distance from the outer plane, + outward
    out = field.copy()
    slab = np.clip(np.minimum(-s, s + thickness), -3, 3).astype(np.float32)
    out[ix, iy] = np.minimum(slab, field[ix, iy])      # the carve is an outer bound: a slab never adds material outside it (e.g. over a bevel)
    return out


def profile(shape, width, height, top_r, embed=1.0, n_arc=17):
    """Cross-section polygon [(u across, v outward)] of a ridge: base from -width/2 to width/2 at v = 0, extended embed mm into the part.
    box: rectangle; rounded: sides straight up, top a half circle of radius width/2; hill: straight sides tangent from the base corners to a
    top circle of radius top_r (a trapezoid / triangle with a rounded peak)."""
    b = width / 2
    if shape == "box":
        return np.array([(-b, -embed), (b, -embed), (b, height), (-b, height)])
    r = b if shape == "rounded" else min(top_r, b, height)
    c = np.array([0.0, height - r])
    if shape == "rounded":
        a0, a1 = 0.0, np.pi
        side = [(b, -embed)]
    else:
        d = np.array([b, 0.0]) - c
        L = np.linalg.norm(d)
        ang = np.arctan2(d[1], d[0]) + np.arccos(min(r / L, 1.0))      # tangent point from the right base corner, upper side
        a0, a1 = ang, np.pi - ang
        side = [(b, -embed), (b, 0.0)]
    arc = [tuple(c + r * np.array([np.cos(t), np.sin(t)])) for t in np.linspace(a0, a1, n_arc)]
    return np.array(side + arc + [(-u, v) for u, v in side[::-1]])


def side_slope_deg(poly, width):
    """Angle of a hill's straight side from the base plane (degrees)."""
    p0, p1 = poly[1], poly[2]
    return float(np.degrees(np.arctan2(p1[1] - p0[1], abs(p1[0] - p0[0]) + 1e-12)))


def sd_poly(u, v, poly):
    """Signed 2D distance (negative inside) of points (u, v) to a closed polygon."""
    P = np.stack([u, v], -1)
    d = np.full(u.shape, np.inf)
    s = np.ones(u.shape)
    for i in range(len(poly)):
        a, b = poly[i - 1], poly[i]
        e, w = b - a, P - a
        q = w - e * np.clip((w @ e) / (e @ e), 0, 1)[..., None]
        d = np.minimum(d, (q ** 2).sum(-1))
        c1, c2, c3 = P[..., 1] >= a[1], P[..., 1] < b[1], e[0] * w[..., 1] > e[1] * w[..., 0]
        s = np.where((c1 & c2 & c3) | (~c1 & ~c2 & ~c3), -s, s)
    return s * np.sqrt(d)


def ridge_op(field, g, r, length, width, height, top_r, embed=1.0, margin=3.0):
    """Union of an extruded ridge profile with the field (evaluated within margin mm of it). r: base_mid, axis, across, normal, shape."""
    c, a, w, n = (np.asarray(r[k], float) for k in ("base_mid", "axis", "across", "normal"))
    poly = profile(r.get("shape", "hill"), width, height, top_r, embed)
    sg = np.array([[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (0, 1)], float)
    P = c + sg[:, :1] * a * length / 2 + sg[:, 1:2] * w * width / 2 + (sg[:, 2:] * (height + embed) - embed) * n
    hi = g.lo + (np.array(field.shape) - 1) * g.vs
    if (P < g.lo).any() or (P > hi).any():
        raise ValueError(f"added feature reaches {P.min(0).round(2).tolist()}..{P.max(0).round(2).tolist()}, outside the model grid {g.lo.round(2).tolist()}..{hi.round(2).tolist()}")
    sl, X = _sub(g, field.shape, P.min(0) - margin, P.max(0) + margin)
    Q = X - c
    d2 = sd_poly(Q @ w, Q @ n, poly)
    da = np.abs(Q @ a) - length / 2
    sd = np.sqrt(np.maximum(d2, 0) ** 2 + np.maximum(da, 0) ** 2) + np.minimum(np.maximum(d2, da), 0)
    out = field.copy()
    out[sl] = np.maximum(field[sl], np.clip(-sd, -3, 3).astype(np.float32))
    return out


def plane_of(p, feats):
    """A plane given as {"n", "d"} or by name (feats["planes"][name]): (unit outward normal, d) with n.x + d = 0."""
    p = feats["planes"][p] if isinstance(p, str) else p
    n = np.asarray(p["n"], float)
    return n / np.linalg.norm(n), float(p["d"]) / np.linalg.norm(n)


def remove_region_op(field, g, region):
    """Remove material inside a prism: region polygon_xy (columns) between z_mm[0] and z_mm[1] (default: all z). Signed distance to the
    prism (2D distance from distance transforms of the column mask), so the cut surface lies on the region boundary, not on a voxel face."""
    m = poly_mask(g, field.shape, region["polygon_xy"]).astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d2 = (np.where(m > 0, -(din - 0.5), dout - 0.5) * g.vs)[..., None]
    z0, z1 = region.get("z_mm") or (-1e9, 1e9)
    Z = g.lo[2] + g.vs * np.arange(field.shape[2])
    dz = np.maximum(z0 - Z, Z - z1)[None, None, :]
    sd = np.sqrt(np.maximum(d2, 0) ** 2 + np.maximum(dz, 0) ** 2) + np.minimum(np.maximum(d2, dz), 0)
    return np.minimum(field, np.clip(sd, -3, 3).astype(np.float32))


def field_at(field, g, X):
    return map_coordinates(field, ((np.atleast_2d(X) - g.lo) / g.vs).T, order=1, mode="nearest")


def opposite_face(field, g, base, normal, step=0.05, max_mm=30.0):
    """Where a feature standing on a face at base (outward normal) would stand on the other side of the material: march from base into the
    part along -normal until the field turns outside. Returns (new base point, new outward normal, material thickness)."""
    n = np.asarray(normal, float)
    ts = np.arange(step, max_mm, step)
    f = field_at(field, g, np.asarray(base, float) - ts[:, None] * n)
    k = int(np.argmax(f < 0)) if (f < 0).any() else None
    if k is None or k == 0:
        raise ValueError("no material behind the feature's face: cannot move it to the opposite face")
    t = ts[k - 1] + step * f[k - 1] / (f[k - 1] - f[k])                 # zero crossing
    return np.asarray(base, float) - t * n, -n, float(t)


def correct_features(field, g, feats, corrections):
    """Feature-level corrections (move to the opposite face, set shape, add / replace a feature): returns new features (input unchanged)."""
    feats = {k: (dict(v) if isinstance(v, dict) else v) for k, v in feats.items()}
    for c in corrections or []:
        op = c["op"]
        if op == "add_feature":
            feats[c["feature_id"]] = dict(c["feature"])
        elif op == "set_feature_shape":
            b = feats[c["feature_id"]]
            b.update(shape=c["shape"], shape_corrected=True, **({"top_radius_mm": c["top_radius_mm"]} if c.get("top_radius_mm") is not None else {}))
        elif op == "move_feature_to_opposite_face":
            b = feats[c["feature_id"]]
            p, n, t = opposite_face(field, g, b["base_mid"], b["normal"])
            b.update(base_mid=p.tolist(), normal=n.tolist(), across=(-np.asarray(b["across"], float)).tolist(), moved_through_mm=t)
        elif op == "flatten_region" and c.get("feature_id"):              # re-places a slab feature (the plate); its thickness stays an answer
            n, d = plane_of(c["to_plane"], feats)
            P = np.asarray(c["region"]["polygon_xy"], float)
            feats[c["feature_id"]] = {"polygon_xy": P.tolist(), "normal": n.tolist(), "d": d,
                                      "region_xy": [P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()]}
    return feats


def region_corrections(field, g, feats, corrections, answers):
    """Region-level corrections (flatten_region, remove_region) applied to the field in order."""
    f = field
    for c in corrections or []:
        if c["op"] == "flatten_region" and not c.get("feature_id"):
            n, d = plane_of(c["to_plane"], feats)
            th = c.get("thickness_mm") or answers.get(c.get("thickness_answer", "plate_thickness_mm"))
            f = plate_op(f, g, {"polygon_xy": c["region"]["polygon_xy"], "normal": n, "d": d}, th)
        elif c["op"] == "remove_region":
            f = remove_region_op(f, g, c["region"])
    return f


def ridge_placed(b, width):
    """A pipeline ridge is anchored on the wall it stands on (wall_mid): its base midpoint is wall_mid + across * width / 2, so answering
    the base width keeps the ridge's wall side where the video put it. Other features are returned unchanged."""
    if b and b.get("kind") == "ridge" and "base_mid" not in b and "wall_mid" in b:
        return {**b, "base_mid": (np.asarray(b["wall_mid"], float) + np.asarray(b["across"], float) * width / 2).tolist()}
    return b


def apply(field, g, feats, answers, corrections=None):
    if feats.get("bump"):
        feats = {**feats, "bump": ridge_placed(feats["bump"], answers.get("bump_width_mm", 0.0))}
    feats = correct_features(field, g, feats, corrections)
    f = field
    if feats.get("plate") and answers.get("plate_thickness_mm"):
        f = plate_op(f, g, feats["plate"], answers["plate_thickness_mm"])
    f = region_corrections(f, g, feats, corrections, answers)
    for c, axes, half in groove_ops(feats.get("grooves", []), answers["groove_depth_mm"]) if feats.get("grooves") and answers.get("groove_depth_mm") else []:
        f = oriented_box(f, g, c, axes, half, add=False)
    b = feats.get("bump")
    if b and answers.get("bump_height_mm"):
        if b.get("kind") == "ridge":
            top_r = b["top_radius_mm"] if "top_radius_mm" in b else answers.get("bump_top_radius_mm", C.BUMP_TOP_RADIUS_MM)   # a correction wins
            if not b.get("shape_corrected") and answers.get("bump_shape"):                                               # ... over the answer
                b = {**b, "shape": answers["bump_shape"]}
            f = ridge_op(f, g, b, answers["bump_length_mm"], answers["bump_width_mm"], answers["bump_height_mm"], top_r)
        else:
            f = oriented_box(f, g, *bump_op(b, answers["bump_length_mm"], answers["bump_width_mm"], answers["bump_height_mm"]), add=True)
    return f


def mesh(field, g):
    v, fc = carve.to_mesh(field, g)
    return carve.taubin(v, fc, C.TAUBIN_ITERS), fc


def part_height(field, g) -> float | None:
    """Height (mm) of the carved part: z extent of the inside voxels."""
    zs = np.nonzero((field > 0).any((0, 1)))[0]
    return float((zs[-1] - zs[0]) * g.vs) if len(zs) else None


def estimate_answers(features=None, height_mm=None) -> tuple[dict, dict]:
    """Per-job defaults for every answer the geometry needs, and where each came from ("measured" from the video, or "estimated").
    A measured value under learnings.MIN_FEATURE_MM is not claimed: it becomes an estimate and the question stays open."""
    from .learnings import MIN_FEATURE_MM
    f = features or {}
    h = height_mm or 10.0
    a, src = {}, {}

    def put(k, v, measured):
        a[k] = round(float(v), 3)
        src[k] = "measured" if measured and v >= MIN_FEATURE_MM else "estimated"
    put("groove_depth_mm", 0.5 * h, False)                             # the camera cannot see a pocket's depth
    b = f.get("bump") or {}
    put("bump_length_mm", b.get("detected_length_mm", 0.5 * h), "detected_length_mm" in b)
    ext = b.get("extent_mm")
    put("bump_height_mm", ext if ext else 1.0, bool(ext))
    put("bump_width_mm", a["bump_height_mm"], False)                    # a hill about as wide as it is high, until someone measures it
    put("bump_top_radius_mm", C.BUMP_TOP_RADIUS_MM, False)
    sep = (f.get("plate") or {}).get("measured_separation_mm")
    put("plate_thickness_mm", sep if sep else 0.25 * h, bool(sep))
    if sep and sep < MIN_FEATURE_MM:
        a["plate_thickness_mm"] = round(max(float(sep), 0.25 * h), 3)     # too thin to claim from the video
    return a, src


def default_answers(features=None, height_mm=None) -> dict:
    return estimate_answers(features, height_mm)[0]


def save(path: Path, field, g, feats):
    """field + grid in <path>.npz, features in <path>.json (plate mask stored in the npz)."""
    feats = dict(feats)
    extra = {}
    if feats.get("plate") and "mask" in feats["plate"]:
        p = dict(feats["plate"])
        extra["plate_mask"] = np.asarray(p.pop("mask"), bool)
        feats["plate"] = p
    np.savez_compressed(path.with_suffix(".npz"), field=field, lo=g.lo, vs=g.vs, shape=np.array(g.shape), **extra)
    path.with_suffix(".json").write_text(json.dumps(feats, indent=1, default=lambda o: np.asarray(o).tolist()))


def load(path: Path):
    z = np.load(path.with_suffix(".npz"), allow_pickle=False)
    g = carve.Grid(z["lo"], z["lo"] + float(z["vs"]), float(z["vs"]))
    g.shape = tuple(int(x) for x in z["shape"])
    feats = json.loads(path.with_suffix(".json").read_text())
    if "plate_mask" in z:
        feats["plate"]["mask"] = z["plate_mask"]
    return {"field": z["field"], "grid": g, "features": feats}


def resolve_answers(answers, features=None, height_mm=None):
    """Answers as apply() uses them: the job's estimates for missing keys, the top-shape choice turned into a radius."""
    a = {**default_answers(features, height_mm), **(answers or {})}
    if a.get("bump_top_shape"):                                       # the card's choice -> radius
        a["bump_top_radius_mm"] = C.TOP_SHAPE_RADIUS_MM[a["bump_top_shape"]]
    return a


def rebuild(model, answers, corrections=None):
    """Apply the answers (any missing key falls back to the frozen defaults) and the corrections (in order) to the stored carve field and
    re-mesh. Returns (verts, faces)."""
    a = resolve_answers(answers, model["features"], part_height(model["field"], model["grid"]))
    return mesh(apply(model["field"], model["grid"], model["features"], a, corrections), model["grid"])
