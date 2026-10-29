"""Editable part model: the carve field plus feature operations whose sizes are answers (measured or user-supplied).

rebuild(model, answers) re-applies every feature to the stored carve field and re-meshes: no model call, no images. Features (all in the part frame):
  grooves: pockets (centre, axes rows = long / short / outward normal, opening size measured), depth = answers["groove_depth_mm"]
  bump:    block on the groove face, anchored on its detected far edge line; length / width / height from answers
  plate:   flat slab over a footprint region between two parallel planes; thickness from answers, anchored on the outer (mat-side) face
Field convention: carve field, inside positive (mm), zero level = surface.
"""
import json
from pathlib import Path

import numpy as np

from . import carve, config as C
from .features import box_sdf


def oriented_box(field, g, c, axes, half, add):
    """Union (add) or subtraction of an oriented box with the field. An added box must fit inside the grid (it would be clipped otherwise)."""
    if add:
        sg = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
        P = c + (sg * half) @ axes
        hi = g.lo + (np.array(g.shape) - 1) * g.vs
        if (P < g.lo).any() or (P > hi).any():
            raise ValueError(f"added feature reaches {P.min(0).round(2).tolist()}..{P.max(0).round(2).tolist()}, outside the model grid {g.lo.round(2).tolist()}..{hi.round(2).tolist()}")
    d = box_sdf(g.points(), c, axes, half).astype(np.float32).reshape(field.shape)
    return np.maximum(field, -d) if add else np.minimum(field, d)


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


def plate_op(field, g, p, thickness):
    """Inside the plate's footprint region (2D mask over the grid's x, y), replace the field by the slab between the outer plane and the
    plane `thickness` inside it (signed distance to the slab, clipped like the carve field)."""
    X = g.points().reshape(*field.shape, 3)
    n = np.asarray(p["normal"], float)                 # outward (from the plate into the air on the outer side)
    s = X @ n + p["d"]                                 # signed distance from the outer plane, + outward
    slab = np.clip(np.minimum(-s, s + thickness), -3, 3).astype(np.float32)
    m = np.asarray(p["mask"], bool)                    # (nx, ny): columns of the grid inside the plate region
    out = field.copy()
    out[m] = slab[m]
    return out


def apply(field, g, feats, answers):
    f = field
    if feats.get("plate") and answers.get("plate_thickness_mm"):
        f = plate_op(f, g, feats["plate"], answers["plate_thickness_mm"])
    for c, axes, half in groove_ops(feats.get("grooves", []), answers["groove_depth_mm"]) if feats.get("grooves") and answers.get("groove_depth_mm") else []:
        f = oriented_box(f, g, c, axes, half, add=False)
    if feats.get("bump") and answers.get("bump_height_mm"):
        f = oriented_box(f, g, *bump_op(feats["bump"], answers["bump_length_mm"], answers["bump_width_mm"], answers["bump_height_mm"]), add=True)
    return f


def mesh(field, g):
    v, fc = carve.to_mesh(field, g)
    return carve.taubin(v, fc, C.TAUBIN_ITERS), fc


def default_answers():
    return {"groove_depth_mm": C.GROOVE_DEPTH_MM, "bump_length_mm": C.BUMP_LENGTH_USER_MM, "bump_width_mm": C.BUMP_WIDTH_USER_MM,
            "bump_height_mm": C.BUMP_HEIGHT_USER_MM, "plate_thickness_mm": C.PLATE_THICKNESS_USER_MM}


def save(path: Path, field, g, feats):
    """field + grid in <path>.npz, features in <path>.json (plate mask stored in the npz)."""
    feats = dict(feats)
    extra = {}
    if feats.get("plate"):
        p = dict(feats["plate"])
        extra["plate_mask"] = np.asarray(p.pop("mask"), bool)
        feats["plate"] = p
    np.savez_compressed(path.with_suffix(".npz"), field=field, lo=g.lo, vs=g.vs, shape=np.array(g.shape), **extra)
    path.with_suffix(".json").write_text(json.dumps(feats, indent=1, default=lambda o: np.asarray(o).tolist()))


def load(path: Path):
    z = np.load(path.with_suffix(".npz"))
    g = carve.Grid(z["lo"], z["lo"] + float(z["vs"]), float(z["vs"]))
    g.shape = tuple(int(x) for x in z["shape"])
    feats = json.loads(path.with_suffix(".json").read_text())
    if "plate_mask" in z:
        feats["plate"]["mask"] = z["plate_mask"]
    return {"field": z["field"], "grid": g, "features": feats}


def rebuild(model, answers):
    """Apply the answers (any missing key falls back to the frozen defaults) to the stored carve field and re-mesh. Returns (verts, faces)."""
    a = {**default_answers(), **(answers or {})}
    return mesh(apply(model["field"], model["grid"], model["features"], a), model["grid"])
