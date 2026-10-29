"""Report card for one capture run: report.json with measured numbers, feature statuses, the questions the app must ask (what the camera
cannot see), low-confidence regions, warnings and timings. rebuild(model, answers) applies the user's answers with no model call.

Feature status: measured (from the photos) | user_supplied (an answer) | estimated (a default nobody confirmed) | not_detected.
"""
import json
from pathlib import Path

import numpy as np

from . import carve, config as C
from .model import default_answers, rebuild  # noqa: F401  (rebuild is part of this module's interface)

QUESTIONS = {
    "groove_depth_mm": ("grooves", "How deep are the rectangular slots along the end of the part? The camera only sees their openings.", "mm",
                        "Push the depth rod of your calipers into one slot until it touches the bottom."),
    "bump_height_mm": ("lock_bump", "How far does the raised lock bump stand out from the surface it sits on?", "mm",
                       "Rest the calipers' depth rod on the surface next to the bump and measure to the bump's top."),
    "bump_width_mm": ("lock_bump", "How wide is the lock bump (across its short side)?", "mm", "Close the outside jaws across the bump's short side."),
    "bump_length_mm": ("lock_bump", "How long is the lock bump (end to end)?", "mm", "Close the outside jaws over the bump's two ends."),
    "plate_thickness_mm": ("plate", "How thick is the flat plate between the large window and the slot column?", "mm",
                           "Measure it in the middle of the plate with the outside jaws, not at its edge."),
}


def questions(features, answers_given=None):
    """One question per answer the geometry depends on that the photos cannot give (or gave in conflict with an answer)."""
    given = answers_given or {}
    d = default_answers()
    out = []
    for key, (fid, text, unit, hint) in QUESTIONS.items():
        if fid == "grooves" and not features.get("grooves") or fid in ("lock_bump",) and not features.get("bump") or fid == "plate" and not features.get("plate"):
            continue
        out.append({"id": key, "feature_id": fid, "question": text, "unit": unit, "default": d[key], "answered": key in given,
                    "answer": given.get(key), "hint": hint})
    return out


def low_confidence(views_moved, verts, faces, tol_mm=0.3, min_views=3, cell_mm=8.0, top=8):
    """Surface fixed by fewer than min_views silhouettes (a silhouette edge within tol_mm): share of the area and the largest 8 mm cells."""
    cnt = np.zeros(len(verts), int)
    for v in views_moved:
        cnt += np.abs(carve.view_values(v, verts)) < tol_mm
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    low = (cnt < min_views)[faces].all(1)
    cells = {}
    for c, a in zip(tri[low].mean(1), area[low]):
        k = tuple((c // cell_mm).astype(int))
        cells.setdefault(k, [0.0, []])
        cells[k][0] += a
        cells[k][1].append(c)
    regs = sorted(cells.values(), key=lambda x: -x[0])[:top]
    return {"area_share": float(area[low].sum() / area.sum()), "min_views": min_views,
            "regions": [{"centre_mm": np.mean(c, 0).round(1).tolist(), "area_mm2": round(float(a), 1)} for a, c in regs]}


def features_list(feats, answers, user_given):
    """Every feature value with its status and view count."""
    out = []
    gro = feats.get("grooves") or []
    st = lambda k: "user_supplied" if k in user_given else "estimated"
    if gro:
        out += [{"id": "grooves.count", "status": "measured", "value": len(gro), "unit": "", "views": int(sum(g["views"] for g in gro))},
                {"id": "grooves.long", "status": "measured", "value": round(float(np.mean([g["long_mm"] for g in gro])), 2), "unit": "mm", "views": int(min(g["views"] for g in gro))},
                {"id": "grooves.short", "status": "measured", "value": round(float(np.mean([g["short_mm"] for g in gro])), 2), "unit": "mm", "views": int(min(g["views"] for g in gro))},
                {"id": "grooves.depth", "status": st("groove_depth_mm"), "value": answers["groove_depth_mm"], "unit": "mm", "views": 0}]
    else:
        out.append({"id": "grooves", "status": "not_detected", "value": None, "unit": "", "views": 0})
    b = feats.get("bump")
    if b:
        out += [{"id": "lock_bump.position", "status": "measured", "value": [round(x, 2) for x in b["edge_mid"]], "unit": "mm", "views": b["agreeing_views"]},
                {"id": "lock_bump.length_detected", "status": "measured", "value": round(b["detected_length_mm"], 2), "unit": "mm", "views": b["agreeing_views"]},
                {"id": "lock_bump.length", "status": st("bump_length_mm"), "value": answers["bump_length_mm"], "unit": "mm", "views": 0},
                {"id": "lock_bump.width", "status": st("bump_width_mm"), "value": answers["bump_width_mm"], "unit": "mm", "views": 0},
                {"id": "lock_bump.height", "status": st("bump_height_mm"), "value": answers["bump_height_mm"], "unit": "mm", "views": 0},
                {"id": "lock_pin", "status": "user_supplied", "value": "part of the lock bump", "unit": "", "views": 0}]
    else:
        out.append({"id": "lock_bump", "status": "not_detected", "value": None, "unit": "", "views": 0})
    p = feats.get("plate")
    if p:
        out += [{"id": "plate.faces_separation", "status": "measured", "value": None if p.get("measured_separation_mm") is None else round(p["measured_separation_mm"], 2),
                 "unit": "mm", "views": int(sum(f["views"] for f in (p.get("faces") or p.get("measured_faces") or {}).values()))},
                {"id": "plate.thickness", "status": "measured" if answers.get("plate_thickness_from") == "measured" else st("plate_thickness_mm"),
                 "value": answers["plate_thickness_mm"], "unit": "mm", "views": 0}]
    else:
        out.append({"id": "plate", "status": "not_detected", "value": None, "unit": "", "views": 0})
    return out


def build(unfeatured_mesh, feats, answers, user_given=(), caliper=None, held_out_iou=None, views_moved=None, timings=None, warnings=()):
    """report dict. unfeatured_mesh: (verts, faces) of the plain carve (dimensions are measured on it, not on user-supplied features)."""
    v, fc = unfeatured_mesh
    pl = carve.obb_plan(v)
    meas = {"length": pl["length_mm"], "width": pl["width_mm"], "height": pl["z_max"] - pl["z_min"]}
    dims = {k: {"measured": round(x, 2), "caliper": (caliper or {}).get(k), "error": None if (caliper or {}).get(k) is None else round(x - caliper[k], 2)} for k, x in meas.items()}
    w = list(warnings)
    p = feats.get("plate")
    if p and p.get("measured_separation_mm") is not None and abs(p["measured_separation_mm"] - answers["plate_thickness_mm"]) > 0.3:
        w.append(f"plate: the photos put its two faces {p['measured_separation_mm']:.2f} mm apart, the answer is {answers['plate_thickness_mm']} mm; the answer is used. Please re-measure in the middle of the plate.")
    w.append("underside and recesses: silhouettes cannot see into concave regions (the recess under the plate between the rails, the face the slots open in), "
             "and each video never sees the face it rests on; those surfaces come from other videos or from answers.")
    return {"held_out_iou": held_out_iou, "dims_mm": dims, "features": features_list(feats, answers, set(user_given)),
            "questions": questions(feats, {k: answers[k] for k in user_given if k in answers}),
            "low_confidence_regions": low_confidence(views_moved, v, fc) if views_moved else None, "warnings": w, "timings": timings}


def write(path: Path, rep):
    Path(path).write_text(json.dumps(rep, indent=1, default=float))
