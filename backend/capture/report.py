"""Report card for one capture run: report.json with measured numbers, feature statuses, the questions the app must ask (what the camera
cannot see), low-confidence regions, warnings and timings. rebuild(model, answers) applies the user's answers with no model call.

Feature status: measured (from the photos) | user_supplied (an answer) | estimated (a default nobody confirmed) | not_detected.
"""
import json
from pathlib import Path

import numpy as np

from . import carve, config as C, learnings as L
from .model import estimate_answers, rebuild  # noqa: F401  (rebuild is part of this module's interface)
from .question_cards import CHOICES, TEXT

FEATURE_OF = {"groove_depth_mm": "grooves", "bump_height_mm": "lock_bump", "bump_width_mm": "lock_bump", "bump_length_mm": "lock_bump",
              "bump_shape": "lock_bump", "plate_thickness_mm": "plate"}     # one shape question (box / rounded / hill); bump_top_shape is
                                                                    # still accepted from older stored answers, but no longer asked
QUESTIONS = {k: (FEATURE_OF[k], *TEXT[k]) for k in FEATURE_OF}
RIDGE_ONLY = {"bump_shape"}


def questions(features, answers_given=None, height_mm=None):
    """One question per answer the geometry depends on that the photos cannot give (or gave in conflict with an answer). Each default is
    the job's own estimate (model.estimate_answers), with "default_source" measured or estimated."""
    given = answers_given or {}
    est, src = estimate_answers(features, height_mm)
    d = {**est, "bump_top_shape": next(k for k, r in C.TOP_SHAPE_RADIUS_MM.items() if r == C.BUMP_TOP_RADIUS_MM),
         "bump_shape": (features.get("bump") or {}).get("shape", "hill")}
    src = {**src, "bump_top_shape": "estimated", "bump_shape": "estimated"}
    out = []
    for key, (fid, text, unit, hint) in QUESTIONS.items():
        if fid == "grooves" and not features.get("grooves") or fid in ("lock_bump",) and not features.get("bump") or fid == "plate" and not features.get("plate"):
            continue
        if key in RIDGE_ONLY and (features.get("bump") or {}).get("kind") != "ridge":
            continue
        out.append({"id": key, "feature_id": fid, "question": text, "unit": unit, "default": d[key], "default_source": src[key], "answered": key in given,
                    "answer": given.get(key), "hint": hint, **({"choices": CHOICES[key]} if key in CHOICES else {})})
    return out


def low_confidence(views_moved, verts, faces, tol_mm=0.3, min_views=3, cell_mm=8.0, top=8, floors=()):
    """Surface fixed by fewer than min_views silhouettes (a silhouette edge within tol_mm) and not lying on a mat plane (z = 0 of the
    reference video, or a floor T: part -> another video's mat frame): share of the area and the largest 8 mm cells, each with its mean
    outward normal and how many cameras faced it (within 60 deg of the normal)."""
    cnt = np.zeros(len(verts), int)
    for v in views_moved:
        cnt += np.abs(carve.view_values(v, verts)) < tol_mm
    on_floor = np.abs(verts[:, 2]) < tol_mm
    for T in floors:
        on_floor |= np.abs(verts @ np.asarray(T)[2, :3] + np.asarray(T)[2, 3]) < tol_mm
    cnt[on_floor] = max(cnt.max(), min_views)                                 # a face resting on the mat is fixed by the mat plane
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(2 * area[:, None], 1e-12)
    low = (cnt < min_views)[faces].all(1)
    cells = {}
    for c, a, n in zip(tri[low].mean(1), area[low], fn[low]):
        k = tuple((c // cell_mm).astype(int))
        cells.setdefault(k, [0.0, [], np.zeros(3)])
        cells[k][0] += a
        cells[k][1].append(c)
        cells[k][2] += a * n
    regs = sorted(cells.values(), key=lambda x: -x[0])[:top]
    cams = np.array([-v.R.T @ v.t for v in views_moved])
    out = []
    for a, cs, n in regs:
        ctr = np.mean(cs, 0)
        n = n / max(np.linalg.norm(n), 1e-12)
        d = cams - ctr
        facing = int(((d @ n) / np.linalg.norm(d, axis=1) > 0.5).sum())
        out.append({"centre_mm": ctr.round(1).tolist(), "area_mm2": round(float(a), 1), "normal": n.round(3).tolist(), "cameras_facing": facing})
    allc = {}
    for c, a, k in zip(tri.mean(1), area, cnt[faces].mean(1)):
        key = tuple((c // cell_mm).astype(int))
        allc.setdefault(key, [0.0, [], []])
        allc[key][0] += a
        allc[key][1].append(c)
        allc[key][2].append(k)
    cmap = [{"centre_mm": np.mean(cs, 0).round(1).tolist(), "area_mm2": round(float(a), 1), "median_views": float(np.median(ks))}
            for a, cs, ks in allc.values()]
    return {"area_share": float(area[low].sum() / area.sum()), "min_views": min_views, "regions": out,
            "coverage_map": {"cell_mm": cell_mm, "cells": cmap}}


SIDE = ["right", "far right", "far", "far left", "left", "near left", "near", "near right"]


def coverage_requests(low: dict | None, orbit_deg_per_s: float = 12.0) -> tuple[list[dict], list[dict]]:
    """(requests to film more, recesses). A weakly seen area that few cameras faced needs more footage from its direction; one that
    cameras did face but silhouettes still could not fix is a recess (more footage would not help): not_detected (learnings rule 2)."""
    req, rec = [], []
    for i, r in enumerate((low or {}).get("regions") or []):
        if r["area_mm2"] < L.COVERAGE_MIN_AREA_MM2:
            continue
        n = np.asarray(r.get("normal") or [0, 0, 1.0])
        if r.get("cameras_facing", 0) >= L.FACING_CAMERAS_MIN:
            rec.append({"id": f"unseen_{i + 1}", "status": "not_detected", "centre_mm": r["centre_mm"], "area_mm2": r["area_mm2"],
                        "reason": "a recess or hidden face: cameras faced it but silhouettes cannot see into it; its shape is the carve's guess"})
            continue
        if n[2] < -0.6:
            where, video = "the underside of the part", "the flipped-part video"
        elif n[2] > 0.6:
            where, video = "the top of the part", "a high orbit"
        else:
            side = SIDE[int(((np.degrees(np.arctan2(n[1], n[0])) % 360) + 22.5) // 45) % 8]
            where, video = f"the {side} side of the part", "a low orbit, near table height"
        secs = max(5, int(np.ceil(90 / orbit_deg_per_s)))
        req.append({"id": f"film_{i + 1}", "centre_mm": r["centre_mm"], "area_mm2": r["area_mm2"], "normal": n.round(3).tolist(),
                    "cameras_facing": r.get("cameras_facing", 0), "blocking": r["area_mm2"] >= L.COVERAGE_BLOCK_MM2,
                    "message": f"The camera barely saw {where}. Film about {secs} more seconds of it ({video})."})
    return req, rec


def features_list(feats, answers, user_given, sources=None):
    """Every feature value with its status and view count. sources: where each non-user answer came from (measured / estimated)."""
    out = []
    gro = feats.get("grooves") or []
    st = lambda k: "user_supplied" if k in user_given else (sources or {}).get(k, "estimated")
    if gro:
        out += [{"id": "grooves.count", "status": "measured", "value": len(gro), "unit": "", "views": int(sum(g["views"] for g in gro))},
                {"id": "grooves.long", "status": "measured", "value": round(float(np.mean([g["long_mm"] for g in gro])), 2), "unit": "mm", "views": int(min(g["views"] for g in gro))},
                {"id": "grooves.short", "status": "measured", "value": round(float(np.mean([g["short_mm"] for g in gro])), 2), "unit": "mm", "views": int(min(g["views"] for g in gro))},
                {"id": "grooves.depth", "status": st("groove_depth_mm"), "value": answers["groove_depth_mm"], "unit": "mm", "views": 0}]
    else:
        out.append({"id": "grooves", "status": "not_detected", "value": None, "unit": "", "views": 0})
    b = feats.get("bump")
    if b and b.get("kind") == "ridge":
        out += [{"id": "lock_ridge.position", "status": "measured", "value": [round(x, 2) for x in np.mean(b["ends"], 0)], "unit": "mm", "views": b["agreeing_views"]},
                {"id": "lock_ridge.face_and_direction", "status": "measured", "value": {"face": b.get("face"), "normal": [round(x, 3) for x in b["normal"]]},
                 "unit": "", "views": b.get("extent_views", 0)},
                {"id": "lock_ridge.length_detected", "status": "measured", "value": round(b["detected_length_mm"], 2), "unit": "mm", "views": b["agreeing_views"]},
                {"id": "lock_ridge.extent_out_of_wall", "status": "measured", "value": round(b["extent_mm"], 2), "unit": "mm", "views": b.get("extent_views", 0)},
                {"id": "lock_ridge.length", "status": st("bump_length_mm"), "value": answers["bump_length_mm"], "unit": "mm", "views": 0},
                {"id": "lock_ridge.base_width", "status": st("bump_width_mm"), "value": answers["bump_width_mm"], "unit": "mm", "views": 0},
                {"id": "lock_ridge.height", "status": st("bump_height_mm"), "value": answers["bump_height_mm"], "unit": "mm", "views": 0},
                {"id": "lock_ridge.top_radius", "status": "user_supplied" if "bump_top_shape" in user_given else "estimated",
                 "value": answers.get("bump_top_radius_mm", C.BUMP_TOP_RADIUS_MM), "unit": "mm", "views": 0},
                {"id": "lock_ridge.shape", "status": "user_supplied" if "bump_shape" in user_given or b.get("shape_corrected") else "estimated",
                 "value": answers.get("bump_shape", b.get("shape", "hill")) if not b.get("shape_corrected") else b.get("shape"), "unit": "", "views": 0},
                {"id": "lock_pin", "status": "user_supplied", "value": "part of the lock ridge", "unit": "", "views": 0}]
    elif b:
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


def build(unfeatured_mesh, feats, answers, user_given=(), caliper=None, held_out_iou=None, views_moved=None, timings=None, warnings=(),
          corrections=None, review_items=None, video_checks=None, details=None, quality=None):
    """report dict. unfeatured_mesh: (verts, faces) of the plain carve (dimensions are measured on it, not on user-supplied features).
    corrections: the model edits rebuild() applies (corrections.py); review_items: the anomaly finder's items (review.py).
    video_checks: per-video detections for the learnings rules, {"registration": {feature: {video: {"point", "views"}}},
    "location": {feature: {video: {"ends", "views"}}}}."""
    v, fc = unfeatured_mesh
    pl = carve.obb_plan(v)
    meas = {"length": pl["length_mm"], "width": pl["width_mm"], "height": pl["z_max"] - pl["z_min"]}
    _, sources = estimate_answers(feats, meas["height"])
    dims = {k: {"measured": round(x, 2), "caliper": (caliper or {}).get(k), "error": None if (caliper or {}).get(k) is None else round(x - caliper[k], 2)} for k, x in meas.items()}
    w = list(warnings)
    p = feats.get("plate")
    if p and p.get("measured_separation_mm") is not None and abs(p["measured_separation_mm"] - answers["plate_thickness_mm"]) > 0.3:
        w.append(f"plate: the photos put its two faces {p['measured_separation_mm']:.2f} mm apart, the answer is {answers['plate_thickness_mm']} mm; the answer is used. Please re-measure in the middle of the plate.")
    w.append("underside and recesses: silhouettes cannot see into concave regions (pockets, the space under a plate between rails, a face a feature "
             "opens in), and each video never sees the face it rests on; those surfaces come from other videos or from answers.")
    checks = {}
    for fid, per in ((video_checks or {}).get("registration") or {}).items():          # learnings rule 3
        r = L.registration_disagreement(per)
        if r:
            checks[f"registration.{fid}"] = r
            if r["warn"]:
                w.append(f"{fid}: {r['message']}")
    for fid, per in ((video_checks or {}).get("location") or {}).items():              # learnings rule 5
        r = L.location_agreement(per)
        checks[f"location.{fid}"] = r
        if r["ask"]:
            w.append(f"{fid}: location not confirmed by two videos ({r.get('reason') or str(r.get('gap_mm')) + ' mm apart'}); please check it.")
    low = low_confidence(views_moved, v, fc) if views_moved else None
    requests, recesses = coverage_requests(low)
    return {"held_out_iou": held_out_iou, "dims_mm": dims, "features": features_list(feats, answers, set(user_given), sources),
            "questions": questions(feats, {k: answers[k] for k in user_given if k in answers}, meas["height"]),
            "low_confidence_regions": low, "not_detected": recesses, "coverage_requests": requests, "checks": checks, "warnings": w, "timings": timings,
            "corrections": list(corrections or []), "review_items": list(review_items or []), "details": list(details or []),
            "quality": quality}


def write(path: Path, rep):
    Path(path).write_text(json.dumps(rep, indent=1, default=float))
