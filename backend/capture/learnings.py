"""What the part-B runs taught, as generic named rules. Each rule says its evidence (docs/accuracy.md, docs/learnings.md) and is protected by
backend/tests/test_learnings.py. Nothing here is specific to one part: thresholds are stated once and apply to every job."""
import numpy as np

from . import config as C

MIN_FEATURE_MM = 1.0             # rule 6: features (and measured sizes) under about 1 mm are never claimed
REGISTRATION_WARN_MM = 0.5       # rule 3: videos that place the same feature further apart than this are reported
PLATE_CONFLICT_MM = 0.3          # rule 4: a user's caliper value this far from the video's is a conflict to ask about
LOCATION_AGREE_MM = 1.5          # rule 5: a feature located this far apart by two videos is asked about
COVERAGE_MIN_AREA_MM2 = 20.0     # rule 2: weakly seen areas smaller than this are not reported
COVERAGE_BLOCK_MM2 = 40.0        # rule 2: a weakly seen area at least this large that few cameras faced blocks "done" until filmed or skipped
FACING_CAMERAS_MIN = 3           # rule 2: cameras within 60 deg of an area's normal needed before a weak area counts as a recess


def k_for_views(n_views: int) -> int:
    """Rule 1a: the robust carve's k (views that must agree a voxel is outside) scales with the number of views.
    Evidence: k = 12 was tuned on held-out IoU at 229 views (v1 stage 5); with 114 browser-sized views k = 12 kept the hull fat
    (held-out IoU 0.916) and k = 6 gave 0.936 (v3 stage 3c)."""
    return max(2, round(C.K_ROBUST * n_views / C.K_TUNED_AT_VIEWS))


def voxel_mm(n_views: int) -> float:
    """Rule 1b: carve voxel size: 0.15 mm in production at any view count (chosen in v1 stage 5). Measured on part B: held-out IoU 0.9529
    (3 folds) at 0.15 mm; the golden test's single fold at 0.25 mm gives 0.9512. Finer voxels were not measured. Kept as a rule so a
    future view-count scaling has one place to live."""
    return C.VOXEL_MM


def not_detected_regions(low_confidence: dict | None, min_area_mm2: float = COVERAGE_MIN_AREA_MM2) -> list[dict]:
    """Rule 2: silhouettes cannot see into recesses, so surface fixed by fewer than 3 silhouettes is reported not_detected (with the
    reason) and asked about, never presented as measured. Evidence: on part B the face the slots open in lies about 2 mm inside the carve
    and the pocket under the plate was filled; 2.8 % of the surface is fixed by fewer than 3 views."""
    regs = (low_confidence or {}).get("regions") or []
    return [{"id": f"unseen_{i + 1}", "status": "not_detected", "centre_mm": r["centre_mm"], "area_mm2": r["area_mm2"],
             "reason": "too few camera views see this area (a recess or a hidden face); its shape is the carve's guess"}
            for i, r in enumerate(regs) if r["area_mm2"] >= min_area_mm2]


def registration_disagreement(per_video: dict) -> dict | None:
    """Rule 3: when two videos place the same feature at different positions, report the gap and use the video with more views.
    per_video: {video: {"point": [x, y, z], "views": n}}. Evidence: videos 2 and 3 put the same four slot openings 1.58-1.72 mm apart
    in height on part B."""
    if len(per_video) < 2:
        return None
    vids = sorted(per_video, key=lambda v: -per_video[v]["views"])
    pts = np.array([per_video[v]["point"] for v in vids], float)
    gap = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
    return {"gap_mm": round(gap, 2), "use": vids[0], "warn": gap > REGISTRATION_WARN_MM,
            "message": f"The videos disagree by {gap:.2f} mm about where this feature is; the one with the most views ({vids[0]}) is used."
            if gap > REGISTRATION_WARN_MM else None}


def plate_conflict(measured_mm: float | None, answer_mm: float | None) -> dict | None:
    """Rule 4: a thickness measured by plane sweeps that disagrees with the user's caliper value is a conflict: ask again and show both,
    never silently pick one. Evidence: part B's plate, video 0.2-0.9 mm vs caliper 2.6 mm (v3, v4)."""
    if measured_mm is None or answer_mm is None or abs(measured_mm - answer_mm) <= PLATE_CONFLICT_MM:
        return None
    return {"candidates": [{"label": "video", "value": round(measured_mm, 2)}, {"label": "your answer", "value": round(answer_mm, 2)}],
            "message": f"The video measured {measured_mm:.2f} mm but you entered {answer_mm:.2f} mm. Please check which is right "
                       "(measure in the middle, not at an edge)."}


def location_agreement(per_video: dict) -> dict:
    """Rule 5: a feature's location (e.g. a ridge's end points) must agree between videos that found it; when they disagree, or only
    one video found it, the user is asked which side / where it is. per_video: {video: {"ends": [[x,y,z], [x,y,z]], "views": n}} with at
    least 3 agreeing views to count. Evidence: part B's ridge was found by video 3 (29 views) only; video 2 had 2 agreeing views."""
    found = {v: d for v, d in per_video.items() if d.get("views", 0) >= 3}
    if len(found) < 2:
        return {"agree": False, "ask": True, "reason": "found in one video only" if found else "not found by any video",
                "videos": sorted(found)}
    ends = [np.asarray(d["ends"], float) for d in found.values()]
    gap = max(min(np.linalg.norm(a - b, axis=1).max(), np.linalg.norm(a - b[::-1], axis=1).max()) for a in ends for b in ends)
    return {"agree": gap <= LOCATION_AGREE_MM, "ask": gap > LOCATION_AGREE_MM, "gap_mm": round(float(gap), 2), "videos": sorted(found)}


def claimable(size_mm: float | None) -> bool:
    """Rule 6: a measured size under MIN_FEATURE_MM is not claimed as measured (it is reported as an estimate and asked). Evidence: the
    silhouettes' edge precision is about 0.2-0.3 mm and the plate sweeps' spread 0.2-0.5 mm; below about 1 mm a single error is a
    large fraction of the value."""
    return size_mm is not None and size_mm >= MIN_FEATURE_MM
