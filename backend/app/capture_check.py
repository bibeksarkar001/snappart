"""Fast deterministic capture check (no model call): per frame, mat markers, sharpness, exposure, pose and whether the part box is in view;
per video, usable frames and angular coverage, with concrete tips; a combined azimuth x elevation coverage map of the part.

The pose uses an approximate camera (focal = 0.75 x the long edge, centred principal point): good enough for where the camera stood, not
for measuring. Videos: 1 = high orbit, 2 = low orbit, 3 = part flipped."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from capture import config as C, mat

ROLE = {1: "high orbit", 2: "low orbit", 3: "flipped part"}
SIDE = ["right", "far right", "far", "far left", "left", "near left", "near", "near right"]      # 45-deg sectors from +x of the mat, CCW
AZ_BINS, EL_EDGES = 12, (0, 20, 35, 50, 65, 90)
_LOCAL = threading.local()


def _detector():
    """One ArUco detector per thread (detectors are not shared across threads)."""
    if not hasattr(_LOCAL, "det"):
        _LOCAL.det = mat.make_detector()
    return _LOCAL.det


def frame_stats(jpeg: bytes, settings) -> dict:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"ok": False, "reason": "not a readable JPEG"}
    h, w = img.shape
    s = min(1.0, 1280 / max(h, w))
    g = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    ids, cs = mat.detect(_detector(), g)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean, clip_hi, clip_lo = float(g.mean()), float((g >= 250).mean()), float((g <= 5).mean())
    r = {"ok": True, "markers": int(len(ids)), "sharpness": round(lap, 1), "brightness": round(mean, 1), "clipped_bright": round(clip_hi, 3),
         "clipped_dark": round(clip_lo, 3), "size": [w, h], "az": None, "el": None, "part_in_view": False}
    if len(ids) >= 4:
        gh, gw = g.shape
        K = np.array([[0.75 * max(gw, gh), 0, gw / 2], [0, 0.75 * max(gw, gh), gh / 2], [0, 0, 1.0]])
        pose = mat.solve_pose(ids, cs, K, None)
        if pose is not None:
            R = cv2.Rodrigues(pose[0])[0]
            cam = (-R.T @ pose[1]).ravel()
            d = cam - np.array([mat.PAGE_W / 2, mat.PAGE_H / 2, 0.0])
            r["az"] = round(math.degrees(math.atan2(d[1], d[0])), 1) % 360
            r["el"] = round(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), 1)
            lo, hi = np.array(C.BOX_LO), np.array(C.BOX_HI)
            hi[2] = 25.0                                                 # a part up to 25 mm tall must be fully in frame
            box = np.array([[x, y, z] for x in (lo[0] + 10, hi[0] - 10) for y in (lo[1] + 10, hi[1] - 10) for z in (0.0, hi[2])])
            uv = cv2.projectPoints(box, pose[0], pose[1], K, None)[0].reshape(-1, 2)
            r["part_in_view"] = bool((uv[:, 0] > 0).all() and (uv[:, 0] < gw).all() and (uv[:, 1] > 0).all() and (uv[:, 1] < gh).all())
    r["blurry"] = lap < settings.blur_min_laplacian_var
    r["badly_exposed"] = mean < 50 or mean > 225 or clip_hi > 0.25 or clip_lo > 0.25
    r["usable"] = bool(r["markers"] >= 4 and r["az"] is not None and r["part_in_view"] and not r["blurry"] and not r["badly_exposed"])
    return r


def largest_gap(missing: list[int]) -> tuple[str, int]:
    """The biggest circular run of missing 30-deg azimuth bins: (side word of its centre, its width in degrees)."""
    if len(missing) >= AZ_BINS:
        return "whole", 360
    miss, best = set(missing), (0, 0)
    for b in missing:
        if (b - 1) % AZ_BINS in miss:
            continue                                   # not the start of a run
        n = 0
        while (b + n) % AZ_BINS in miss:
            n += 1
        best = max(best, (n, b))
    n, b = best
    centre = ((b + n / 2) * 30) % 360
    return SIDE[int(((centre + 22.5) % 360) // 45)], n * 30


def check(frames_by_video: dict, settings) -> dict:
    """frames_by_video: {video (1..3): {index: jpeg bytes}}. Returns the check result (verdict per video and overall, tips, coverage map)."""
    jobs = [(v, i, b) for v, fr in frames_by_video.items() for i, b in fr.items()]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as ex:            # OpenCV releases the GIL
        stats = list(ex.map(lambda j: frame_stats(j[2], settings), jobs))
    per = {v: [] for v in (1, 2, 3)}
    for (v, i, _), st in zip(jobs, stats):
        per[v].append({"index": i, **st})
    cover = np.zeros((len(EL_EDGES) - 1, AZ_BINS), int)
    videos, usable_all = {}, True
    for v in (1, 2, 3):
        fr = sorted(per[v], key=lambda f: f["index"])
        good = [f for f in fr if f.get("usable")]
        bins = sorted({int(f["az"] // 30) % AZ_BINS for f in good})
        for f in good:
            cover[min(np.searchsorted(EL_EDGES, f["el"], side="right") - 1, len(EL_EDGES) - 2), int(f["az"] // 30) % AZ_BINS] += 1
        tips = []
        if not fr:
            verdict = "missing"
            tips.append(f"Video {v} ({ROLE[v]}) is missing. Film it: {'high, looking down' if v == 1 else 'low, near table height' if v == 2 else 'with the part turned over'}, all the way round.")
        else:
            n = len(fr)
            counts = {k: sum(1 for f in fr if f.get(k)) for k in ("blurry", "badly_exposed")}
            nomat = sum(1 for f in fr if f.get("ok") and f["markers"] < 4)
            out_of_view = sum(1 for f in fr if f.get("az") is not None and not f["part_in_view"])
            unreadable = sum(1 for f in fr if not f.get("ok"))
            if unreadable:
                tips.append(f"{unreadable} of {n} frames could not be read as JPEG.")
            if nomat > n / 4:
                tips.append(f"In {nomat} of {n} frames fewer than 4 mat codes are visible. Keep more of the printed mat in the picture.")
            if counts["blurry"] > n / 4:
                tips.append(f"{counts['blurry']} of {n} frames are blurry. Move more slowly and keep the part in focus.")
            if counts["badly_exposed"] > n / 4:
                tips.append(f"{counts['badly_exposed']} of {n} frames are too dark or too bright. Use soft, even light and no flash.")
            if out_of_view > n / 4:
                tips.append(f"In {out_of_view} of {n} frames the part is not fully in view. Step back a little.")
            missing = [b for b in range(AZ_BINS) if b not in bins]
            if good and len(bins) < settings.min_coverage_bins:
                side, width = largest_gap(missing)
                secs = math.ceil(width / settings.orbit_deg_per_s)
                tips.append(f"Video {v} ({ROLE[v]}) is missing the {side} side of the part (about {width} deg of the orbit): "
                            f"film about {secs} more seconds from there.")
            els = [f["el"] for f in good]
            if els and v == 1 and float(np.median(els)) < 45:
                tips.append(f"Video 1 should look down from higher up (median elevation {np.median(els):.0f} deg; aim for 50-70).")
            if els and v == 2 and float(np.median(els)) > 45:
                tips.append(f"Video 2 should be filmed lower, near table height (median elevation {np.median(els):.0f} deg; aim for 20-40).")
            verdict = "unusable" if len(good) < settings.min_usable_frames else "weak" if len(bins) < settings.min_coverage_bins else "ok"
            if verdict == "unusable":
                tips.insert(0, f"Video {v} ({ROLE[v]}) has only {len(good)} usable frames of {n} (needs {settings.min_usable_frames}).")
        usable_all &= verdict in ("ok", "weak")
        videos[v] = {"role": ROLE[v], "verdict": verdict, "frames": len(fr), "usable_frames": len(good), "azimuth_bins_covered": bins,
                     "median_elevation_deg": round(float(np.median([f["el"] for f in good])), 1) if good else None, "tips": tips,
                     "frame_stats": fr}
    return {"usable": usable_all, "verdict": "usable" if usable_all else "unusable", "videos": videos,
            "coverage_map": {"azimuth_bin_deg": 30, "elevation_edges_deg": list(EL_EDGES), "counts": cover.tolist()},
            "method": "deterministic: ArUco markers, Laplacian sharpness, exposure, approximate pose (focal 0.75 x long edge)",
            "second_opinion": None}
