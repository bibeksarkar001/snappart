"""Frames from an orbit video: marker detection, per-video intrinsics, per-frame poses, view selection, undistorted extraction."""
import pickle
from pathlib import Path

import cv2
import numpy as np

from . import mat

CENTRE = np.array([mat.PAGE_W / 2, mat.PAGE_H / 2, 0.0])


def marker_sharpness(gray, corners) -> float:
    """Median over markers of (95th percentile gradient magnitude / local contrast): about 1 / blur-sigma, independent of marker size."""
    out = []
    for c in corners:
        x0, y0 = np.floor(c.min(0)).astype(int) - 2
        x1, y1 = np.ceil(c.max(0)).astype(int) + 2
        p = gray[max(y0, 0):y1, max(x0, 0):x1].astype(np.float32)
        if p.size < 100 or p.shape[0] < 8 or p.shape[1] < 8:
            continue
        g = np.hypot(cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        con = np.percentile(p, 95) - np.percentile(p, 5)
        if con > 20:
            out.append(np.percentile(g, 95) / con)
    return float(np.median(out)) if out else 0.0


def scan(path: Path, stride=3, cache: Path | None = None, limit=None, log=print) -> dict:
    """Detect markers on every `stride`-th frame. {"n_read", "size", "fps", "frames": {idx: (ids, corners, sharp)}}; cached.
    The cache is pickle, so it is only for internal runs on the owner's own videos (spikes, golden test): the API never calls this, and
    tests/test_security.py fails if app/ ever does."""
    if cache is not None and cache.exists():
        return pickle.loads(cache.read_bytes())
    cap = cv2.VideoCapture(str(path))
    det = mat.make_detector()
    frames, i = {}, 0
    size = None
    while True:
        if i % stride:
            ok = cap.grab()
        else:
            ok, fr = cap.read()
        if not ok or (limit and i >= limit):
            break
        if i % stride == 0:
            size = (fr.shape[1], fr.shape[0])
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ids, cs = mat.detect(det, gray)
            frames[i] = (ids, cs, marker_sharpness(gray, cs) if len(ids) else 0.0)
            if (i // stride) % 100 == 0:
                log(f"  {path.name}: frame {i}, {len(ids)} markers")
        i += 1
    res = {"n_read": i, "size": size, "fps": cap.get(cv2.CAP_PROP_FPS), "stride": stride, "frames": frames}
    if cache is not None:
        cache.write_bytes(pickle.dumps(res))
    return res


def initial_K(size):
    f = 0.73 * size[0]                      # ~69 deg horizontal field of view, a phone main camera
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])


def calibrate(sc: dict, n_views=80, min_markers=10) -> dict:
    """cv2.calibrateCamera (k1, k2, p1, p2; k3 fixed) from the best-spread frames with many markers."""
    cand = [(i, f) for i, f in sc["frames"].items() if len(f[0]) >= min_markers]
    if len(cand) < 10:
        raise RuntimeError(f"only {len(cand)} frames with >= {min_markers} markers")
    sharp = np.array([f[2] for _, f in cand])
    cand = [c for c, s in zip(cand, sharp) if s >= np.median(sharp)]
    pick = [cand[j] for j in np.unique(np.linspace(0, len(cand) - 1, min(n_views, len(cand))).astype(int))]
    obj = [mat.correspondences(f[0], f[1])[0].astype(np.float32) for _, f in pick]
    img = [mat.correspondences(f[0], f[1])[1].astype(np.float32) for _, f in pick]
    K0 = initial_K(sc["size"])
    rms, K, dist, _, _, sd_int, _, per_view = cv2.calibrateCameraExtended(
        obj, img, sc["size"], K0, np.zeros(5), flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
    return {"K": K, "dist": dist.ravel(), "rms_px": float(rms), "n_views": len(pick), "frames": [i for i, _ in pick],
            "K_std": sd_int.ravel()[:4].tolist(), "dist_std": sd_int.ravel()[4:8].tolist()}


def poses(sc: dict, K, dist, min_markers=4, max_rms=1.0, world=None) -> dict:
    """Per-frame world->camera pose. {idx: dict(rvec, tvec, rms, n, ok, why)} for every frame with detections."""
    out = {}
    for i, (ids, cs, sharp) in sc["frames"].items():
        if len(ids) == 0:
            continue
        r = {"n_detected": int(len(ids)), "sharp": sharp, "ok": False, "why": "fewer than 4 markers"}
        if len(ids) >= min_markers:
            sol = mat.solve_pose(ids, cs, K, dist, world=world)
            if sol is None:
                r["why"] = "pose failed"
            else:
                rv, tv, rms, used = sol
                r.update(rvec=rv.ravel(), tvec=tv.ravel(), rms=rms, n=int(len(used)))
                r["ok"], r["why"] = (len(used) >= min_markers and rms <= max_rms), ("" if rms <= max_rms and len(used) >= min_markers else f"rms {rms:.2f} px or too few markers after outlier pass")
        out[i] = r
    return out


def reaccept(ps: dict, max_rms: float, min_markers=4) -> int:
    """Re-apply the acceptance rule with a different rms threshold, in place; returns the accepted count."""
    for p in ps.values():
        if "rms" in p:
            p["ok"] = p["n"] >= min_markers and p["rms"] <= max_rms
    return sum(p["ok"] for p in ps.values())


def viewpoint(rvec, tvec, centre=CENTRE):
    """(azimuth deg 0-360, elevation deg, distance mm) of the camera as seen from the mat centre."""
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    c = -R.T @ np.asarray(tvec, float) - centre
    return float(np.degrees(np.arctan2(c[1], c[0])) % 360), float(np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1])))), float(np.linalg.norm(c))


def centre_visible(rvec, tvec, K, size, margin=0.12, centre=CENTRE + [0, 0, 5]) -> bool:
    p = cv2.projectPoints(centre.reshape(1, 3), np.asarray(rvec, float), np.asarray(tvec, float), K, None)[0].ravel()
    return bool(margin * size[0] <= p[0] <= (1 - margin) * size[0] and margin * size[1] <= p[1] <= (1 - margin) * size[1])


def select(ps: dict, K, size, n=80, az_bins=16, el_bins=3, min_gap_frames=0) -> list[int]:
    """~n sharp accepted frames spread over azimuth x elevation: round-robin over the cells, sharpest first in each cell."""
    ok = [(i, p) for i, p in ps.items() if p["ok"] and centre_visible(p["rvec"], p["tvec"], K, size)]
    vp = {i: viewpoint(p["rvec"], p["tvec"]) for i, p in ok}
    if not ok:
        return []
    els = np.array([vp[i][1] for i, _ in ok])
    edges = np.linspace(els.min(), els.max() + 1e-6, el_bins + 1)
    cells: dict = {}
    for i, p in ok:
        a, e, _ = vp[i]
        cells.setdefault((int(a // (360 / az_bins)), int(np.searchsorted(edges, e, side="right") - 1)), []).append((p["sharp"], i))
    for c in cells.values():
        c.sort(reverse=True)
    chosen = []
    while len(chosen) < n and any(cells.values()):
        for key in sorted(cells, key=lambda k: (-len(cells[k]), k)):
            if cells[key] and len(chosen) < n:
                chosen.append(cells[key].pop(0)[1])
    return sorted(chosen)


def coverage_grid(idx, ps, az_bins=12, el_edges=None):
    """Counts of frames per (elevation band, azimuth sector). Returns (grid, el_edges)."""
    vp = [viewpoint(ps[i]["rvec"], ps[i]["tvec"]) for i in idx]
    el = np.array([v[1] for v in vp])
    el_edges = el_edges if el_edges is not None else np.linspace(el.min(), el.max() + 1e-6, 5)
    g = np.zeros((len(el_edges) - 1, az_bins), int)
    for a, e, _ in vp:
        g[min(max(np.searchsorted(el_edges, e, side="right") - 1, 0), g.shape[0] - 1), int(a // (360 / az_bins))] += 1
    return g, el_edges


def extract(path: Path, idx: list[int], K, dist, outdir: Path):
    """Read the chosen frames sequentially, undistort with (K, dist) keeping K, save JPEG q95. Returns {idx: file}."""
    outdir.mkdir(parents=True, exist_ok=True)
    want, files = set(idx), {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    while want:
        if i in want:
            ok, fr = cap.read()
            if not ok:
                break
            f = outdir / f"f{i:05d}.jpg"
            cv2.imwrite(str(f), cv2.undistort(fr, K, dist, None, K), [cv2.IMWRITE_JPEG_QUALITY, 95])
            files[i] = f
            want.discard(i)
        elif not cap.grab():
            break
        i += 1
    return files
