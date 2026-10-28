"""Production path: three orbit videos of one part on the ArUco mat -> calibrated, carved, featured STL and a report.

Videos (by role): `ref` defines the part frame, `flat` is a second flat orbit, `stand` shows the part on its edge.
Stages (wall clock printed for each): scan markers, mat self-calibration, poses + frame selection + extraction, crops + SAM masks,
registration (coarse-to-fine), joint carve, groove detection + pockets, write STL. Caches: pass `cache` to reuse scans, the mat and SAM
results; SAM calls are made only on cache misses (and only when `sam_live` is set).
"""
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import carve, config as C, features as F, frames, masks as M, mat, matcal, register as Rg

cv2.setNumThreads(0)        # no OpenCV thread pool: worker processes are forked, and a fork can inherit a held pool lock (a carve deadlock was seen)


class Timer:
    def __init__(self, log):
        self.log, self.t, self.rows = log, {}, []

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()
                timer.log(f"{time.strftime('%H:%M:%S')} [{name}] start")

            def __exit__(self, *a):
                dt = time.perf_counter() - self.t0
                timer.rows.append((name, dt))
                timer.log(f"{time.strftime('%H:%M:%S')} [{name}] {dt:.1f} s")
        return _T()


PROCS = min(8, os.cpu_count() or 1)
_G: dict = {}                                    # large inputs shared with forked workers (not pickled)


def _pool(n):
    return mp.get_context("fork").Pool(n)


def _scan_one(args):
    p, c = args
    return frames.scan(p, stride=C.SCAN_STRIDE, cache=c, log=lambda m: None)


def scan_videos(videos: dict, cache: Path | None, log=print):
    """One process per video (decoding and marker detection dominate)."""
    jobs = [(p, (cache / f"scan_{n}.pkl") if cache else None) for n, p in videos.items()]
    with _pool(len(jobs)) as pool:
        return dict(zip(videos, pool.map(_scan_one, jobs)))


def calibrate_mat(scans: dict, cache: Path | None):
    """Joint mat self-calibration (marker corners free in 3D, per-video intrinsics). Returns (world corners dict, {video: (K, dist)})."""
    f = cache / "mat_calibrated.json" if cache else None
    if f and f.exists():
        W, intr = matcal.load(f)[:2]
    else:
        obs = {}
        for n, sc in scans.items():
            cal = frames.calibrate(sc)
            cand = [(i, fr) for i, fr in sc["frames"].items() if len(fr[0]) >= 14]
            pick = [cand[j] for j in np.linspace(0, len(cand) - 1, C.CALIB_FRAMES_PER_VIDEO * 2).astype(int)][0::2]
            o = [(fr[0], fr[1]) for _, fr in pick]
            ps = []
            for ids, cs in o:
                rv, tv, _, _ = mat.solve_pose(ids, cs, cal["K"], cal["dist"])
                ps.append((rv.ravel(), tv.ravel()))
            obs[n] = dict(obs=o, poses=ps, K=cal["K"], dist=cal["dist"])
        r = matcal.calibrate_joint(obs, prior=C.MAT_PRIOR_PX_PER_MM)
        W, intr = r["W"], r["intr"]
        if f:
            matcal.save(f, W, intr, {"rms_px": r["rms_px"]})
    out = {}
    for n, i in intr.items():
        out[n] = (np.array([[i[0], 0, i[2]], [0, i[1], i[3]], [0, 0, 1.0]]), np.r_[i[4:8], 0])
    return matcal.world_dict(W), out


def select_frames(videos, scans, world, intr, frame_dir: Path):
    """Poses with the calibrated mat, acceptance (relaxed only if too few pass), spread selection, undistorted extraction."""
    sel = {}
    for n, sc in scans.items():
        K, dist = intr[n]
        ps = frames.poses(sc, K, dist, max_rms=C.MAX_RMS_PX, world=world)
        thr = C.MAX_RMS_PX
        while frames.reaccept(ps, thr) < C.MIN_ACCEPTED and thr < C.MAX_RMS_RELAXED_PX:
            thr += 0.25
        idx = frames.select(ps, K, sc["size"], n=C.FRAMES_PER_VIDEO)
        sel[n] = {"K": K, "dist": dist, "size": sc["size"], "rms_threshold": thr, "idx": idx, "ps": ps}
    have = {n: {i: frame_dir / n / f"f{i:05d}.jpg" for i in s["idx"]} for n, s in sel.items()}
    todo = [n for n in sel if not all(f.exists() for f in have[n].values())]
    with _pool(max(len(todo), 1)) as pool:                          # extraction: one process per video, skipped when the frames exist
        got = dict(zip(todo, pool.starmap(frames.extract, [(videos[n], sel[n]["idx"], sel[n]["K"], sel[n]["dist"], frame_dir / n) for n in todo])))
    files = [got.get(n, have[n]) for n in sel]
    for (n, s), fl in zip(sel.items(), files):
        ps = s.pop("ps")
        s["frames"] = {i: {"file": fl[i], "rvec": ps[i]["rvec"], "tvec": ps[i]["tvec"]} for i in s.pop("idx") if i in fl}
    return sel


def _cached_sam(sam_dir: Path, key: str):
    meta = sam_dir / f"{key}.json"
    if not meta.exists():
        return None
    return [(c, cv2.imread(str(sam_dir / f"{key}_{i}.png"), 0)) for i, c in enumerate(json.loads(meta.read_text())["confs"])]


def masks_for(sel, sam_dir: Path, live_call=None, budget=0, log=print):
    """Crop around the part, SAM per frame (cache '<video>_<frame>'; fallback concepts '<key>__<concept>'), pick + edge refine, accept.
    live_call(key, crop_path, concept) -> [(conf, mask)] is used only on cache misses, at most `budget` times."""
    out, info = {}, {"calls": 0, "misses": 0, "frames": 0, "accepted": 0}
    for n, s in sel.items():
        K = s["K"]
        rows = []
        for i, fr in s["frames"].items():
            img = cv2.imread(str(fr["file"]))
            box = M.crop_box(K, fr["rvec"], fr["tvec"], s["size"])
            crop = img[box[1]:box[3], box[0]:box[2]]
            col = M.colour_mask(crop)
            key, res = f"{n}_{i}", None
            for ci, concept in enumerate(C.SAM_CONCEPTS):
                k2 = key if ci == 0 else f"{key}__{concept.replace(' ', '_')}"
                res = _cached_sam(sam_dir, k2)
                if res is None and live_call and info["calls"] < budget:
                    info["misses"] += 1
                    info["calls"] += 1
                    res = live_call(k2, crop, concept)
                if res:
                    break
            info["frames"] += 1
            pick = M.pick_component(res, col) if res else None
            if pick is None:
                continue
            fin = M.refine_sam(pick[1], crop)
            a = M.agreement(pick[1], col)
            area_mm2 = (pick[1] > 0).sum() / (K[0, 0] / np.linalg.norm(-cv2.Rodrigues(np.asarray(fr["rvec"], float))[0].T @ np.asarray(fr["tvec"], float) - frames.CENTRE)) ** 2
            rows.append((i, box, crop, fin, a["colour_in_sam"], pick[0], (col > 0).sum(), (pick[1] > 0).sum(), area_mm2))
        med = float(np.median([r[8] for r in rows])) if rows else 0
        keep = []
        for i, box, crop, fin, cis, conf, carea, sarea, amm in rows:
            ok = cis >= C.MIN_COLOUR_IN_SAM and conf >= C.MIN_SAM_CONF
            ok = ok or (carea > 1.5 * sarea and 0.5 * med < amm < 1.5 * med and conf >= C.MIN_SAM_CONF)     # colour mask unreliable: judge by area
            if ok:
                keep.append((i, box, crop, fin))
        out[n] = keep
        info["accepted"] += len(keep)
    return out, info


def build_views(sel, msk):
    """Crop-space views (mat frame of each video)."""
    views = {}
    for n, rows in msk.items():
        vs = []
        for i, box, crop, fin in rows:
            fr = sel[n]["frames"][i]
            K = np.array(sel[n]["K"], float).copy()
            K[0, 2] -= box[0]
            K[1, 2] -= box[1]
            v = carve.View(K, cv2.Rodrigues(np.asarray(fr["rvec"], float))[0], np.asarray(fr["tvec"], float), carve.mask_sdf(fin), f"{n}_{i}", n)
            vs.append(v)
        views[n] = vs
    return views


def _hull_points(views, n=3000):
    g, f, _ = carve.carve_adaptive(views, C.BOX_LO, C.BOX_HI, 0.5, 2)
    v, fc = carve.to_mesh(f, g)
    return carve.sample_surface(v, fc, n)


def register_fast(views, ref, flat, stand, log=print, refine_vs=C.REG_REFINE_VOXEL_MM):
    """Part frame = ref's mat frame. flat -> ref, then stand -> joint hull of ref + flat (flat refined again with it).
    Coarse-to-fine ICP from 24 x 6 starts, best fine solution, then one Powell round on the joint volume (k=2, refine_vs voxels)."""
    T = {ref: np.eye(4)}
    g, f, _ = carve.carve_adaptive(views[ref], C.BOX_LO, C.BOX_HI, refine_vs, 2)
    X = g.points()[f.ravel() > -0.7]
    centre = X.mean(0)
    dst = _hull_points(views[ref])
    c, R, t = Rg.candidates_fast(_hull_points(views[flat]), dst)[0]
    T[flat] = np.linalg.inv(Rg.hom(R, t))
    T, _ = Rg.refine({n: views[n] for n in (ref, flat)}, T, [flat], X, centre, 2, refine_vs, log=lambda m: None, maxiter=1)
    allv = list(views[ref]) + [v.moved(T[flat]) for v in views[flat]]
    g13, f13, _ = carve.carve_adaptive(allv, C.BOX_LO, C.BOX_HI, 0.5, 2, floors=[T[flat]])
    v13, fc13 = carve.to_mesh(f13, g13)
    c2, R2, t2 = Rg.candidates_fast(_hull_points(views[stand]), carve.sample_surface(v13, fc13, 3000))[0]
    T[stand] = np.linalg.inv(Rg.hom(R2, t2))
    T, _ = Rg.refine(views, T, [stand, flat], X, centre, 2, refine_vs, log=lambda m: None, maxiter=1)
    return T


def transform_gap(T1, T2, pts):
    """Largest displacement (mm) of the points under T1 vs T2 (part frame -> video frame)."""
    a = pts @ T1[:3, :3].T + T1[:3, 3]
    b = pts @ T2[:3, :3].T + T2[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).max())


def _field_chunk(sl):
    return carve.carve_field(_G["views"], _G["X"][sl], C.K_ROBUST, 0.0, floors=_G["floors"], grow_mm=C.GROW_MM)


def carve_joint(views, T, ref, procs=PROCS):
    """carve.carve_adaptive with the fine pass split over forked workers (same field)."""
    allv = [v.moved(T[n]) for n in views for v in views[n]]
    floors = [T[n] for n in views if n != ref]
    g, f, _ = carve.carve(allv, C.BOX_LO, C.BOX_HI, 1.0, C.K_ROBUST, floors=floors, grow_mm=C.GROW_MM)
    idx = np.argwhere(f > 0)
    lo = np.maximum(np.asarray(C.BOX_LO, float), g.lo + idx.min(0) * g.vs - 3.0)
    hi = np.minimum(np.asarray(C.BOX_HI, float), g.lo + idx.max(0) * g.vs + 3.0)
    g = carve.Grid(lo, hi, C.VOXEL_MM)
    X = g.points()
    _G.update(views=allv, X=X, floors=floors)
    step = -(-len(X) // (procs * 4))
    with _pool(procs) as pool:
        parts = pool.map(_field_chunk, [slice(s, s + step) for s in range(0, len(X), step)])
    _G.clear()
    return allv, g, np.concatenate(parts).reshape(g.shape)


def _detect_face(face):
    return F.detect_on_face(_G["vv"], _G["imgs"], face)


def find_grooves_parallel(v, fc, vv, imgs, procs=PROCS):
    faces = F.candidate_faces(v, fc)
    _G.update(vv=vv, imgs=imgs)
    with _pool(min(procs, len(faces))) as pool:
        pers = pool.map(_detect_face, faces)
    _G.clear()
    return F.find_grooves(v, fc, vv, imgs, faces=faces, pers=pers)


def mesh_of(field, g):
    v, fc = carve.to_mesh(field, g)
    return carve.taubin(v, fc, C.TAUBIN_ITERS), fc


def run(videos: dict, roles: dict, work: Path, cache: Path | None = None, sam_dir: Path | None = None, live_call=None, sam_budget=0,
        T_cached: dict | None = None, log=print) -> dict:
    """videos: {name: path}; roles: {"ref": name, "flat": name, "stand": name}. Writes work/part.stl and work/report.json."""
    work.mkdir(parents=True, exist_ok=True)
    tm = Timer(log)
    rep = {}
    with tm("scan markers"):
        scans = scan_videos(videos, cache, log)
    with tm("mat self-calibration"):
        world, intr = calibrate_mat(scans, cache)
    with tm("poses, selection, frame extraction"):
        sel = select_frames(videos, scans, world, intr, work / "frames")
    with tm("crops + SAM (cached) + masks"):
        msk, info = masks_for(sel, sam_dir or (cache / "sam"), live_call, sam_budget, log)
    rep["masks"] = info
    views = build_views(sel, msk)
    with tm("registration"):
        T = T_cached if T_cached is not None else register_fast(views, roles["ref"], roles["flat"], roles["stand"], log)
    with tm("joint carve"):
        allv, g, field = carve_joint(views, T, roles["ref"])
        v0, f0 = mesh_of(field, g)
    vv = [(n, v.moved(T[n])) for n in views for v in views[n]]
    imgs = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for n in msk for _, _, crop, _ in msk[n]]
    with tm("grooves + pockets"):
        face, ops, per, _ = find_grooves_parallel(v0, f0, vv, imgs)
        if ops:
            field = F.cut_pockets(field, g, F.groove_boxes(ops, face, C.GROOVE_DEPTH_USER_MM))
            v1, f1 = mesh_of(field, g)
        else:
            v1, f1 = v0, f0
    with tm("write STL"):
        carve.write_stl(work / "part.stl", v1, f1)
    pl, cen = carve.obb_plan(v1), carve.edge_census(v1, f1)
    rep.update({"T": {n: m.tolist() for n, m in T.items()}, "registration_from_cache": T_cached is not None,
                "footprint_mm": [pl["length_mm"], pl["width_mm"]], "height_mm": pl["z_max"] - pl["z_min"], "watertight": cen["watertight"], "volume_mm3": cen["volume_mm3"],
                "grooves": [{"centre": o["centre"].tolist(), "long_mm": o["L"], "short_mm": o["S"], "views": o["n_views"]} for o in ops],
                "groove_depth_mm": C.GROOVE_DEPTH_USER_MM, "groove_depth_source": "user (caliper)",
                "views": {n: len(v) for n, v in views.items()}, "seconds": {k: v for k, v in tm.rows}, "total_s": sum(v for _, v in tm.rows)})
    (work / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    run.views, run.T, run.mesh, run.unfeatured = views, T, (v1, f1), (v0, f0)
    return rep
