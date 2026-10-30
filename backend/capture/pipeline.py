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


def _init_worker():
    cv2.setNumThreads(0)        # forked workers never use the OpenCV thread pool they inherit (using it deadlocked the carve, v2 stage 4)


def _pool(n):
    return mp.get_context("fork").Pool(n, initializer=_init_worker)


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


def masks_for(sel, sam_dir: Path, live_call=None, budget=0, log=print, concepts=None):
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
            for ci, concept in enumerate(concepts or C.SAM_CONCEPTS):
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
    return carve.carve_field(_G["views"], _G["X"][sl], _G["k"], 0.0, floors=_G["floors"], grow_mm=C.GROW_MM)


def k_for(n_views):
    """Robust k scaled with the number of views (learnings rule 1a)."""
    from .learnings import k_for_views
    return k_for_views(n_views)


def carve_joint(views, T, ref, procs=PROCS, k=None, vs=None):
    """carve.carve_adaptive with the fine pass split over forked workers (same field). k defaults to k_for(number of views), vs to the frozen voxel."""
    allv = [v.moved(T[n]) for n in views for v in views[n]]
    k = k_for(len(allv)) if k is None else k
    floors = [T[n] for n in views if n != ref]
    g, f, _ = carve.carve(allv, C.BOX_LO, C.BOX_HI, 1.0, k, floors=floors, grow_mm=C.GROW_MM)
    idx = np.argwhere(f > 0)
    lo = np.maximum(np.asarray(C.BOX_LO, float), g.lo + idx.min(0) * g.vs - 3.0)
    hi = np.minimum(np.asarray(C.BOX_HI, float), g.lo + idx.max(0) * g.vs + 3.0)
    g = carve.Grid(lo, hi, vs or C.VOXEL_MM)
    X = g.points()
    _G.update(views=allv, X=X, floors=floors, k=k)
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
        T_cached: dict | None = None, log=print, groove_depth_mm: float | None = None, concepts=None) -> dict:
    """videos: {name: path}; roles: {"ref": name, "flat": name, "stand": name}. Writes work/part.stl and work/report.json.
    groove_depth_mm: the slot depth (a user's answer); slots are cut only when it is given (the camera cannot see the depth)."""
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
        msk, info = masks_for(sel, sam_dir or (cache / "sam"), live_call, sam_budget, log, concepts)
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
        if ops and groove_depth_mm:
            field = F.cut_pockets(field, g, F.groove_boxes(ops, face, groove_depth_mm))
            v1, f1 = mesh_of(field, g)
        else:
            v1, f1 = v0, f0
    with tm("write STL"):
        carve.write_stl(work / "part.stl", v1, f1)
    pl, cen = carve.obb_plan(v1), carve.edge_census(v1, f1)
    rep.update({"T": {n: m.tolist() for n, m in T.items()}, "registration_from_cache": T_cached is not None,
                "footprint_mm": [pl["length_mm"], pl["width_mm"]], "height_mm": pl["z_max"] - pl["z_min"], "watertight": cen["watertight"], "volume_mm3": cen["volume_mm3"],
                "grooves": [{"centre": o["centre"].tolist(), "long_mm": o["L"], "short_mm": o["S"], "views": o["n_views"]} for o in ops],
                "groove_depth_mm": groove_depth_mm, "groove_depth_source": "user" if groove_depth_mm else "not given: slots not cut",
                "views": {n: len(v) for n, v in views.items()}, "seconds": {k: v for k, v in tm.rows}, "total_s": sum(v for _, v in tm.rows)})
    (work / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    run.views, run.T, run.mesh, run.unfeatured = views, T, (v1, f1), (v0, f0)
    return rep


# ---------------------------------------------------------------- browser path: ~40 frames per video at ~1920 px (v3 stage 3)

def detect_frames(frames_by_video):
    """{video: {idx: BGR}} -> scans {video: {"size", "frames": {idx: (ids, corners, sharpness)}}}."""
    det = mat.make_detector()
    out = {}
    for n, fr in frames_by_video.items():
        fd = {}
        for i, img in fr.items():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ids, cs = mat.detect(det, gray)
            fd[i] = (ids, cs, frames.marker_sharpness(gray, cs) if len(ids) else 0.0)
        img0 = next(iter(fr.values()))
        out[n] = {"size": (img0.shape[1], img0.shape[0]), "frames": fd}
    return out


def calibrate_frames(scans, min_markers=8):
    """Joint mat self-calibration from the frames' detections. Returns (world, {video: (K, dist)}, rms px)."""
    obs = {}
    for n, sc in scans.items():
        cal = frames.calibrate(sc, min_markers=min_markers)
        o = [(f[0], f[1]) for f in sc["frames"].values() if len(f[0]) >= min_markers]
        ps = []
        for ids, cs in o:
            rv, tv, _, _ = mat.solve_pose(ids, cs, cal["K"], cal["dist"])
            ps.append((rv.ravel(), tv.ravel()))
        obs[n] = dict(obs=o, poses=ps, K=cal["K"], dist=cal["dist"])
    r = matcal.calibrate_joint(obs, prior=C.MAT_PRIOR_PX_PER_MM)
    intr = {n: (np.array([[i[0], 0, i[2]], [0, i[1], i[3]], [0, 0, 1.0]]), np.r_[i[4:8], 0]) for n, i in r["intr"].items()}
    return matcal.world_dict(r["W"]), intr, r["rms_px"]


def crops_from_frames(frames_by_video, scans, world, intr):
    """Poses (rms <= the relaxed limit), undistorted crops around the part. {video: [(idx, box, crop BGR, rvec, tvec)]}, K per video."""
    out = {}
    for n, fr in frames_by_video.items():
        K, dist = intr[n]
        rows = []
        for i, img in fr.items():
            ids, cs, _ = scans[n]["frames"][i]
            if len(ids) < C.MIN_MARKERS:
                continue
            sol = mat.solve_pose(ids, cs, K, dist, world=world)
            if sol is None or sol[2] > C.MAX_RMS_RELAXED_PX:
                continue
            rv, tv = sol[0].ravel(), sol[1].ravel()
            box = M.crop_box(K, rv, tv, scans[n]["size"])
            und = cv2.undistort(img, K, dist, None, K)
            rows.append((i, box, und[box[1]:box[3], box[0]:box[2]].copy(), rv, tv))
        out[n] = rows
    return out


def masks_from_crops(crops, intr, sam, concepts=None):
    """sam(jobs) with jobs [(key, crop, concept, crop box in the frame)] -> {key: [(conf, crop-space mask)] or None}: main concept for every crop, fallbacks only for misses.
    Then pick + edge refine + accept (colour agreement, or area when the colour mask is unreliable). Returns ({video: [(idx, box, crop, mask, rvec, tvec)]}, calls)."""
    keys = {f"{n}_{i}": (n, i, crop, box) for n, rows in crops.items() for i, box, crop, _, _ in rows}
    got, calls, todo = {}, 0, list(keys)
    for ci, concept in enumerate(concepts or C.SAM_CONCEPTS):
        if not todo:
            break
        jobs = [(k if ci == 0 else f"{k}__{concept.replace(' ', '_')}", keys[k][2], concept, keys[k][3]) for k in todo]
        res = sam(jobs)
        calls += len(jobs)
        for (jk, _, _, _), k in zip(jobs, todo):
            if res.get(jk):
                got[k] = res[jk]
        todo = [k for k in todo if k not in got]
    # the part's colour, learned from SAM's best masks (None for an achromatic part: then no colour gating, area consistency only)
    part_hue = M.learn_colour([(keys[k][2], max(got[k], key=lambda cm: cm[0])[1]) for k in list(got)[:12]])
    masks_from_crops.part_hue = part_hue
    out = {}
    for n, rows in crops.items():
        K = intr[n][0]
        cand = []
        for i, box, crop, rv, tv in rows:
            r = got.get(f"{n}_{i}")
            if not r:
                continue
            if part_hue:
                col = M.colour_mask(crop, hue_min=part_hue[0] - 2, hue_max=part_hue[1])
                pick = M.pick_component(r, col)
                fin = M.refine_sam(pick[1], crop, part_hue=part_hue)
                cis = M.agreement(pick[1], col)["colour_in_sam"]
            else:
                col = np.zeros(crop.shape[:2], np.uint8)
                pick = max(r, key=lambda cm: cm[0])
                fin = (cv2.GaussianBlur((pick[1] > 0).astype(np.float32), (0, 0), 1.0) > 0.5).astype(np.uint8) * 255
                cis = 0.0
            dist = np.linalg.norm(-cv2.Rodrigues(rv)[0].T @ tv - frames.CENTRE)
            cand.append((i, box, crop, fin, rv, tv, cis, pick[0], (col > 0).sum(), (pick[1] > 0).sum(), (pick[1] > 0).sum() / (K[0, 0] / dist) ** 2))
        med = float(np.median([c[10] for c in cand])) if cand else 0
        out[n] = [(i, box, crop, fin, rv, tv) for i, box, crop, fin, rv, tv, cis, conf, ca, sa, amm in cand
                  if conf >= C.MIN_SAM_CONF and (cis >= C.MIN_COLOUR_IN_SAM or (ca > 1.5 * sa and 0.5 * med < amm < 1.5 * med))]
    return out, calls


def views_from_masks(msk, intr):
    views = {}
    for n, rows in msk.items():
        K0 = intr[n][0]
        vs = []
        for i, box, crop, fin, rv, tv in rows:
            K = K0.copy()
            K[0, 2] -= box[0]
            K[1, 2] -= box[1]
            vs.append(carve.View(K, cv2.Rodrigues(rv)[0], tv, carve.mask_sdf(fin), f"{n}_{i}", n))
        views[n] = vs
    return views


def run_frames(frames_by_video, roles, work: Path, sam, answers=None, log=print, concepts=None):
    """Browser path. frames_by_video: {video: {idx: BGR (~1920 px)}}; roles {"ref", "flat", "stand"}; sam: see masks_from_crops.
    Builds the editable model (carve field + measured features), applies the answers (frozen defaults where missing), writes work/part.stl,
    work/model.npz/.json and returns the run record (timings, features, measured values)."""
    from . import model as Mo
    work.mkdir(parents=True, exist_ok=True)
    tm = Timer(log)
    with tm("marker detection"):
        scans = detect_frames(frames_by_video)
    with tm("mat self-calibration"):
        world, intr, rms = calibrate_frames(scans)
    with tm("poses + undistorted crops"):
        crops = crops_from_frames(frames_by_video, scans, world, intr)
    with tm("SAM"):
        msk_calls = masks_from_crops(crops, intr, sam, concepts)
    msk, calls = msk_calls
    views = views_from_masks(msk, intr)
    with tm("registration"):
        T = register_fast(views, roles["ref"], roles["flat"], roles["stand"], log)
    with tm("joint carve"):
        allv, g, field = carve_joint(views, T, roles["ref"])
        v0, f0 = mesh_of(field, g)
    vv = [(n, v.moved(T[n])) for n in views for v in views[n]]
    imgs = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for n in msk for _, _, crop, _, _, _ in msk[n]]
    feats = {}
    with tm("features (grooves, bump, plate)"):
        face, ops, _, _ = find_grooves_parallel(v0, f0, vv, imgs)
        if ops:
            feats["grooves"] = F.groove_features(ops, face)
            dl = float(np.mean([o["dist_from_face_mm"] for o in ops]))
            b = F.detect_bump(vv, imgs, face, dl)
            if b:
                feats["bump"] = F.bump_feature(b, feats["grooves"])
                region = F.plate_region(field, g, b["ends"])
                pf = F.plate_faces(vv, imgs, region, {"lower": roles["flat"], "upper": roles["ref"]})
                if "upper" in pf:
                    X = g.lo[0] + np.arange(field.shape[0]) * g.vs
                    Y = g.lo[1] + np.arange(field.shape[1]) * g.vs
                    mask = np.zeros(field.shape[:2], bool)
                    mask[np.ix_((X >= region[0]) & (X <= region[1]), (Y >= region[2]) & (Y <= region[3]))] = True
                    sep = pf["upper"]["z"] - pf["lower"]["z"] if "lower" in pf else None
                    feats["plate"] = {"normal": [0, 0, 1.0], "d": -pf["upper"]["z"], "mask": mask, "region_xy": list(region), "faces": pf, "measured_separation_mm": sep}
    with tm("apply answers + mesh + STL"):
        a = Mo.resolve_answers(answers, feats, Mo.part_height(field, g))
        if feats.get("plate") and feats["plate"]["measured_separation_mm"] is not None and abs(feats["plate"]["measured_separation_mm"] - a["plate_thickness_mm"]) <= 0.3:
            a["plate_thickness_mm"] = feats["plate"]["measured_separation_mm"]
        v1, f1 = Mo.mesh(Mo.apply(field, g, feats, a), g)
        carve.write_stl(work / "part.stl", v1, f1)
        Mo.save(work / "model", field, g, feats)
    run_frames.views, run_frames.T, run_frames.mesh, run_frames.field = views, T, (v1, f1), (field, g)
    return {"seconds": dict(tm.rows), "total_s": sum(s for _, s in tm.rows), "sam_calls": calls, "calib_rms_px": rms, "views": {n: len(v) for n, v in views.items()},
            "features": feats, "answers": a, "T": T}
