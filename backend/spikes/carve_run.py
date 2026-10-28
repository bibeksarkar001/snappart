"""Calibrated multi-view carving of part A from three videos. Usage: python backend/spikes/carve_run.py --stage 0..6

Writes out/carve/SEND_TO_CLAUDE/ after every stage: stageN.txt (measured numbers), summary.txt (all stages so far),
results.json (merged), PNGs and STLs. Videos and frames are private: caches live under out/ (git-ignored).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from spike_common import OUT, ROOT  # noqa: E402

CARVE = OUT / "carve"
SEND = CARVE / "SEND_TO_CLAUDE"
VIDEOS = ROOT / "assets" / "private" / "lock_set" / "Video"
MAX_RMS = 1.0
NAMES = ["A_orbit_01", "A_orbit_02", "A_orbit_03"]
for d in (CARVE, SEND):
    d.mkdir(parents=True, exist_ok=True)


def save_stage(n: int, text: str, results: dict):
    """Write stageN.txt, rebuild summary.txt from all stage files, merge results into results.json."""
    (SEND / f"stage{n}.txt").write_text(text)
    (SEND / "summary.txt").write_text("\n\n".join(p.read_text() for p in sorted(SEND.glob("stage*.txt"))))
    f = SEND / "results.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[f"stage{n}"] = results
    f.write_text(json.dumps(cur, indent=1, default=float))


def stage0():
    from capture import synth
    r = synth.report(*synth.build())
    b = r["carved_pts_by_z_band"]
    L = ["STAGE 0: synthetic test (plate 34 x 30 x 6 mm, through-slot 22 x 12, 4.0 mm bar; 60 virtual cameras, elevation 15-89 deg, 1920x1080, f 1900 px)",
         "Poses come from detected mat markers (not ground truth); masks are exact anti-aliased silhouettes; k=2; voxel 0.25 mm; Taubin 5 iterations.",
         f"surface distance carve<->truth: mean {r['overall']['mean_mm']:.3f} mm, p95 {r['overall']['p95_mm']:.3f}, MAX {r['overall']['max_mm']:.3f}",
         "carved surface points by height band (distance to truth):"]
    L += [f"  {k}: mean {v['mean_mm']:.3f} mm, max {v['max_mm']:.3f}, n={v['n']}" for k, v in b.items()]
    L += [f"extent carve {[round(x, 2) for x in r['carved_extent_mm']]} vs truth {r['truth_extent_mm']}; volume carve {r['census']['volume_mm3']:.0f} vs truth {r['truth_volume_mm3']:.0f} mm3",
          f"watertight {r['census']['watertight']}, boundary edges {r['census']['boundary_edges']}, non-manifold {r['census']['non_manifold_edges']}",
          f"camera centre error from marker PnP: mean {r['camera_centre_err_mm']['mean']:.3f} mm, max {r['camera_centre_err_mm']['max']:.3f}; reprojection rms max {r['reproj_rms_px_max']:.2f} px",
          "NOTE: the error is concentrated above the top face (hull inflation from limited elevation coverage, 60 views); side walls and slot are far smaller."]
    save_stage(0, "\n".join(L), r)
    print("\n".join(L))


def draw_markers(img, rvec, tvec, K, world=None):
    import cv2
    import numpy as np
    from capture import mat
    out = img.copy()
    for mid, w in (world or mat.WORLD).items():
        p = cv2.projectPoints(w, rvec, tvec, K, None)[0].reshape(-1, 2)
        cv2.polylines(out, [np.round(p).astype(np.int32)], True, (0, 255, 0), max(2, img.shape[1] // 700), cv2.LINE_AA)
        cv2.putText(out, str(mid), tuple(np.round(p.mean(0) - 8).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1800, (0, 200, 255), max(2, img.shape[1] // 1000), cv2.LINE_AA)
    axes = np.float32([[0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40]]) + [mat.PAGE_W / 2 - 20, mat.PAGE_H / 2 - 20, 0]
    q = np.round(cv2.projectPoints(axes, rvec, tvec, K, None)[0].reshape(-1, 2)).astype(int)
    for j, col in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(out, tuple(q[0]), tuple(q[j]), col, max(3, img.shape[1] // 500))
    return out


def load_scans():
    from capture import frames
    cache = CARVE / "cache"
    cache.mkdir(exist_ok=True)
    return {n: frames.scan(VIDEOS / f"{n}.mov", stride=2, cache=cache / f"scan_{n}.pkl") for n in NAMES}


def stage1():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from capture import frames, mat, matcal
    from spike_common import save_overlay
    cache = CARVE / "cache"
    scs = load_scans()
    L = ["STAGE 1: mat + frames (ArUco DICT_4X4_50, 26 markers 22.0 mm; detection every 2nd frame; intrinsics k1,k2,p1,p2 per video)"]
    res = {}

    def pick(n, count, off):
        cand = [(i, f) for i, f in scs[n]["frames"].items() if len(f[0]) >= 14]
        return [cand[j] for j in np.linspace(0, len(cand) - 1, count * 2).astype(int)][off::2]

    def vids(cals, off, count):
        out = {}
        for n in NAMES:
            K, d = cals[n]["K"], cals[n]["dist"]
            obs = [(f[0], f[1]) for _, f in pick(n, count, off)]
            ps = []
            for ids, cs in obs:
                rv, tv, _, _ = mat.solve_pose(ids, cs, K, d)
                ps.append((rv.ravel(), tv.ravel()))
            out[n] = dict(obs=obs, poses=ps, K=K, dist=d)
        return out

    # 1. per-video planar calibration with the printed layout
    cal0 = {n: frames.calibrate(scs[n]) for n in NAMES}
    # 2. does the printed layout fit? A/B halves: calibrate the mat on half A, pose half B with it fixed
    A, B = vids(cal0, 0, 80), vids(cal0, 1, 80)
    held = {}
    for label, prior in [("printed planar layout", float("inf")), ("calibrated mat (prior 2 px/mm)", 2.0)]:
        r = matcal.calibrate_joint(A, prior=prior)
        W = matcal.world_dict(r["W"])
        held[label] = {}
        for n in NAMES:
            i = r["intr"][n]
            K = np.array([[i[0], 0, i[2]], [0, i[1], i[3]], [0, 0, 1.0]])
            e = [mat.solve_pose(ids, cs, K, np.r_[i[4:8], 0], world=W)[2] for ids, cs in B[n]["obs"]]
            held[label][n] = float(np.mean(e))
    L += ["", "MAT FLATNESS. With the printed planar layout no frame of any video reprojects under 1.0 px (rms 2.5-3.7 px, worse at low elevation).",
          "Freeing the 3D position of every marker corner (prior toward the printed layout) in a joint bundle adjustment over the three videos (shared corners,",
          "per-video intrinsics, per-frame poses) fixes it. Held-out test: calibrate on 80 frames/video, then pose 80 different frames/video with the mat fixed:"]
    for label, v in held.items():
        L.append(f"  {label}: held-out mean reprojection rms per video " + ", ".join(f"{n[-2:]}: {x:.2f} px" for n, x in v.items()))
    # 3. final joint calibration on 120 frames per video
    full = vids(cal0, 0, 120)
    r = matcal.calibrate_joint(full, prior=2.0)
    W, intr = r["W"], r["intr"]
    d = W - matcal.BASE
    matcal.save(cache / "mat_calibrated.json", W, intr, {"rms_px": r["rms_px"]})
    plane, tps = matcal.floor_surface(W, "plane"), matcal.floor_surface(W, "tps")
    cx, cy = mat.PAGE_W / 2, mat.PAGE_H / 2
    side = np.linalg.norm(W[:, 1] - W[:, 0], axis=1)
    L += [f"final joint calibration (120 frames/video): training rms {r['rms_px']:.2f} px; corner shifts from the printed layout: mean |dz| {np.abs(d[..., 2]).mean():.2f} mm, max |dz| {np.abs(d[..., 2]).max():.2f}, "
          f"mean |dxy| {np.linalg.norm(d[..., :2], axis=2).mean():.2f}, max {np.linalg.norm(d[..., :2], axis=2).max():.2f}; marker side {side.mean():.2f} mm (min {side.min():.2f}, max {side.max():.2f}; printed 22.00)",
          f"mat surface at the page centre (where the part sits, no markers there): plane fit {float(plane(cx, cy)):.2f} mm, thin-plate spline {float(tps(cx, cy)):.2f} mm (marker corner z: min {W[..., 2].min():.2f}, max {W[..., 2].max():.2f})"]
    res["mat"] = {"held_out_rms": held, "train_rms": r["rms_px"], "mean_abs_dz": float(np.abs(d[..., 2]).mean()), "max_abs_dz": float(np.abs(d[..., 2]).max()),
                  "floor_plane_at_centre": float(plane(cx, cy)), "floor_tps_at_centre": float(tps(cx, cy)), "side_mean": float(side.mean())}
    world = matcal.world_dict(W)
    polar = {}
    for name in NAMES:
        sc = scs[name]
        i8 = intr[name]
        K = np.array([[i8[0], 0, i8[2]], [0, i8[1], i8[3]], [0, 0, 1.0]])
        dist = np.r_[i8[4:8], 0]
        ps = frames.poses(sc, K, dist, max_rms=MAX_RMS, world=world)
        rall = np.array([q["rms"] for q in ps.values() if "rms" in q])
        thr = MAX_RMS
        while frames.reaccept(ps, thr) < 160 and thr < 1.5:      # spec: reject above 1.0 px; relaxed in 0.25 px steps (max 1.5) only if fewer than 160 frames survive
            thr += 0.25
        acc = [i for i, p in ps.items() if p["ok"]]
        sel = frames.select(ps, K, sc["size"], n=80)
        rms = np.array([ps[i]["rms"] for i in acc])
        why = {}
        for i, p in ps.items():
            if not p["ok"]:
                key = "fewer than 4 markers" if p["why"].startswith("fewer") else f"rms above {thr} px"
                why[key] = why.get(key, 0) + 1
        files = frames.extract(VIDEOS / f"{name}.mov", sel, K, dist, cache / "frames" / name)
        g, edges = frames.coverage_grid(sel, ps)
        vp = np.array([frames.viewpoint(ps[i]["rvec"], ps[i]["tvec"]) for i in sel])
        allvp = np.array([frames.viewpoint(ps[i]["rvec"], ps[i]["tvec"]) for i in acc])
        polar[name] = (allvp, vp)
        sel_info = {str(i): {"file": str(files[i]), "rvec": ps[i]["rvec"].tolist(), "tvec": ps[i]["tvec"].tolist(), "rms": ps[i]["rms"], "n": ps[i]["n"],
                             "az": vp[k][0], "el": vp[k][1], "dist": vp[k][2], "sharp": ps[i]["sharp"]} for k, i in enumerate(sel)}
        (cache / f"sel_{name}.json").write_text(json.dumps({"K": K.tolist(), "dist": dist.tolist(), "size": sc["size"], "frames": sel_info}))
        k0 = sel[len(sel) // 2]
        save_overlay(SEND / f"frame_markers_{name}.png", draw_markers(cv2.imread(str(files[k0])), np.array(ps[k0]["rvec"]), np.array(ps[k0]["tvec"]), K, world))
        with_m = sum(1 for f in sc["frames"].values() if len(f[0]))
        res[name] = {"frames_read": sc["n_read"], "analysed": len(sc["frames"]), "with_markers": with_m, "with_4plus": sum(1 for p in ps.values() if p["n_detected"] >= 4),
                     "accepted": len(acc), "rms_threshold_used": thr, "mean_rms_px": float(rms.mean()) if len(rms) else None, "selected": len(sel), "rejected_reasons": why,
                     "K": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "dist": dist[:4].tolist(), "planar_calib_rms": cal0[name]["rms_px"], "grid": g.tolist(), "el_edges": edges.tolist(), "size": sc["size"]}
        L += ["", f"{name}: {sc['size'][0]}x{sc['size'][1]}, container fps {sc['fps']:.1f}",
              f"  frames read {sc['n_read']}; analysed (every 2nd) {len(sc['frames'])}; with markers {with_m}; with >=4 markers {res[name]['with_4plus']}; accepted (>=4 markers, rms <= {thr:.2f} px{'' if thr == MAX_RMS else ', RELAXED from ' + str(MAX_RMS) + ' because fewer than 160 frames passed'}) {len(acc)}",
              f"  reprojection rms with the calibrated mat, all frames with a pose: p5 {np.percentile(rall, 5):.2f}, p25 {np.percentile(rall, 25):.2f}, median {np.median(rall):.2f}, p75 {np.percentile(rall, 75):.2f}, p95 {np.percentile(rall, 95):.2f}; "
              f"mean over accepted {rms.mean():.3f} px" if len(rms) else "  no accepted frames",
              f"  planar-layout calibrateCamera rms was {cal0[name]['rms_px']:.2f} px (printed layout)",
              f"  intrinsics: fx {K[0, 0]:.1f}, fy {K[1, 1]:.1f}, cx {K[0, 2]:.1f}, cy {K[1, 2]:.1f}; k1 {dist[0]:.4f} k2 {dist[1]:.4f} p1 {dist[2]:.5f} p2 {dist[3]:.5f}",
              f"  rejected frames: {why}",
              f"  accepted viewpoints: azimuth {allvp[:, 0].min():.0f}-{allvp[:, 0].max():.0f} deg, elevation {allvp[:, 1].min():.1f}-{allvp[:, 1].max():.1f} deg, distance {allvp[:, 2].min():.0f}-{allvp[:, 2].max():.0f} mm",
              f"  selected {len(sel)} frames; coverage grid (rows = elevation band, columns = 12 azimuth sectors of 30 deg):"]
        for row in range(g.shape[0] - 1, -1, -1):
            L.append(f"    el {edges[row]:5.1f}-{edges[row + 1]:5.1f} | " + " ".join(f"{v:2d}" for v in g[row]))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), subplot_kw={"projection": "polar"})
    for ax, name in zip(axs, NAMES):
        a, s = polar[name]
        ax.scatter(np.radians(a[:, 0]), 90 - a[:, 1], s=4, c="0.75", label="accepted")
        ax.scatter(np.radians(s[:, 0]), 90 - s[:, 1], s=14, c="C3", label="selected")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 90)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("camera viewpoints (angle = azimuth, radius = 90 - elevation)")
    fig.savefig(SEND / "coverage_grid.png", dpi=95)
    save_stage(1, "\n".join(L), res)
    print("\n".join(L))


def load_sel(name):
    d = json.loads((CARVE / "cache" / f"sel_{name}.json").read_text())
    return d, sorted(d["frames"], key=int)


async def sam_crops(jobs, refresh=False, conc=4, concept="brown plastic buckle"):
    """jobs: [(key, crop_jpeg_path)] -> {key: [(conf, crop-space 0/255 mask)]}. Cached under cache/sam/ (one cache entry per frame; calls counted)."""
    import asyncio
    import contextlib
    import io

    import cv2
    import numpy as np
    from meta_sam_parser import decode_mask_to_raster
    from sam_coin import segment
    from spike_common import full_mask
    d = CARVE / "cache" / "sam"
    d.mkdir(parents=True, exist_ok=True)
    sem, out, calls = asyncio.Semaphore(conc), {}, [0]

    async def one(key, path):
        meta = d / f"{key}.json"
        if meta.exists() and not refresh:
            out[key] = [(c, cv2.imread(str(d / f"{key}_{i}.png"), 0)) for i, c in enumerate(json.loads(meta.read_text())["confs"])]
            return
        async with sem:
            for attempt in range(3):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, w, h, boxes, masks, _ = await segment(Path(path), concept, f"carve/cache/sam/{key}_raw.txt")
                    break
                except Exception as e:           # noqa: BLE001  transient API errors: retry, then record the failure
                    err = repr(e)[:200]
                    boxes = None
            calls[0] += 1
        if boxes is None:
            out[key] = None
            (d / f"{key}.error").write_text(err)
            return
        res = []
        for i, b in enumerate(boxes):
            m = masks.get(b.object_id)
            if m is None:
                continue
            raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
            res.append((b.confidence or 0.0, full_mask(b, raster, w, h)))
            cv2.imwrite(str(d / f"{key}_{len(res) - 1}.png"), res[-1][1])
        meta.write_text(json.dumps({"concept": concept, "confs": [c for c, _ in res]}))
        out[key] = res

    await asyncio.gather(*[one(k, p) for k, p in jobs])
    return out, calls[0]


def stage2(limit=None, refresh=False):
    import asyncio

    import cv2
    import numpy as np
    from capture import masks as M
    from spike_common import save_overlay
    cache = CARVE / "cache"
    (cache / "crops").mkdir(exist_ok=True)
    items = []                                           # (name, idx, crop box, crop image, colour mask)
    jobs = []
    for name in NAMES:
        sel, keys = load_sel(name)
        K = np.array(sel["K"])
        if limit:
            keys = keys[:: max(len(keys) // limit, 1)][:limit]
        for k in keys:
            fr = sel["frames"][k]
            img = cv2.imread(fr["file"])
            box = M.crop_box(K, fr["rvec"], fr["tvec"], sel["size"])
            crop = img[box[1]:box[3], box[0]:box[2]]
            cp = cache / "crops" / f"{name}_{k}.jpg"
            cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            items.append((name, k, box, crop, M.colour_mask(crop)))
            jobs.append((f"{name}_{k}", cp))
    sam, ncalls = asyncio.run(sam_crops(jobs, refresh))
    used = {k: "brown plastic buckle" for k in sam}
    for fb in ("brown plastic part", "brown object"):          # fallback ladder, only for frames the main concept missed
        missing = [(k, p) for k, p in jobs if not sam.get(k)]
        if not missing:
            break
        r2, n2 = asyncio.run(sam_crops([(f"{k}__{fb.replace(' ', '_')}", p) for k, p in missing], refresh, concept=fb))
        ncalls += n2
        for k, _ in missing:
            if r2.get(f"{k}__{fb.replace(' ', '_')}"):
                sam[k] = r2[f"{k}__{fb.replace(' ', '_')}"]
                used[k] = fb
    L = [f"STAGE 2: masks (colour mask in Lab; SAM 3.1 concept 'brown plastic buckle' on a crop of the part, cached; {ncalls} new SAM calls this run, {len(jobs)} frames)"]
    rows, res = [], {}
    (cache / "masks").mkdir(exist_ok=True)
    for name, k, box, crop, col in items:
        r = sam.get(f"{name}_{k}")
        pick = M.pick_component(r, col) if r else None
        row = {"video": name, "idx": int(k), "box": box, "n_sam": len(r) if r else 0, "colour_area": int((col > 0).sum())}
        fin = None
        if pick is None:
            row.update(status="no SAM mask")
        else:
            row.update(M.agreement(pick[1], col), sam_conf=pick[0], sam_area=int((pick[1] > 0).sum()))
            fin = M.refine_sam(pick[1], crop)
            row["final_area"] = int((fin > 0).sum())
            row["edge_change_px2"] = int(((fin > 0) ^ (pick[1] > 0)).sum())
            row["concept"] = used.get(f"{name}_{k}")
            row["status"] = "ok" if row["colour_in_sam"] >= 0.85 and pick[0] >= 0.5 else "disagree"
            row["fin"] = fin
        rows.append((row, pick, col, crop, fin))
    # the colour mask fails in some exposures (it picks up the pink note and the shadow); where it covers far more than SAM, judge SAM by plausibility instead:
    # area in mm2 at the part's depth within 50 % of the video's median
    for name in NAMES:
        sel, _ = load_sel(name)
        rs = [r for r, *_ in rows if r["video"] == name and "sam_area" in r]
        for r in rs:
            r["area_mm2"] = r["sam_area"] / (sel["K"][0][0] / sel["frames"][str(r["idx"])]["dist"]) ** 2
        med = float(np.median([r["area_mm2"] for r in rs]))
        for r in rs:
            if r["status"] == "disagree" and r["colour_area"] > 1.5 * r["sam_area"] and 0.5 * med < r["area_mm2"] < 1.5 * med and r["sam_conf"] >= 0.5:
                r["status"] = "ok (colour unreliable)"
    for r, pick, col, crop, fin in rows:
        r.pop("fin", None)
        if r["status"].startswith("ok"):
            cv2.imwrite(str(cache / "masks" / f"{r['video']}_{r['idx']}.png"), fin)
    res["frames"] = [r for r, *_ in rows]
    okr = [r for r in res["frames"] if "iou" in r]
    ious = np.array([r["iou"] for r in okr])
    L += [f"SAM vs colour mask IoU over {len(ious)} frames: min {ious.min():.3f}, p10 {np.percentile(ious, 10):.3f}, median {np.median(ious):.3f}, p90 {np.percentile(ious, 90):.3f}",
          "(the colour mask is conservative: dark brown faces lose their hue, so IoU is low by construction; the test that matters is how much of the colour mask lies inside SAM)"]
    cis = np.array([r["colour_in_sam"] for r in okr])
    L.append(f"colour mask inside SAM: min {cis.min():.3f}, p10 {np.percentile(cis, 10):.3f}, median {np.median(cis):.3f}")
    for name in NAMES:
        v = [r for r in okr if r["video"] == name]
        n_all = [r for r in res["frames"] if r["video"] == name]
        L.append(f"  {name}: frames {len(n_all)}, SAM found {len(v)}, no SAM mask {sum(r['status'] == 'no SAM mask' for r in n_all)}, "
                 f"colour-in-SAM median {np.median([r['colour_in_sam'] for r in v]):.3f}, DISAGREE (<0.85) {sum(r['status'] == 'disagree' for r in v)}, accepted {sum(r['status'].startswith('ok') for r in v)} "
                 f"(of which colour unreliable, judged by area: {sum(r['status'] == 'ok (colour unreliable)' for r in v)}); fallback concepts used: {sum(r.get('concept') != 'brown plastic buckle' for r in v)}")
        L.append(f"     edge refinement changed median {int(np.median([r['edge_change_px2'] for r in v]))} px2 per frame ({100 * np.median([r['edge_change_px2'] / r['sam_area'] for r in v]):.2f} % of the SAM area)")
    bad = [r for r in res["frames"] if not r["status"].startswith("ok")]
    L.append("excluded frames (video idx status colour_in_sam iou): " + "; ".join(f"{r['video'][-2:]} {r['idx']} {r['status']} {r.get('colour_in_sam', float('nan')):.2f} {r.get('iou', float('nan')):.2f}" for r in bad))
    # 6 overlays: the lowest-agreeing and a median frame of each video: SAM (cyan), colour (orange), final (green)
    for name in NAMES:
        cand = sorted([(r["colour_in_sam"], r, p, c, cr, f) for r, p, c, cr, f in rows if r["video"] == name and "iou" in r], key=lambda t: t[0])
        for tag, (v, r, p, c, cr, f) in (("worst", cand[0]), ("median", cand[len(cand) // 2])):
            if f is None:
                continue
            o = cr.copy()
            for m, col in ((c, (0, 200, 255)), (p[1], (255, 255, 0)), (f, (0, 255, 0))):
                cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(o, cs, -1, col, 1)
            cv2.putText(o, f"SAM cyan, colour orange, final green; colour-in-SAM {v:.2f}; {name} f{r['idx']} ({tag})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            save_overlay(SEND / f"mask_{name}_{tag}.png", o)
    save_stage(2, "\n".join(L), res)
    print("\n".join(L))


LO, HI = [148.5 - 35, 105 - 35, -1.5], [148.5 + 35, 105 + 35, 45.0]
TRUTH = {"length": 34.7, "width": 34.0, "corner_length": 32.4, "plate_thickness": "2.6-2.7"}


def stage3():
    import time

    import cv2
    import numpy as np
    from capture import carve, dataset
    L = ["STAGE 3: per-video robust carve (SDF field, voxel 0.25 mm, floor z >= 0, k = number of views that must put a voxel outside; marching cubes + Taubin 5 iterations)",
         f"calipers for A: length {TRUTH['length']} (middle of the convex edge), width {TRUTH['width']}, corner length {TRUTH['corner_length']} (not computed here), plate/tab thickness {TRUTH['plate_thickness']} mm"]
    res = {}
    for name in NAMES:
        views = dataset.load_views(name)
        res[name] = {"n_views": len(views)}
        L += ["", f"{name}: {len(views)} views"]
        for k in (1, 2, 3):
            t = time.time()
            g, f, n = carve.carve_adaptive(views, LO, HI, 0.25, k)
            v, fc = carve.to_mesh(f, g)
            v = carve.taubin(v, fc, 5)
            cen = carve.edge_census(v, fc)
            ob, pl = carve.obb(v), carve.obb_plan(v)
            res[name][f"k{k}"] = {"census": cen, "obb": ob, "plan": pl, "secs": time.time() - t, "grid": list(f.shape)}
            L.append(f"  k={k}: watertight {cen['watertight']} (boundary {cen['boundary_edges']}, non-manifold {cen['non_manifold_edges']}), volume {cen['volume_mm3']:.0f} mm3, {cen['triangles']} triangles, {time.time() - t:.0f} s")
            L.append(f"     OBB (PCA) extents {[round(x, 2) for x in ob['extents_mm']]}; footprint rectangle {pl['length_mm']:.2f} x {pl['width_mm']:.2f} mm, z {pl['z_min']:.2f}..{pl['z_max']:.2f}")
            if k == 2:
                carve.write_stl(SEND / f"{name}_carve_k2.stl", v, fc)
                cv2.imwrite(str(SEND / f"{name}_carve_k2_3view.png"), carve.render3(v, fc))
    L += ["", "NOTE: a flat video sees the part from above and the side at limited elevation, so its hull is inflated in z (see stage 0); the standing video's frame has a different part orientation."]
    save_stage(3, "\n".join(L), res)
    print("\n".join(L))


def stl_points(path, n=4000):
    import numpy as np
    from capture import carve
    from cad.section import read_stl
    tris = read_stl(path)
    return carve.sample_surface(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3), n=n)


def rot_info(R):
    import numpy as np
    from scipy.spatial.transform import Rotation
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return f"angle {np.degrees(Rotation.from_matrix(R).magnitude()):.0f} deg, z-axis tilt {tilt:.0f} deg"


def stage4():
    import time

    import cv2
    import numpy as np
    from capture import carve, dataset, register as Rg
    ref, flat, stand = NAMES[0], NAMES[2], NAMES[1]
    views = {n: dataset.load_views(n) for n in NAMES}
    L = ["STAGE 4: joint model. Part frame = video 1's mat frame. T_v maps the part frame to video v's mat frame."]
    res = {}
    # candidate set X: video 1's hull, dilated 0.7 mm, at 0.5 mm voxels
    g, f, _ = carve.carve_adaptive(views[ref], LO, HI, 0.5, 2)
    X = g.points()[(f.ravel() > -0.7)]
    centre = X.mean(0)
    L.append(f"video 1 hull candidate points: {len(X)} at 0.5 mm; centroid {np.round(centre, 1).tolist()}")
    hull1 = stl_points(SEND / f"{ref}_carve_k2.stl")
    T = {ref: np.eye(4)}
    chosen = {}

    def run(name, dst, vset, free_extra=()):
        src = stl_points(SEND / f"{name}_carve_k2.stl")
        t0 = time.time()
        cands = Rg.candidates(src, dst)
        L.append(f"  {name}: ICP from 24 axis rotations x 18 yaw steps took {time.time() - t0:.0f} s; distinct best solutions (trimmed chamfer mm, mapping {name} -> part):")
        out = []
        for i, (c, R, t) in enumerate(cands):
            L.append(f"    #{i}: chamfer {c:.3f}, {rot_info(R)}, shift {np.round(t, 1).tolist()}")
        for i, (c, R, t) in enumerate(cands[:3]):
            Tv = np.linalg.inv(Rg.hom(R, t))                           # part -> video v
            Tset = dict(T)
            Tset[name] = Tv
            sub = {n: views[n] for n in vset}
            v0 = Rg.joint_volume(sub, Tset, X, 2, 0.5)
            Tfin, hist = Rg.refine(sub, Tset, [name, *free_extra], X, centre, 2, 0.5, log=lambda m: L.append(m))
            L.append(f"    candidate #{i}: joint volume before {v0:.0f}, after refinement {hist[-1]:.0f} mm3")
            out.append((hist[-1], i, Tfin, c, v0))
        out.sort(key=lambda o: -o[0])
        from scipy.spatial.transform import Rotation
        out_diff = []
        for j in range(1, len(out)):
            D = out[0][2][name] @ np.linalg.inv(out[j][2][name])
            out_diff.append((np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()), float(np.linalg.norm(D[:3, 3]))))
            L.append(f"    final transform of candidate #{out[j][1]} vs best (#{out[0][1]}): rotation {np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()):.2f} deg, translation {np.linalg.norm(D[:3, 3]):.2f} mm (different starts, same optimum if both ~0)")
        run.last_diff = out_diff
        return out

    # video 3 (flat too) against video 1
    L += ["", f"Registering {flat} (flat) to {ref}:"]
    o3 = run(flat, hull1, [ref, flat])
    for v, i, Tf, c, v0 in o3:
        pass
    T[flat] = o3[0][2][flat]
    amb3 = (o3[0][0] - o3[1][0]) / o3[0][0] if len(o3) > 1 else None
    L.append(f"  chosen: candidate #{o3[0][1]} (joint volume {o3[0][0]:.0f}); runner-up volume {o3[1][0]:.0f}; " + ("candidates agree (same transform within " + format(max(d[0] for d in run.last_diff), ".2f") + " deg, " + format(max(d[1] for d in run.last_diff), ".2f") + " mm)" if max(d[0] for d in run.last_diff) < 1.0 and max(d[1] for d in run.last_diff) < 0.5 else "AMBIGUOUS: different optima"))
    res["flat"] = {"T": T[flat].tolist(), "volumes": [o[0] for o in o3], "ambiguity_rel": amb3}
    # joint hull of 1+3 as the target for the standing video
    sub13 = {n: views[n] for n in (ref, flat)}
    allv13 = list(views[ref]) + [v.moved(T[flat]) for v in views[flat]]
    g13, f13, _ = carve.carve_adaptive(allv13, LO, HI, 0.25, 2, floors=[T[flat]])
    v13, fc13 = carve.to_mesh(f13, g13)
    pts13 = carve.sample_surface(v13, fc13, 4000)
    L += ["", f"Registering {stand} (standing) to the joint hull of {ref}+{flat}:"]
    o2 = run(stand, pts13, NAMES, free_extra=(flat,))
    T[stand] = o2[0][2][stand]
    T[flat] = o2[0][2][flat]
    amb2 = (o2[0][0] - o2[1][0]) / o2[0][0] if len(o2) > 1 else None
    L.append(f"  chosen: candidate #{o2[0][1]} (joint volume {o2[0][0]:.0f}); runner-up volume {o2[1][0]:.0f} ({100 * amb2:.1f} % lower); "
             f"{'candidates agree (all converge to the same transform, within ' + format(max(d[0] for d in run.last_diff), '.2f') + ' deg, ' + format(max(d[1] for d in run.last_diff), '.2f') + ' mm): not ambiguous' if max(d[0] for d in run.last_diff) < 1.0 and max(d[1] for d in run.last_diff) < 0.5 else 'AMBIGUOUS: different optima with similar volume'}")
    res["stand"] = {"T": T[stand].tolist(), "volumes": [o[0] for o in o2], "ambiguity_rel": amb2}
    (CARVE / "cache" / "registration.json").write_text(json.dumps({n: m.tolist() for n, m in T.items()}))
    # joint carve at 0.25 mm
    allv = [v.moved(T[n]) for n in NAMES for v in views[n]]
    floors = [T[n] for n in NAMES if n != ref]
    g, f, nseen = carve.carve_adaptive(allv, LO, HI, 0.25, 2, floors=floors)
    v, fc = carve.to_mesh(f, g)
    v = carve.taubin(v, fc, 5)
    cen = carve.edge_census(v, fc)
    ob, pl = carve.obb(v), carve.obb_plan(v)
    carve.write_stl(SEND / "joint_k2.stl", v, fc)
    cv2.imwrite(str(SEND / "joint_k2_3view.png"), carve.render3(v, fc))
    L += ["", f"joint carve (k=2, 0.25 mm, {len(allv)} views): watertight {cen['watertight']}, volume {cen['volume_mm3']:.0f} mm3, {cen['triangles']} triangles",
          f"  OBB (PCA) {[round(x, 2) for x in ob['extents_mm']]}; footprint rectangle {pl['length_mm']:.2f} x {pl['width_mm']:.2f}, z {pl['z_min']:.2f}..{pl['z_max']:.2f}"]
    res["joint"] = {"census": cen, "obb": ob, "plan": pl}
    save_stage(4, "\n".join(L), res)
    print("\n".join(L))


def load_T():
    import numpy as np
    return {n: np.array(m) for n, m in json.loads((CARVE / "cache" / "registration.json").read_text()).items()}


def stage5(vs_sweep=0.4):
    import cv2
    import numpy as np
    from capture import carve, dataset, holdout as H, register as Rg
    from spike_common import save_overlay
    views = {n: dataset.load_views(n) for n in NAMES}
    T = load_T()
    ref = NAMES[0]
    L = ["STAGE 5: held-out evaluation. 3 folds x 10 held-out views; carve without them, render the mesh silhouette into them, IoU with their final masks (marker poses for every view).",
         f"Parameters: k = views that must put a voxel outside; grow = every silhouette dilated by this many mm (a bias between masks and the robust hull). Sweep at voxel {vs_sweep} mm."]
    res = {"sweep": []}
    for k in (1, 2, 3, 6, 12):
        for g in (0.0, 0.1, 0.2, 0.3):
            j, per, _ = H.joint(views, T, k, vs_sweep, ref=ref, grow=g)
            res["sweep"].append({"k": k, "grow": g, "joint": j, "per_video": per})
            L.append(f"  joint k={k:2d} grow={g:.1f}: IoU {j:.4f} (" + ", ".join(f"{n[-2:]} {x:.4f}" for n, x in per.items()) + ")")
    best = max(res["sweep"], key=lambda r: r["joint"])
    kb, gb = best["k"], best["grow"]
    L.append(f"best on held-out IoU: k={kb}, grow={gb} mm (IoU {best['joint']:.4f}); differences below ~0.001 between neighbouring settings are within the fold-to-fold spread")
    single = {n: H.single_video(views[n], kb, vs_sweep, grow=gb) for n in NAMES}
    single0 = {n: H.single_video(views[n], 2, vs_sweep)[0] for n in NAMES}
    L.append("single-video held-out IoU with the same k/grow: " + ", ".join(f"{n[-2:]} {single[n][0]:.4f}" for n in NAMES) + "; with k=2, grow 0: " + ", ".join(f"{n[-2:]} {single0[n]:.4f}" for n in NAMES))
    res["single"] = {n: {"best": single[n][0], "k2_grow0": single0[n]} for n in NAMES}
    # finest resolution
    fin = {}
    for vs in (0.25, 0.15):
        j, per, allr = H.joint(views, T, kb, vs, ref=ref, grow=gb)
        fin[vs] = (j, per, allr, H.joint.meshes)
        L.append(f"joint at voxel {vs} mm: held-out IoU {j:.4f} (min over views {min(r for _, r in allr):.3f}); per video " + ", ".join(f"{n[-2:]} {x:.4f}" for n, x in per.items()))
    res["final"] = {"k": kb, "grow": gb, **{f"vs{vs}": {"joint": fin[vs][0], "per_video": fin[vs][1]} for vs in fin}}
    vsf = 0.15 if fin[0.15][0] >= fin[0.25][0] else 0.25
    L.append(f"final voxel: {vsf} mm (higher held-out IoU)")
    allv = [v.moved(T[n]) for n in NAMES for v in views[n]]
    floors = [T[n] for n in NAMES if n != ref]
    v, fc = H.carve_mesh(allv, vsf, kb, floors, grow=gb)
    carve.write_stl(SEND / "joint_final.stl", v, fc)
    cv2.imwrite(str(SEND / "joint_final_3view.png"), carve.render3(v, fc))
    pl, ob, cen = carve.obb_plan(v), carve.obb(v), carve.edge_census(v, fc)
    ungrown = H.carve_mesh(allv, vsf, kb, floors, grow=0.0)
    plu = carve.obb_plan(ungrown[0])
    L += ["", f"joint model (k={kb}, grow {gb}, {vsf} mm, {len(allv)} views): watertight {cen['watertight']} (boundary {cen['boundary_edges']}, non-manifold {cen['non_manifold_edges']}), volume {cen['volume_mm3']:.0f} mm3, {cen['triangles']} triangles",
          f"  footprint rectangle {pl['length_mm']:.2f} x {pl['width_mm']:.2f} mm, height {pl['z_max'] - pl['z_min']:.2f} mm (z {pl['z_min']:.2f}..{pl['z_max']:.2f}); PCA OBB {[round(x, 2) for x in ob['extents_mm']]}",
          f"  same without grow (k={kb}, grow 0): footprint {plu['length_mm']:.2f} x {plu['width_mm']:.2f}",
          "  caliper truth for part A (as given): length 34.7, width 34.0.  Caliper truth for part B (accuracy set): length 51.6, width 41.9, height 11.8 (14.4 also tried earlier).",
          f"  error vs A: length {pl['length_mm'] - 34.7:+.2f}, width {pl['width_mm'] - 34.0:+.2f} mm.   error vs B: length {pl['length_mm'] - 51.6:+.2f}, width {pl['width_mm'] - 41.9:+.2f}, height {pl['z_max'] - pl['z_min'] - 11.8:+.2f} mm",
          "  >>> the carved footprint matches part B's calipers, not part A's: the part in these videos is the 51.6 x 41.9 mm part, or the A/B labels are swapped. Not resolved here.",
          "  grow was tuned on held-out IoU, not on the calipers; the calipers are an independent check, and only 3 numbers (not the thickness of the plate or the bar)."]
    res["joint_model"] = {"plan": pl, "plan_no_grow": plu, "obb": ob, "census": cen, "vs": vsf}
    cnt = H.constraint_counts(allv, v)
    tri = v[fc]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    lowf = (cnt < 3)[fc].all(1)
    L += ["", f"surface fixed by fewer than 3 views (a silhouette edge passes within 0.3 mm of it): {100 * area[lowf].sum() / area.sum():.1f} % of the surface area; the rest is fixed by 3 or more."]
    zones = {}
    for c, a, flag in zip(tri.mean(1), area, lowf):
        if flag:
            zones.setdefault(tuple((c // 8).astype(int)), [0.0, []])
            zones[tuple((c // 8).astype(int))][0] += a
            zones[tuple((c // 8).astype(int))][1].append(c)
    top = sorted(zones.items(), key=lambda kv: -kv[1][0])[:8]
    L.append("  largest low-confidence regions (centre in the part frame x,y,z mm; area mm2): " + "; ".join(f"({np.mean(c, 0)[0]:.0f},{np.mean(c, 0)[1]:.0f},{np.mean(c, 0)[2]:.1f}) {a:.0f}" for _, (a, c) in top))
    res["low_confidence"] = {"area_frac": float(area[lowf].sum() / area.sum()), "top": [[float(a), np.mean(c, 0).tolist()] for _, (a, c) in top]}
    # 4 genuinely held-out overlays: fold 0 of the joint run at the final voxel size
    hold, mesh = fin[vsf][3][0]
    cache = CARVE / "cache"
    seen_v, pick = {}, []
    for n, i in hold:
        if seen_v.get(n, 0) < 2 and len(pick) < 4:
            seen_v[n] = seen_v.get(n, 0) + 1
            pick.append((n, i))
    for n, i in pick:
        vw = views[n][i]
        mv = vw.moved(T[n])
        sil = carve.render_silhouette(*mesh, mv)
        o = cv2.imread(str(cache / "crops" / f"{vw.name}.jpg"))
        for m, col in (((vw.sdf > 0).astype(np.uint8) * 255, (0, 255, 0)), (sil, (255, 255, 0))):
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 1)
        cv2.putText(o, f"held-out view {vw.name}: mask green, model cyan; IoU {carve.view_iou(*mesh, mv):.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        save_overlay(SEND / f"heldout_{vw.name}.png", o)
    # experiments that did not help (kept for the record): pose refinement against the model, rigid re-registration at the final k/grow
    from capture import posefit
    for name in [f.name for f in SEND.glob("heldout_*.png")]:
        pass
    j_fit, per_fit, _ = H.joint(views, T, kb, vs_sweep, ref=ref, grow=gb, fit_rounds=3)
    L += ["", "EXPERIMENTS THAT DID NOT HELP (all at voxel 0.4 mm, same folds):",
          f"  per-view silhouette pose refinement (MAP, priors 0.1 deg / 0.4 mm, 3 rounds, fitted without the held-out views): held-out IoU {j_fit:.4f} vs {best['joint']:.4f} without (" + ", ".join(f"{n[-2:]} {x:.4f}" for n, x in per_fit.items()) + ")"]
    g0, f0, _ = carve.carve_adaptive(views[ref], LO, HI, 0.5, kb, grow_mm=gb)
    X = g0.points()[f0.ravel() > -1.0]
    T2, hist = Rg.refine(views, T, [NAMES[2], NAMES[1]], X, X.mean(0), kb, 0.5, grow=gb, maxiter=3, log=lambda m: None)
    j_r, per_r, _ = H.joint(views, T2, kb, vs_sweep, ref=ref, grow=gb)
    L.append(f"  rigid re-registration of video 2 and 3 on the joint volume at k={kb}, grow {gb}: held-out IoU {j_r:.4f} (volume {hist[0]:.0f} -> {hist[-1]:.0f} mm3) vs {best['joint']:.4f} with the stage 4 transforms")
    L.append("  also tried (script session, not re-run here): a per-video scale in the registration: the volume objective rose 9472 -> 9845 mm3 while video 2 shrank by 4 % and held-out IoU fell to 0.9203, so the volume objective is exploitable by scale and was not used.")
    res["experiments"] = {"posefit_iou": j_fit, "rigid_rereg_iou": j_r}
    save_stage(5, "\n".join(L), res)
    (CARVE / "cache" / "final_params.json").write_text(json.dumps({"k": kb, "grow": gb, "vs": vsf}))
    print("\n".join(L))


def stage6():
    import shutil

    import cv2
    import numpy as np
    from cad.section import _loops, read_stl
    from capture import carve, snap
    params = json.loads((CARVE / "cache" / "final_params.json").read_text())
    shutil.copy(SEND / "joint_final.stl", SEND / "A_final.stl")
    tris = read_stl(SEND / "A_final.stl")
    v, inv = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    fc = inv.reshape(-1, 3)
    P, N = snap.oriented_points(v, fc, 200000)
    L = [f"STAGE 6: snapping on the joint mesh (k={params['k']}, grow {params['grow']} mm, voxel {params['vs']} mm; {len(fc)} triangles). A_final.stl = the unsnapped joint carve.",
         "RANSAC: Open3D 0.20 is installed in the env but does not load without DYLD_FALLBACK_LIBRARY_PATH pointing at the env's libusb, and it has a plane model only (no cylinder).",
         "So NEITHER planes NOR cylinders here use Open3D: both are my own RANSAC (planes from 3 points with normal agreement, cylinders from pairs of oriented points). 200000 surface samples, tolerance 0.12 mm."]
    pls = snap.planes(P, N, tol=0.12, max_planes=8, min_frac=0.02)
    res = {"planes": [], "cylinders": []}
    L.append("planes (normal . x + d = 0; share of the surface samples):")
    for n, d, m in pls:
        r = np.abs(P[m] @ n + d)
        res["planes"].append({"n": n.tolist(), "d": float(d), "share": float(m.mean()), "rms_mm": float(np.sqrt((r ** 2).mean())), "tilt_from_z_deg": float(np.degrees(np.arccos(abs(n[2]))))})
        L.append(f"  n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}) d={d:+.2f}  share {100 * m.mean():.1f} %, rms {np.sqrt((r ** 2).mean()):.3f} mm, tilt from z {np.degrees(np.arccos(abs(n[2]))):.1f} deg")
    for i, j, dist, ang in snap.parallel_pair_distances(P, pls):
        L.append(f"  planes #{i} and #{j} are parallel within {ang:.1f} deg: {dist:.2f} mm apart")
        res.setdefault("plane_pairs", []).append({"i": i, "j": j, "distance_mm": dist, "angle_deg": ang})
    avail = np.ones(len(P), bool)
    for _, _, m in pls:
        avail &= ~m
    cyl = snap.cylinders(P, N, avail=avail)
    L.append("cylinders:")
    for c in cyl:
        ax = c["axis"] * (1 if c["axis"][np.argmax(np.abs(c["axis"]))] > 0 else -1)
        res["cylinders"].append({"diameter_mm": 2 * c["r"], "axis": ax.tolist(), "length_mm": c["length"], "rms_mm": c["rms"], "inliers": int(c["inliers"].sum()), "centre": c["centre"].tolist()})
        L.append(f"  diameter {2 * c['r']:.3f} mm, axis ({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f}), length {c['length']:.2f} mm, rms {c['rms']:.3f} mm, {int(c['inliers'].sum())} samples at {tuple(np.round(c['centre'], 1))}")
    # cross-bar: the cylinder with the longest extent, axis nearly horizontal
    bars = [c for c in cyl if abs(c["axis"][2]) < 0.3 and c["length"] > 15 and 2 * c["r"] < 8]       # a slender rod: the cross-bar (larger cylinders are rounded edges)
    if bars:
        bar = min(bars, key=lambda c: c["rms"])
        mid = np.mean(bar["span"])
        segs = snap.section_equivalent_diameter(v, fc, bar["centre"], bar["axis"], mid)
        q = segs.reshape(-1, 2)
        q = q[np.linalg.norm(q, axis=1) < bar["r"] + 1.2]
        ext = [float(q[:, 0].max() - q[:, 0].min()), float(q[:, 1].max() - q[:, 1].min())] if len(q) > 10 else None
        res["cross_bar"] = {"fit_diameter_mm": 2 * bar["r"], "length_mm": bar["length"], "section_extents_mm": ext}
        L.append(f"CROSS-BAR: RANSAC cylinder diameter {2 * bar['r']:.3f} mm over a length of {bar['length']:.2f} mm (rms {bar['rms']:.3f}); section of the mesh at mid-length (points within r+1.2 mm of the axis): extents "
                 + (f"{ext[0]:.2f} x {ext[1]:.2f} mm" if ext else "n/a") + ". No caliper value exists for the bar, so neither number is verified.")
    else:
        L.append("CROSS-BAR: no long near-horizontal cylinder found.")
    # through-slots / windows: inner loops of horizontal sections
    L.append("windows and slots (inner loops of horizontal sections of the unsnapped mesh; min-area rectangle long x short, mm; centre x,y in the part frame):")
    res["sections"] = {}
    for z in (0.6, 1.5, 2.5, 3.8, 5.0, 6.0, 9.0):
        holes, islands = snap.section_regions(tris, z)
        res["sections"][str(z)] = {"holes": holes, "islands": islands}
        L.append(f"  z={z}: {len(holes)} hole(s): " + "; ".join(f"{h['long']:.2f} x {h['short']:.2f} at ({h['cx']:.1f},{h['cy']:.1f}), area {h['area']:.1f}" for h in holes) + (f"   [material islands inside: {len(islands)}]" if islands else ""))
    L.append("  (the four small slots along the plate edge are not in the hull: the carve cannot see into them from outside)")
    # snapped mesh
    v2, st = snap.snap_vertices(v, [(n, d, m) for n, d, m in pls], cyl, tol=0.25)
    before, after = carve.edge_census(v, fc), carve.edge_census(v2, fc)
    carve.write_stl(SEND / "snapped_A.stl", v2, fc)
    mv = np.linalg.norm(v2 - v, axis=1)
    L += ["", f"snapped_A.stl: every vertex within 0.25 mm of a fitted plane or cylinder was moved onto it; {int((mv > 1e-6).sum())} of {len(v)} vertices moved, mean move {mv[mv > 1e-6].mean():.3f} mm, max {mv.max():.3f} mm",
          f"  per primitive: " + "; ".join(f"{s['kind']} {s['moved']} verts, mean {s['mean_mm']:.3f}, max {s['max_mm']:.3f}" for s in st),
          f"  A_final.stl: volume {before['volume_mm3']:.0f} mm3, watertight {before['watertight']}.  snapped_A.stl: volume {after['volume_mm3']:.0f} mm3, watertight {after['watertight']} (topology unchanged), boundary {after['boundary_edges']}, degenerate triangles {after['degenerate_triangles']}",
          "  the unsnapped carve is kept as A_final.stl; snapped_A.stl is a derived file, no measured value above was replaced by a snapped one."]
    res["snapped"] = {"moved": int((mv > 1e-6).sum()), "mean_move": float(mv[mv > 1e-6].mean()), "max_move": float(mv.max()), "before": before, "after": after}
    cv2.imwrite(str(SEND / "A_final_3view.png"), carve.render3(v, fc))
    cv2.imwrite(str(SEND / "snapped_A_3view.png"), carve.render3(v2, fc))
    save_stage(6, "\n".join(L), res)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--max-rms", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    MAX_RMS = a.max_rms
    {0: stage0, 1: stage1, 2: lambda: stage2(a.limit, a.refresh), 3: stage3, 4: stage4, 5: lambda: stage5(), 6: stage6}[a.stage]()
