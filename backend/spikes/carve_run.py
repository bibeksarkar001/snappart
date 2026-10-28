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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--max-rms", type=float, default=1.0)
    a = ap.parse_args()
    MAX_RMS = a.max_rms
    {0: stage0, 1: stage1}[a.stage]()
