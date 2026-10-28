"""v2: features on the carved part (grooves, lock bump, lock pin), fast pipeline, frozen settings. Usage: python backend/spikes/v2_run.py --stage base|1|2|3|4

Reads the v1 caches (out/carve/cache); writes out/v2/SEND_TO_CLAUDE/ (summary.txt first, measured numbers only).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from spike_common import OUT  # noqa: E402

CARVE = OUT / "carve"
CACHE = CARVE / "cache"
V1 = CARVE / "SEND_TO_CLAUDE"
V2 = OUT / "v2"
SEND = V2 / "SEND_TO_CLAUDE"
NAMES = ["A_orbit_01", "A_orbit_02", "A_orbit_03"]
SEND.mkdir(parents=True, exist_ok=True)


def date():
    return subprocess.run(["date"], capture_output=True, text=True).stdout.strip()


def save_stage(key: str, text: str, results: dict):
    """Write stage_<key>.txt, rebuild summary.txt (header first, then the stages), merge results.json."""
    (SEND / f"stage_{key}.txt").write_text(text)
    head = SEND / "header.txt"
    parts = ([head.read_text()] if head.exists() else []) + [p.read_text() for p in sorted(SEND.glob("stage_*.txt"))]
    (SEND / "summary.txt").write_text("\n\n".join(parts))
    f = SEND / "results.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[key] = results
    f.write_text(json.dumps(cur, indent=1, default=float))


def load_mesh(path):
    import numpy as np
    from cad.section import read_stl
    tris = read_stl(path)
    v, inv = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def load_T():
    import numpy as np
    return {n: np.array(m) for n, m in json.loads((CACHE / "registration.json").read_text()).items()}


def surface_stats(v, fc):
    """How clean a mesh is: watertightness, components, dihedral noise between neighbouring faces, and plane-fit rms on the big planes."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from capture import carve, snap
    cen = carve.edge_census(v, fc)
    e = np.sort(np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(fc)), 3)
    o = np.lexsort(e.T[::-1])
    e, fid = e[o], fid[o]
    same = (e[1:] == e[:-1]).all(1)
    a, b = fid[:-1][same], fid[1:][same]
    nrm = np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    dih = np.degrees(np.arccos(np.clip((nrm[a] * nrm[b]).sum(1), -1, 1)))
    ncomp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(fc), len(fc))))[0]
    P, N = snap.oriented_points(v, fc, 100000)
    pls = snap.planes(P, N, tol=0.12, max_planes=6, min_frac=0.02)
    prms = [float(np.sqrt(((P[m] @ n + d) ** 2).mean())) for n, d, m in pls]
    share = float(sum(m.mean() for _, _, m in pls))
    flat = dih < 30                                                       # dihedral noise on smooth regions (not on real edges)
    return {"watertight": cen["watertight"], "triangles": cen["triangles"], "volume_mm3": cen["volume_mm3"], "components": int(ncomp),
            "dihedral_median_deg": float(np.median(dih[flat])), "dihedral_p90_deg": float(np.percentile(dih[flat], 90)),
            "share_on_6_planes": share, "plane_rms_mm": prms}


def stage_base():
    """Compare the owner's baseline joint_k2.stl with A_final.stl: held-out IoU (same folds, same code) and surface cleanliness."""
    import numpy as np
    from capture import carve, dataset, holdout as H
    t0, d0 = time.time(), date()
    views = {n: dataset.load_views(n) for n in NAMES}
    T = load_T()
    L = [f"STAGE base: which mesh is the baseline. start {d0}",
         "Held-out IoU: 3 folds x 10 held-out views spread over the 3 videos; carve without them at the mesh's own settings, render the mesh silhouette into them, IoU with their masks."]
    res = {}
    for name, k, grow, vs in (("joint_k2.stl", 2, 0.0, 0.25), ("A_final.stl", 12, 0.1, 0.15)):
        j, per, allr = H.joint(views, T, k, vs, ref=NAMES[0], grow=grow)
        v, fc = load_mesh(V1 / name)
        s = surface_stats(v, fc)
        pl = carve.obb_plan(v)
        res[name] = {"k": k, "grow": grow, "vs": vs, "heldout_iou": j, "per_video": per, "min_view_iou": float(min(r for _, r in allr)), "surface": s, "plan": pl}
        L += [f"  {name} (k={k}, grow {grow} mm, voxel {vs} mm): held-out IoU {j:.4f} (min view {min(r for _, r in allr):.3f}; " + ", ".join(f"{n[-2:]} {x:.4f}" for n, x in per.items()) + ")",
              f"    footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f}, height {pl['z_max'] - pl['z_min']:.2f} mm; watertight {s['watertight']}, {s['components']} component(s), {s['triangles']} triangles, volume {s['volume_mm3']:.0f} mm3",
              f"    dihedral angle between neighbouring faces on smooth regions: median {s['dihedral_median_deg']:.2f} deg, 90th pct {s['dihedral_p90_deg']:.2f} deg; "
              f"{100 * s['share_on_6_planes']:.1f} % of the surface on 6 RANSAC planes (tol 0.12 mm), plane rms " + ", ".join(f"{r:.3f}" for r in s["plane_rms_mm"]) + " mm"]
    a, b = res["joint_k2.stl"], res["A_final.stl"]
    pick = "A_final.stl" if b["heldout_iou"] > a["heldout_iou"] and b["surface"]["dihedral_median_deg"] <= a["surface"]["dihedral_median_deg"] * 1.2 else "joint_k2.stl"
    res["baseline"] = pick
    L.append(f"BASELINE: {pick}. " + ("It has the higher held-out IoU and a surface at least as clean (dihedral noise within 20 % of joint_k2)." if pick == "A_final.stl"
                                      else "A_final.stl does not beat it on both held-out IoU and surface cleanliness."))
    L.append(f"end {date()}; elapsed {time.time() - t0:.0f} s")
    save_stage("0_base", "\n".join(L), res)
    print("\n".join(L))


PARAMS = {"k": 12, "grow": 0.1, "vs": 0.15}          # A_final settings (v1 stage 5, held-out tuned)
GROOVE_DEPTH_USER = 3.7                               # mm, owner's caliper value: NOT measured from the videos


def all_views():
    """[(video, view moved into the part frame)] and the matching gray crops."""
    import cv2
    from capture import dataset
    T = load_T()
    vv, imgs = [], []
    for n in NAMES:
        for v in dataset.load_views(n):
            vv.append((n, v.moved(T[n])))
            imgs.append(cv2.imread(str(CACHE / "crops" / f"{v.name}.jpg"), 0))
    return vv, imgs


def frame_view(video, idx):
    """Full undistorted frame and its view (part frame) from the stage 1 selection."""
    import cv2
    import numpy as np
    from capture import carve
    sel = json.loads((CACHE / f"sel_{video}.json").read_text())
    fr = sel["frames"][str(idx)]
    img = cv2.imread(fr["file"])
    v = carve.View(np.array(sel["K"], float), cv2.Rodrigues(np.array(fr["rvec"], float))[0], np.array(fr["tvec"], float),
                   np.zeros(img.shape[:2], np.float32), f"{video}_{idx}", video)
    return img, v.moved(load_T()[video])


def joint_field(vv, vs=None):
    from capture import carve, holdout as H
    T = load_T()
    floors = [T[n] for n in NAMES if n != NAMES[0]]
    return carve.carve_adaptive([v for _, v in vv], H.LO, H.HI, vs or PARAMS["vs"], PARAMS["k"], floors=floors, grow_mm=PARAMS["grow"])


def groove_boxes(ops, face, depth_mm, out_mm=3.0):
    """Pocket boxes: the triangulated opening rectangle (centre, long axis from its corners), axis = the face normal, depth_mm into the part
    from the opening plane and out_mm outward so the pocket breaks through whatever hull material lies outside the opening plane."""
    import numpy as np
    out = []
    for o in ops:
        c = o["corners"]
        lng = (c[1] - c[0]) + (c[2] - c[3])
        lng -= face["n"] * (lng @ face["n"])
        lng /= np.linalg.norm(lng)
        axes = np.stack([lng, np.cross(face["n"], lng), face["n"]])
        out.append((o["centre"] + face["n"] * (out_mm - depth_mm) / 2, axes, np.array([o["L"] / 2, o["S"] / 2, (depth_mm + out_mm) / 2])))
    return out


def overlay(img, fv, v, fc, ops, title):
    """Frame | model rendered in the frame's camera pose (shaded, orange) with the triangulated opening rectangles (cyan), cropped to the part."""
    import cv2
    import numpy as np
    from capture import carve, features as F
    sh, mk = F.render_shaded(v, fc, fv)
    o_img = img.copy()
    sel = mk > 0
    o_img[sel] = (0.45 * img[sel] + 0.55 * sh[sel]).astype(np.uint8)
    for o in ops:
        u, vq, _ = carve.project(fv, o["corners"])
        cv2.polylines(o_img, [np.round(np.stack([u, vq], 1) * 16).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA, 4)
    ys, xs = np.nonzero(mk)
    y0, y1, x0, x1 = max(ys.min() - 150, 0), ys.max() + 150, max(xs.min() - 150, 0), xs.max() + 150
    both = np.hstack([img[y0:y1, x0:x1], o_img[y0:y1, x0:x1]])
    cv2.putText(both, f"{title}: frame | model in this camera pose (orange), openings cyan", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    return both


def find_grooves(v, fc, vv, imgs, log=lambda m: None, max_angle_deg=15.0):
    """Openings on the candidate face whose triangulated corner planes agree with it (within max_angle_deg) and that has the most view detections.
    Returns (face, fused openings, per-video detections, per-face table)."""
    import numpy as np
    from capture import features as F
    table, found = [], None
    for fi, face in enumerate(F.candidate_faces(v, fc)):
        per = F.detect_on_face(vv, imgs, face)
        ops = F.fuse(per, face)
        nview = sum(o["n_views"] for o in ops)
        ang = float(np.degrees(np.arccos(np.clip(np.mean([o["normal"] for o in ops], 0) @ face["n"] / np.linalg.norm(np.mean([o["normal"] for o in ops], 0)), -1, 1)))) if ops else None
        row = {"face": fi, "kind": face["kind"], "n": face["n"].tolist(), "centre_x": float(face["O"][0] + face["len_s"] / 2 * face["e1"][0] + face["len_t"] / 2 * face["e2"][0]),
               "inliers": face["inliers"], "rms": face["plane_rms"], "openings_per_video": {k: len(o) for k, o in per.items()}, "fused": len(ops), "view_detections": nview, "corner_plane_angle_deg": ang}
        table.append(row)
        log(f"face #{fi} ({face['kind']}, near x={row['centre_x']:.0f}): n=({face['n'][0]:+.3f},{face['n'][1]:+.3f},{face['n'][2]:+.3f}), {face['inliers']} samples, rms {face['plane_rms']:.3f} mm, "
            f"{sum(F.facing(x, face) >= 0.3 for _, x in vv)} views facing it; openings (>= 3 views in one video): " + (", ".join(f"{k[-2:]}: {len(o)}" for k, o in per.items()) or "none")
            + (f"; fused {len(ops)}, {nview} view detections, triangulated corner planes {ang:.1f} deg from this plane" if ops else ""))
        if ops and ang < max_angle_deg and (found is None or nview > found[3]):
            found = (face, ops, per, nview)
    return (found[:3] if found else (None, [], {})) + (table,)


def stage1():
    import cv2
    import numpy as np
    from capture import carve, features as F
    t0, d0 = time.time(), date()
    v, fc = load_mesh(V1 / "A_final.stl")
    vv, imgs = all_views()
    L = [f"STAGE 1: grooves. start {d0}. Baseline mesh A_final.stl; {len(vv)} views (crops, marker poses, v1 registration).",
         "Candidate faces: RANSAC planes at both ends of the long axis (the vertical end plane and the horizontal planes within 20 mm of each end). On each: rectify every "
         "view facing it (cos >= 0.3), detect dark rectangles, triangulate each opening's centre and 4 corners per video (>= 3 views). Kept: the face whose openings' "
         "triangulated corner planes agree with it (< 15 deg) and that has the most view detections.",
         "Attempt history: attempt 1 assumed the vertical end face and rectified onto it only; the corners triangulated there give a plane 67 deg off it (table below), i.e. the "
         "openings are NOT in the vertical face. Attempt 2 picked by distance to the plane and chose a spurious 1-opening face. This is attempt 3 (corner planes)."]
    res = {}
    face, ops, per, table = find_grooves(v, fc, vv, imgs, L.append)
    res["faces"] = table
    if face is None:
        L.append("NO OPENINGS FOUND.")
        save_stage("1_grooves", "\n".join(L), res)
        return
    L.append(f"CHOSEN: {face['kind']} face n=({face['n'][0]:+.3f},{face['n'][1]:+.3f},{face['n'][2]:+.3f}) (the face that rests on the mat in video 1 and faces up in video 3): {len(ops)} openings")
    L.append("per video (triangulated; centre, ray residual max, corner-plane normal, size from the corners):")
    for vid, lst in per.items():
        for o in lst:
            L.append(f"  {vid}: centre ({o['p'][0]:.2f}, {o['p'][1]:.2f}, {o['p'][2]:.2f}) from {len(o['views'])} views, baseline {o['baseline_deg']:.0f} deg, residual max {max(o['res']):.2f} mm, "
                     f"normal ({o['normal'][0]:+.2f},{o['normal'][1]:+.2f},{o['normal'][2]:+.2f}), size {o['L']:.2f} x {o['S']:.2f}")
    L.append("OPENINGS (size and centre from the video with the most views; caliper 7.4 x 2.6):")
    res["openings"] = []
    for k, o in enumerate(ops):
        c = o["centre"]
        oth = {vid: (r["L"], r["S"], r["p"]) for vid, r in o["by_video"].items() if vid != o["primary"]}
        L.append(f"  #{k + 1}: centre ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}); {o['n_views']} agreeing views in {len(o['by_video'])} video(s) (primary {o['primary'][-2:]}); "
                 f"size {o['L']:.2f} x {o['S']:.2f} mm, error {o['L'] - 7.4:+.2f}, {o['S'] - 2.6:+.2f}; opening plane {o['dist_from_face_mm']:+.2f} mm from the hull face (+ outward)"
                 + "".join(f"; {vid[-2:]}: {x[0]:.2f} x {x[1]:.2f}, centre offset ({x[2][0] - c[0]:+.2f}, {x[2][1] - c[1]:+.2f}, {x[2][2] - c[2]:+.2f})" for vid, x in oth.items()))
        res["openings"].append({"centre": c.tolist(), "corners": o["corners"].tolist(), "long_mm": o["L"], "short_mm": o["S"], "views": o["n_views"], "primary": o["primary"],
                                "videos": list(o["by_video"]), "dist_from_face_mm": o["dist_from_face_mm"],
                                "by_video": {vid: {"L": r["L"], "S": r["S"], "centre": r["p"].tolist(), "views": len(r["views"])} for vid, r in o["by_video"].items()}})
    Ls, Ss = [o["L"] for o in ops], [o["S"] for o in ops]
    L.append(f"  mean size {np.mean(Ls):.2f} x {np.mean(Ss):.2f} mm (caliper 7.4 x 2.6: {np.mean(Ls) - 7.4:+.2f}, {np.mean(Ss) - 2.6:+.2f}); pitch "
             + ", ".join(f"{p:.2f}" for p in np.diff([o["e1"] for o in ops])) + " mm")
    L.append(f"orientation: the openings lie in the {face['kind']} face (normal {np.round(face['n'], 2).tolist()} in the part frame): long side along the end edge (7.4), "
             "short side across it (2.6); 3.7 is along the normal, the depth into the part, as image 4 labels it ('Height/depth', drawn vertically in a video 3 frame where "
             "this face is on top).")
    res["mean_size"] = [float(np.mean(Ls)), float(np.mean(Ss))]
    fieldg, field, _ = joint_field(vv)
    boxes = groove_boxes(ops, face, GROOVE_DEPTH_USER)
    cut = F.cut_pockets(field, fieldg, boxes)
    v0, f0 = carve.to_mesh(field, fieldg)
    v0 = carve.taubin(v0, f0, 5)
    v1, f1 = carve.to_mesh(cut, fieldg)
    v1 = carve.taubin(v1, f1, 5)
    c0, c1 = carve.edge_census(v0, f0), carve.edge_census(v1, f1)
    carve.write_stl(SEND / "B_grooves.stl", v1, f1)
    np.save(V2 / "grooves.npy", np.array([np.r_[b[0], b[1].ravel(), b[2]] for b in boxes]))
    removed = (np.clip(field / fieldg.vs + 0.5, 0, 1) - np.clip(cut / fieldg.vs + 0.5, 0, 1)).sum() * fieldg.vs ** 3
    L += [f"POCKETS cut into the joint carve field (k={PARAMS['k']}, grow {PARAMS['grow']}, voxel {PARAMS['vs']} mm), then marching cubes + Taubin 5: {len(ops)} boxes with the "
          f"measured opening rectangles; depth {GROOVE_DEPTH_USER} mm USER-SUPPLIED (owner's caliper, not measured from the videos) from the measured opening plane, along the face normal; "
          "open 3 mm outward so they break through the hull material outside the opening plane.",
          f"  removed {removed:.1f} mm3 (opening area x {GROOVE_DEPTH_USER} mm = {sum(o['L'] * o['S'] for o in ops) * GROOVE_DEPTH_USER:.1f} mm3; the difference is hull material outside the opening plane and smoothing)",
          f"  B_grooves.stl: watertight {c1['watertight']} (boundary {c1['boundary_edges']}, non-manifold {c1['non_manifold_edges']}), volume {c1['volume_mm3']:.0f} mm3 (uncut {c0['volume_mm3']:.0f}), {c1['triangles']} triangles"]
    res["pockets"] = {"depth_user_mm": GROOVE_DEPTH_USER, "removed_mm3": float(removed), "census": c1, "census_uncut": c0}
    names = {n for o in ops for r in o["by_video"].values() for n in r["views"]}
    for vid, idx in (("A_orbit_03", 2738), ("A_orbit_02", 1128)):
        img, fv = frame_view(vid, idx)
        tag = "used to measure" if f"{vid}_{idx}" in names else "NOT used to measure"
        cv2.imwrite(str(SEND / f"grooves_overlay_{vid[-2:]}_f{idx}.png"), overlay(img, fv, v1, f1, ops, f"{vid} f{idx} ({tag})"))
        L.append(f"overlay grooves_overlay_{vid[-2:]}_f{idx}.png: frame {idx} of {vid} was {tag} the openings (left frame, right model with pockets in that camera pose, cyan = triangulated openings).")
    cv2.imwrite(str(SEND / "B_grooves_3view.png"), carve.render3(v1, f1))
    L.append(f"end {date()}; elapsed {time.time() - t0:.0f} s")
    save_stage("1_grooves", "\n".join(L), res)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"base": stage_base, "1": stage1}[a.stage]()
