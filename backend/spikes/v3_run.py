"""v3: owner-supplied geometry (lock bump, flat plate), speed to a 5-minute user run, report card + rebuild. Usage: python backend/spikes/v3_run.py --stage 1..5

Reads the v1/v2 caches; writes out/v3/SEND_TO_CLAUDE/ (summary.txt first, measured numbers only)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import v2_run as V2R  # noqa: E402
from spike_common import OUT  # noqa: E402

V3 = OUT / "v3"
SEND = V3 / "SEND_TO_CLAUDE"
SEND.mkdir(parents=True, exist_ok=True)
NAMES = V2R.NAMES
date = V2R.date


def save_stage(key, text, results):
    (SEND / f"stage_{key}.txt").write_text(text)
    head = SEND / "header.txt"
    parts = ([head.read_text()] if head.exists() else []) + [p.read_text() for p in sorted(SEND.glob("stage_*.txt"))]
    (SEND / "summary.txt").write_text("\n\n".join(parts))
    f = SEND / "results.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[key] = results
    f.write_text(json.dumps(cur, indent=1, default=float))


def base_features():
    """Grooves (v2 stage 1, measured) and the bump's detected edge (v2 stage 2) in the part frame."""
    import numpy as np
    rows = np.load(V2R.V2 / "grooves.npy")
    gro = []
    for r in rows:
        axes = r[3:12].reshape(3, 3)
        half = r[12:15]
        depth_used = 2 * half[2] - 3.0
        gro.append({"centre": (r[:3] - axes[2] * (3.0 - depth_used) / 2).tolist(), "axes": axes.tolist(), "long_mm": 2 * half[0], "short_mm": 2 * half[1]})
    res = json.loads((V2R.SEND / "results.json").read_text())
    b = res["2_bump"]["per_video"][res["2_bump"]["primary"]]
    e0, e1 = np.array(b["end0"]), np.array(b["end1"])
    n = np.array(gro[0]["axes"][2])
    a = e1 - e0
    a -= n * (a @ n)
    a /= np.linalg.norm(a)
    mid = (e0 + e1) / 2
    oc = np.mean([g["centre"] for g in gro], 0)
    w = oc - mid
    w -= a * (w @ a) + n * (w @ n)
    w /= np.linalg.norm(w)
    bump = {"edge_mid": mid.tolist(), "axis": a.tolist(), "toward_openings": w.tolist(), "normal": n.tolist(), "detected_length_mm": float(np.linalg.norm(e1 - e0)),
            "ends": [e0.tolist(), e1.tolist()], "source_video": res["2_bump"]["primary"], "agreeing_views": b["agree"]}
    return {"grooves": gro, "bump": bump}


def base_field():
    """Joint carve field at the frozen settings, v1 cached views + registration."""
    from capture import dataset, pipeline as P
    views = {n: dataset.load_views(n) for n in NAMES}
    T = V2R.load_T()
    _, g, field = P.carve_joint(views, T, NAMES[0])
    return views, T, g, field


def draw_box(img, view, c, axes, half, col):
    import cv2
    import numpy as np
    from capture import carve
    sg = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    P = c + (sg * half) @ axes
    u, v, _ = carve.project(view, P)
    for i in range(8):
        for j in range(i + 1, 8):
            if (sg[i] != sg[j]).sum() == 1:
                cv2.line(img, (int(u[i] * 16), int(v[i] * 16)), (int(u[j] * 16), int(v[j] * 16)), col, 2, cv2.LINE_AA, 4)


def stage1():
    import cv2
    import numpy as np
    from capture import carve, config as C, features as F, model as Mo
    t0, d0 = time.time(), date()
    feats = base_features()
    views, T, g, field = base_field()
    ans = {**Mo.default_answers(), "plate_thickness_mm": None}
    f_g = Mo.apply(field, g, {"grooves": feats["grooves"]}, ans)
    f_b = Mo.apply(field, g, feats, ans)
    v0, fc0 = Mo.mesh(f_g, g)
    v1, fc1 = Mo.mesh(f_b, g)
    c0, c1 = carve.edge_census(v0, fc0), carve.edge_census(v1, fc1)
    carve.write_stl(SEND / "B_bump.stl", v1, fc1)
    cv2.imwrite(str(SEND / "B_bump_3view.png"), carve.render3(v1, fc1))
    b = feats["bump"]
    box = Mo.bump_op(b, ans["bump_length_mm"], ans["bump_width_mm"], ans["bump_height_mm"])
    X = g.points()
    inside_hull = (field.ravel() > 0)
    bd = F.box_sdf(X, *box) < 0
    added = float((bd & ~inside_hull).sum() * g.vs ** 3)
    blockv = float(bd.sum() * g.vs ** 3)
    n = np.array(b["normal"])
    zs = [np.array(b["edge_mid"]) @ n, (np.array(b["edge_mid"]) + n * ans["bump_height_mm"]) @ n]
    gro_c = np.array([gg["centre"] for gg in feats["grooves"]])
    span = gro_c @ np.array(b["axis"])
    e = np.array(b["ends"]) @ np.array(b["axis"])
    L = [f"STAGE 1: lock bump geometry. start {d0}. Baseline = v2 B_grooves (joint carve, frozen settings, 4 measured groove pockets, depth {ans['groove_depth_mm']} user-supplied).",
         f"POSITION (detected, v2 stage 2, {b['source_video']}, {b['agreeing_views']} agreeing views): far edge of the raised block, triangulated ends "
         f"({b['ends'][0][0]:.2f},{b['ends'][0][1]:.2f},{b['ends'][0][2]:.2f}) .. ({b['ends'][1][0]:.2f},{b['ends'][1][1]:.2f},{b['ends'][1][2]:.2f}), detected length {b['detected_length_mm']:.2f} mm.",
         f"DIMENSIONS (user-supplied, config.py): length {ans['bump_length_mm']} mm (centred on the detected ends' midpoint), width {ans['bump_width_mm']} mm (from the detected far edge "
         f"toward the openings), height {ans['bump_height_mm']} mm above the surface the grooves open in (the detected edge's plane), plus 1 mm embedded so it fuses.",
         f"  block volume {blockv:.1f} mm3, of which only {added:.1f} mm3 lies outside the carved hull: the hull already fills the space above the groove face "
         f"(the face is {-(gro_c @ n).mean():.2f} mm inside the hull's surface per video 3), so the block sticks out of the hull by only "
         f"{max(0.0, -((np.array(b['edge_mid']) + n * ans['bump_height_mm']) @ np.array([0, 0, 1.0]))):.2f} mm (its top reaches z={(np.array(b['edge_mid']) + n * ans['bump_height_mm'])[2]:.2f} in the part frame, the video 1 floor is z=0).",
         f"  where along the grooved edge: openings span {span.min() - 3.7:.1f}..{span.max() + 3.7:.1f} mm along the edge axis; the block spans {e.min():.1f}..{e.max():.1f} "
         f"(gaps {e.min() - (span.min() - 3.7):.1f} mm at one end, {(span.max() + 3.7) - e.max():.1f} mm at the other). The owner says it starts at the top end with the bottom part empty; "
         "the detection puts it in the middle with ~8 mm free at both ends. Not resolved; the detected position is used.",
         f"  B_bump.stl: watertight {c1['watertight']} (boundary {c1['boundary_edges']}, non-manifold {c1['non_manifold_edges']}), volume {c1['volume_mm3']:.0f} mm3 (grooves only {c0['volume_mm3']:.0f}), {c1['triangles']} triangles"]
    detect = {}
    for vid, idx in (("A_orbit_03", 1010), ("A_orbit_03", 2738), ("A_orbit_02", 1204)):
        img, fv = V2R.frame_view(vid, idx)
        crop_v = next(x for x in views[vid] if x.name == f"{vid}_{idx}").moved(T[vid])
        gray = cv2.imread(str(V2R.CACHE / "crops" / f"{vid}_{idx}.jpg"), 0)
        face, dl, _ = V2R.groove_face(*V2R.load_mesh(V2R.V1 / "A_final.stl"))
        gg = F.rectify(crop_v, gray, face, dl, 20)
        oo = F.openings(gg, 20)
        used = False
        if len(oo) >= 3:
            top = float(np.median([o[1] + o[3] / 2 for o in oo]))
            r = F.step_band(gg, 20, top + 0.3, top + 3.2, min(o[0] - o[2] / 2 for o in oo), max(o[0] + o[2] / 2 for o in oo), w_lo=top - 0.3, w_hi=top + 3.2)
            used = r is not None and vid == b["source_video"] and abs((r["s1"] - r["s0"]) - 19.45) <= 1.0
        tag = "USED to detect the position" if used else "NOT used to detect the position"
        detect[f"{vid}_{idx}"] = used
        o_img = V2R.overlay(img, fv, v1, fc1, [], f"{vid} f{idx} ({tag})")
        sh, mk = F.render_shaded(v1, fc1, fv)
        full = img.copy()
        full[mk > 0] = (0.45 * img[mk > 0] + 0.55 * sh[mk > 0]).astype(np.uint8)
        draw_box(full, fv, *box, (255, 0, 255))
        ys, xs = np.nonzero(mk)
        y0, y1, x0, x1 = max(ys.min() - 150, 0), ys.max() + 150, max(xs.min() - 150, 0), xs.max() + 150
        pair = np.hstack([img[y0:y1, x0:x1], full[y0:y1, x0:x1]])
        cv2.putText(pair, f"{vid} f{idx} ({tag}): frame | model + bump block (magenta)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.imwrite(str(SEND / f"bump_overlay_{vid[-2:]}_f{idx}.png"), pair)
    L.append("overlays (frame | model with the bump block outlined in magenta): " + ", ".join(f"bump_overlay_{k.split('_')[2]}_f{k.split('_')[3]}.png ({'used' if u else 'not used'} for the position)" for k, u in detect.items()))
    L.append("visual check (by eye, these three images): f1010 (used): the block outline covers the real raised block between its two steps. f2738 (not used): "
             "the left end and the band above openings 2-3 line up; the real block's right step looks up to ~3-4 mm further right than the outline, not resolved at this "
             "viewing angle. f1204 (video 2, not used): the outline stands beside the opening column where the real rail is, from opening 1 to opening 3, the bottom "
             "opening clear, as the owner describes; the whole model is offset there by the video 2 registration (~1.6 mm, v2 stage 1).")
    feats_out = {**feats}
    Mo.save(V3 / "model_B", field, g, feats_out)
    L.append(f"model saved for rebuild(): out/v3/model_B.npz/.json (carve field + features). end {date()}; elapsed {time.time() - t0:.0f} s")
    save_stage("1_bump", "\n".join(L), {"block_volume_mm3": blockv, "outside_hull_mm3": added, "census": c1, "census_grooves_only": c0, "used_for_detection": detect,
                                        "bump": {**b, "length_user": ans["bump_length_mm"], "width_user": ans["bump_width_mm"], "height_user": ans["bump_height_mm"]}})
    print("\n".join(L))


def plate_region(field, g, bump):
    """Plate footprint from the carve: x from the window's far edge (first material along y=101 at mid height) to the bump's detected far edge;
    y between the rails' inner faces (gap along x in the window, mid height)."""
    import numpy as np
    ix = lambda v, a: int(round((v - g.lo[a]) / g.vs))
    zc = ix(6.0, 2)
    row = field[:, ix(101, 1), zc] > 0
    xs = g.lo[0] + np.where(row)[0] * g.vs
    xw = xs[(xs > 140)].min()                                   # window far edge (material after the window)
    col = field[ix(xw - 5, 0), :, zc] > 0
    ys = g.lo[1] + np.where(col)[0] * g.vs
    y0, y1 = ys[ys < 101].max() + g.vs, ys[ys > 101].min() - g.vs  # rails' inner faces
    x1 = min(e[0] for e in bump["ends"])
    return float(xw), float(x1), float(y0), float(y1)


def face_rms(v, fc, box, up):
    """RMS distance (mm) of the mesh's surface samples in box (x0, x1, y0, y1), facing up (+z) or down, to their least-squares plane."""
    import numpy as np
    from capture import snap
    P, N = snap.oriented_points(v, fc, 200000)
    m = (P[:, 0] > box[0] + 0.5) & (P[:, 0] < box[1] - 0.5) & (P[:, 1] > box[2] + 0.5) & (P[:, 1] < box[3] - 0.5) & ((N[:, 2] > 0.5) if up else (N[:, 2] < -0.5))
    q = P[m]
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][2]
    r = (q - c) @ n
    return float(np.sqrt((r ** 2).mean())), int(m.sum()), float(c[2]), float(np.degrees(np.arccos(abs(n[2]))))


def stage2():
    import cv2
    import numpy as np
    from capture import carve, config as C, features as F, model as Mo, snap
    t0, d0 = time.time(), date()
    m = Mo.load(V3 / "model_B")
    field, g, feats = m["field"], m["grid"], m["features"]
    x0, x1, y0, y1 = plate_region(field, g, feats["bump"])
    vv, imgs = V2R.all_views()

    def face_z(z, up):
        n = np.array([0, 0, 1.0 if up else -1.0])
        return {"O": np.array([x0 + 1, y0 + 1.5, z]), "e1": np.array([0, 1.0, 0]), "e2": np.array([1.0, 0, 0]), "n": n, "len_s": y1 - y0 - 3, "len_t": x1 - x0 - 3}
    zs = np.arange(6.0, 12.01, 0.1)
    meas = {}
    for key, vid, up in (("lower (faces -z)", "A_orbit_03", False), ("upper (faces +z)", "A_orbit_01", True)):
        sel = [(v, im) for (n, v), im in zip(vv, imgs) if n == vid and F.facing(v, face_z(10, up)) > 0.6]
        z, sd, sc = F.plane_sweep([v for v, _ in sel], [im for _, im in sel], lambda z: face_z(z, up), zs)
        meas[key] = {"z": z, "boot_sd": sd, "views": len(sel), "video": vid, "contrast": float(sc.max() / sc.min())}
    sep = meas["upper (faces +z)"]["z"] - meas["lower (faces -z)"]["z"]
    use_meas = abs(sep - C.PLATE_THICKNESS_USER_MM) <= 0.3
    th = sep if use_meas else C.PLATE_THICKNESS_USER_MM
    zu = meas["upper (faces +z)"]["z"]
    mask = np.zeros(field.shape[:2], bool)
    X = g.lo[0] + np.arange(field.shape[0]) * g.vs
    Y = g.lo[1] + np.arange(field.shape[1]) * g.vs
    mask[np.ix_((X >= x0) & (X <= x1), (Y >= y0) & (Y <= y1))] = True
    plate = {"normal": [0, 0, 1.0], "d": -zu, "mask": mask, "region_xy": [x0, x1, y0, y1], "anchor": "upper face, measured (video 1 plane sweep)",
             "measured_faces": meas, "measured_separation_mm": sep}
    feats["plate"] = plate
    ans = {**Mo.default_answers(), "plate_thickness_mm": th}
    f_before = Mo.apply(field, g, {k: feats[k] for k in ("grooves", "bump")}, ans)
    vb, fb = Mo.mesh(f_before, g)
    f_after = Mo.apply(field, g, feats, ans)
    va, fa = Mo.mesh(f_after, g)
    ca = carve.edge_census(va, fa)
    carve.write_stl(SEND / "B_plate.stl", va, fa)
    cv2.imwrite(str(SEND / "B_plate_3view.png"), carve.render3(va, fa))
    box = (x0, x1, y0, y1)
    rb = {k: face_rms(vb, fb, box, up) for k, up in (("lower", False), ("upper", True))}
    ra = {k: face_rms(va, fa, box, up) for k, up in (("lower", False), ("upper", True))}
    # cross-section at y = 101 through the plate
    iy = int(round((101 - g.lo[1]) / g.vs))
    runs = {}
    for name, ff in (("before", f_before), ("after", f_after)):
        out = []
        for x in np.arange(x0 + 1, x1, 2.0):
            colz = ff[int(round((x - g.lo[0]) / g.vs)), iy] > 0
            zz = g.lo[2] + np.where(colz)[0] * g.vs
            out.append((round(float(x), 1), round(float(zz.min()), 2) if len(zz) else None, round(float(zz.max()), 2) if len(zz) else None))
        runs[name] = out
    sec = ((f_after[:, iy, :].T[::-1] > 0) * 255).astype(np.uint8)
    secb = ((f_before[:, iy, :].T[::-1] > 0) * 255).astype(np.uint8)
    pic = cv2.resize(np.vstack([secb, np.full((4, sec.shape[1]), 128, np.uint8), sec]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(SEND / "plate_section_y101.png"), pic)
    Mo.save(V3 / "model_B", field, g, feats)
    L = [f"STAGE 2: thin flat plate. start {d0}.",
         f"region (from the carve): x {x0:.2f}..{x1:.2f} (window far edge .. bump's detected far edge), y {y0:.2f}..{y1:.2f} (rails' inner faces).",
         "RANSAC on the mesh cannot find the plate's two faces: the carve there is a wedge (bottom rising at ~33 deg from z 7.4 at the window to z 0.15 near the "
         "groove column; top = video 3's floor plane). Silhouettes cannot carve under a plate that sits between two rails. So the faces are measured by "
         "photo-consistency plane sweeps instead (horizontal planes z = 6..12 mm step 0.1; views facing the face, cos > 0.6; 20 bootstrap resamples of the views):"]
    for k, r in meas.items():
        L.append(f"  {k} face: z = {r['z']:.2f} mm (bootstrap sd {r['boot_sd']:.2f}), {r['views']} views of {r['video']}, score contrast {r['contrast']:.2f}")
    L += [f"  MEASURED SEPARATION {sep:.2f} mm (the two faces come from different videos; their vertical registration is consistent to ~0.1 mm: the carved height "
          f"between the two videos' floors is 11.69 vs 11.8 caliper).",
          "  region sensitivity: an exploratory sweep over x 151..162.5, y 86..117 (same method) gave faces at 9.70 / 10.20 = 0.50 mm. So the measured "
          "separation is 0.2-0.5 mm depending on the region; the bootstrap sd (<= 0.07) understates that.",
          f"  owner's caliper: {C.PLATE_THICKNESS_USER_MM} mm. |difference| {abs(sep - C.PLATE_THICKNESS_USER_MM):.2f} mm "
          + ("<= 0.3: the MEASURED value is used." if use_meas else "> 0.3: per the rule, 2.6 mm is used and labelled USER-SUPPLIED. THE MEASUREMENT CONTRADICTS IT: the "
             "owner's caliper may have been taken at the plate's edge (the carve shows ~3.1 mm of material at the window edge), not in the middle; this needs the owner."),
          f"  model: plate = slab between the measured upper face z={zu:.2f} and z={zu - th:.2f} (thickness {th:.2f} mm, {'measured' if use_meas else 'user-supplied'}); "
          "anchored on the upper face because anchoring on the lower one would put the slab above video 3's floor plane (impossible).",
          "roughness (RMS distance of the surface samples in the plate region, inset 0.5 mm, to their own least-squares plane; samples / mean z / tilt):"]
    for k in ("lower", "upper"):
        L.append(f"  {k} surface: before {rb[k][0]:.3f} mm ({rb[k][1]} samples, z {rb[k][2]:.2f}, tilt {rb[k][3]:.1f} deg)  after {ra[k][0]:.3f} mm ({ra[k][1]} samples, z {ra[k][2]:.2f}, tilt {ra[k][3]:.1f} deg)")
    L += ["  ('before' lower = the hull wedge, not a plate face; 'before' upper = the hull's top there (video 3's floor plane plus the up-facing facets in the region). "
          "'after' is 0 by construction: the slab faces are exact planes, so this shows the snap worked, not that the plate is flat.)",
          f"B_plate.stl (grooves + bump + plate): watertight {ca['watertight']} (boundary {ca['boundary_edges']}, non-manifold {ca['non_manifold_edges']}), volume {ca['volume_mm3']:.0f} mm3, {ca['triangles']} triangles",
          "cross-section y=101 (x: z_min..z_max of material), before: " + ", ".join(f"{x}:{a}..{b}" for x, a, b in runs["before"]),
          "  after: " + ", ".join(f"{x}:{a}..{b}" for x, a, b in runs["after"]),
          "  picture plate_section_y101.png (top: before, bottom: after; x to the right, z up)",
          f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("2_plate", "\n".join(L), {"region": [x0, x1, y0, y1], "faces": meas, "separation_mm": sep, "used_mm": th, "used_source": "measured" if use_meas else "user-supplied",
                                         "roughness_before": rb, "roughness_after": ra, "census": ca, "section": runs})
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"1": stage1, "2": stage2}[a.stage]()
