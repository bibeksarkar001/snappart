"""v3: owner-supplied geometry (lock bump, flat plate), speed to a 5-minute user run, report card + rebuild. Usage: python backend/spikes/v3_run.py --stage 1..5

Reads the v1/v2 caches; writes out/v3/SEND_TO_CLAUDE/ (summary.txt first, measured numbers only)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import part_b as PB  # noqa: E402
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
    for gv, o in zip(gro, res["1_grooves"]["openings"]):
        gv["views"] = o["views"]
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
    ans = {**PB.answers(), "plate_thickness_mm": None}
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
    use_meas = abs(sep - PB.PLATE_THICKNESS_USER_MM) <= 0.3
    th = sep if use_meas else PB.PLATE_THICKNESS_USER_MM
    zu = meas["upper (faces +z)"]["z"]
    mask = np.zeros(field.shape[:2], bool)
    X = g.lo[0] + np.arange(field.shape[0]) * g.vs
    Y = g.lo[1] + np.arange(field.shape[1]) * g.vs
    mask[np.ix_((X >= x0) & (X <= x1), (Y >= y0) & (Y <= y1))] = True
    plate = {"normal": [0, 0, 1.0], "d": -zu, "mask": mask, "region_xy": [x0, x1, y0, y1], "anchor": "upper face, measured (video 1 plane sweep)",
             "measured_faces": meas, "measured_separation_mm": sep}
    feats["plate"] = plate
    ans = {**PB.answers(), "plate_thickness_mm": th}
    f_before = Mo.apply(field, g, {k: feats[k] for k in ("grooves", "bump")}, ans)
    vb, fb = Mo.mesh(f_before, g)
    f_after = Mo.apply(field, g, feats, ans)
    va, fa = Mo.mesh(f_after, g)
    ca = carve.edge_census(va, fa)
    carve.write_stl(SEND / "B_plate.stl", va, fa)
    cv2.imwrite(str(SEND / "B_plate_3view.png"), carve.render3(va, fa))
    box = (x0, x1, y0, y1)
    rb = {k: F.face_rms(vb, fb, box, up) for k, up in (("lower", False), ("upper", True))}
    ra = {k: F.face_rms(va, fa, box, up) for k, up in (("lower", False), ("upper", True))}
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
          f"  owner's caliper: {PB.PLATE_THICKNESS_USER_MM} mm. |difference| {abs(sep - PB.PLATE_THICKNESS_USER_MM):.2f} mm "
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


def stage3b(n_calls=40, conc=16):
    """Real SAM throughput: n_calls crops (one per frame, from the v1 crop cache), conc in parallel, 429 -> exponential backoff (1, 2, 4, 8, 16 s + jitter).
    Results go to out/v3/sam_test/ (the v1 SAM cache is not touched)."""
    import asyncio
    import contextlib
    import io
    import random

    import openai
    from sam_coin import segment
    crops = sorted((V2R.CACHE / "crops").glob("*.jpg"))
    pick = [crops[i] for i in range(0, len(crops), max(len(crops) // n_calls, 1))][:n_calls]
    d = V3 / "sam_test"
    d.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(conc)
    stats = []

    async def one(path):
        async with sem:
            rec = {"crop": path.name, "tries": 0, "n429": 0, "ok": False}
            t0 = time.perf_counter()
            for k in range(6):
                rec["tries"] += 1
                try:
                    t1 = time.perf_counter()
                    _, w, h, boxes, masks, _ = await segment(path, "brown plastic buckle", f"v3/sam_test/{path.stem}_raw.txt")
                    rec.update(ok=True, latency_s=time.perf_counter() - t1, objects=len(boxes))
                    break
                except openai.RateLimitError:
                    rec["n429"] += 1
                    await asyncio.sleep(2 ** k + random.random())
                except Exception as e:  # noqa: BLE001
                    rec["error"] = type(e).__name__
                    await asyncio.sleep(2 ** k + random.random())
            rec["total_s"] = time.perf_counter() - t0
            stats.append(rec)

    async def main():
        t = time.perf_counter()
        await asyncio.gather(*[one(p) for p in pick])
        return time.perf_counter() - t
    t0, d0 = time.time(), date()
    with contextlib.redirect_stdout(io.StringIO()):            # segment() prints per call; one redirect around the whole run (not per task)
        wall = asyncio.run(main())
    import numpy as np
    ok = [r for r in stats if r["ok"]]
    lat = np.array([r["latency_s"] for r in ok]) if ok else np.array([np.nan])
    res = {"calls": len(pick), "concurrency": conc, "wall_s": wall, "ok": len(ok), "n429": sum(r["n429"] for r in stats), "retries": sum(r["tries"] - 1 for r in stats),
           "errors": [r.get("error") for r in stats if not r["ok"]], "latency_median_s": float(np.median(lat)), "latency_p90_s": float(np.percentile(lat, 90)),
           "with_objects": sum(1 for r in ok if r.get("objects")), "per_call_wall_s": wall / max(len(pick), 1), "start": d0}
    (V3 / "sam_test" / "stats.json").write_text(json.dumps({"summary": res, "calls": stats}, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def stage3c(per_video=40, long_side=1920, k=None):
    """Browser-sized frames: per video, 40 of the selected frames at 1920 px long side. Marker detection, mat self-calibration, poses, registration,
    carve, held-out IoU and grooves from them, with the cached masks and crops downscaled to match. Compared with the full-resolution run."""
    import cv2
    import numpy as np
    from capture import carve, config as C, frames, holdout as H, mat, matcal, pipeline as P
    from spike_common import ROOT
    t0, d0 = time.time(), date()
    res2 = json.loads((V2R.V1 / "results.json").read_text())["stage2"]["frames"]
    boxes = {(r["video"], str(r["idx"])): r["box"] for r in res2 if r["status"].startswith("ok")}
    det = mat.make_detector()
    tm = {"decode": 0.0, "detect": 0.0}
    scans, scale, chosen = {}, {}, {}
    for n in NAMES:
        sel = json.loads((V2R.CACHE / f"sel_{n}.json").read_text())
        idx = sorted(int(k) for k in sel["frames"])[::2][:per_video]
        chosen[n] = idx
        s = long_side / max(sel["size"])
        scale[n] = s
        cap = cv2.VideoCapture(str(ROOT / "assets" / "private" / "lock_set" / "Video" / f"{n}.mov"))
        want, i, fr_out = set(idx), 0, {}
        t = time.perf_counter()
        while want:
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                fr_out[i] = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                want.discard(i)
            elif not cap.grab():
                break
            i += 1
        tm["decode"] += time.perf_counter() - t
        t = time.perf_counter()
        fd = {}
        for i, fr in fr_out.items():
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ids, cs = mat.detect(det, gray)
            fd[i] = (ids, cs, frames.marker_sharpness(gray, cs) if len(ids) else 0.0)
        tm["detect"] += time.perf_counter() - t
        scans[n] = {"size": (fr_out[idx[0]].shape[1], fr_out[idx[0]].shape[0]), "frames": fd}
    t = time.perf_counter()
    obs = {}
    for n, sc in scans.items():
        cal = frames.calibrate(sc, min_markers=8)
        o = [(f[0], f[1]) for f in sc["frames"].values() if len(f[0]) >= 8]
        ps = []
        for ids, cs in o:
            rv, tv, _, _ = mat.solve_pose(ids, cs, cal["K"], cal["dist"])
            ps.append((rv.ravel(), tv.ravel()))
        obs[n] = dict(obs=o, poses=ps, K=cal["K"], dist=cal["dist"])
    r = matcal.calibrate_joint(obs, prior=C.MAT_PRIOR_PX_PER_MM)
    tm["mat calibration"] = time.perf_counter() - t
    W_s, world_s = r["W"], matcal.world_dict(r["W"])
    W_f, intr_f = matcal.load(V2R.CACHE / "mat_calibrated.json")[:2]
    dW = W_s - W_f
    L = [f"STAGE 3c: browser-sized frames ({per_video} per video, long side {long_side} px; scale " + ", ".join(f"{n[-2:]} {s:.3f}" for n, s in scale.items()) + f"). start {d0}.",
         f"  marker detection on {sum(len(c) for c in chosen.values())} frames: {tm['detect']:.1f} s (decode + downscale from the video, not needed when the browser sends frames: {tm['decode']:.1f} s)",
         f"  mat self-calibration: {tm['mat calibration']:.1f} s; training rms {r['rms_px']:.2f} px (at 1920 px); marker corners vs the full-resolution calibration: "
         f"mean |dz| {np.abs(dW[..., 2]).mean():.3f} mm, max {np.abs(dW[..., 2]).max():.3f}; mean |dxy| {np.linalg.norm(dW[..., :2], axis=2).mean():.3f} mm"]
    views, pose_rms = {}, {}
    t = time.perf_counter()
    for n in NAMES:
        i8 = r["intr"][n]
        K = np.array([[i8[0], 0, i8[2]], [0, i8[1], i8[3]], [0, 0, 1.0]])
        dist = np.r_[i8[4:8], 0]
        f8 = intr_f[n]
        L.append(f"  {n}: fx {K[0, 0]:.1f} vs full x scale {f8[0] * scale[n]:.1f} ({100 * (K[0, 0] / (f8[0] * scale[n]) - 1):+.2f} %), cx {K[0, 2]:.1f} vs {f8[2] * scale[n]:.1f}, cy {K[1, 2]:.1f} vs {f8[3] * scale[n]:.1f}")
        vs_, rm = [], []
        for i in chosen[n]:
            ids, cs, _ = scans[n]["frames"][i]
            if (n, str(i)) not in boxes or len(ids) < C.MIN_MARKERS:
                continue
            sol = mat.solve_pose(ids, cs, K, dist, world=world_s)
            if sol is None:
                continue
            rv, tv, rms, used = sol
            rm.append(rms)
            if rms > C.MAX_RMS_RELAXED_PX:
                continue
            box = np.array(boxes[(n, str(i))], float) * scale[n]
            m = cv2.imread(str(V2R.CACHE / "masks" / f"{n}_{i}.png"), 0)
            ms = (cv2.resize(m, None, fx=scale[n], fy=scale[n], interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
            Kc = K.copy()
            Kc[0, 2] -= box[0]
            Kc[1, 2] -= box[1]
            vs_.append(carve.View(Kc, cv2.Rodrigues(rv)[0], tv.ravel(), carve.mask_sdf(ms), f"{n}_{i}", n))
        views[n] = vs_
        pose_rms[n] = (float(np.median(rm)), len(vs_), len(chosen[n]))
    tm["poses + views"] = time.perf_counter() - t
    L.append("  poses with the 1920-px calibration (median rms px at 1920 / views kept / frames): " + ", ".join(f"{n[-2:]} {a:.2f} / {b} / {c}" for n, (a, b, c) in pose_rms.items()))
    t = time.perf_counter()
    T = P.register_fast(views, NAMES[0], NAMES[2], NAMES[1], refine_vs=C.REG_REFINE_VOXEL_MM)
    tm["registration"] = time.perf_counter() - t
    t = time.perf_counter()
    k = P.k_for(sum(len(x) for x in views.values())) if k is None else k
    _, g, field = P.carve_joint(views, T, NAMES[0], k=k)
    v, fc = P.mesh_of(field, g)
    tm["carve"] = time.perf_counter() - t
    pl = carve.obb_plan(v)
    t = time.perf_counter()
    vv = [(n, x.moved(T[n])) for n in NAMES for x in views[n]]
    imgs = [cv2.resize(cv2.imread(str(V2R.CACHE / "crops" / f"{x.name}.jpg"), 0), None, fx=scale[n], fy=scale[n], interpolation=cv2.INTER_AREA) for n in NAMES for x in views[n]]
    face, ops, _, _ = P.find_grooves_parallel(v, fc, vv, imgs)
    tm["grooves"] = time.perf_counter() - t
    t = time.perf_counter()
    iou, per, _ = H.joint(views, T, k, 0.25, ref=NAMES[0], grow=C.GROW_MM)
    tm["held-out IoU (3 folds, 0.25 mm; evaluation only)"] = time.perf_counter() - t
    L.append(f"  robust k = {k} (k=12 was tuned at 229 views). Attempt 1 of this stage used k=12 with these 114 views: footprint 51.52 x 41.93, height 11.96, held-out IoU 0.9163 (video 1 0.8685).")
    L += [f"  registration {tm['registration']:.1f} s, carve {tm['carve']:.1f} s, grooves {tm['grooves']:.1f} s ({len(views[NAMES[0]]) + len(views[NAMES[1]]) + len(views[NAMES[2]])} views)",
          f"  RESULT: footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f} (caliper 51.6 x 41.9: {pl['length_mm'] - 51.6:+.2f}, {pl['width_mm'] - 41.9:+.2f}), height {pl['z_max'] - pl['z_min']:.2f} "
          f"(11.8: {pl['z_max'] - pl['z_min'] - 11.8:+.2f}); held-out IoU {iou:.4f} at 0.25 mm (full resolution, 229 views: 0.9526 at 0.25 mm) per video "
          + ", ".join(f"{n[-2:]} {x:.4f}" for n, x in per.items()),
          f"  grooves: {len(ops)} found" + (": " + ", ".join(f"{o['L']:.2f} x {o['S']:.2f} ({o['n_views']} views)" for o in ops) if ops else ""),
          "  approximation: the cached masks were made on frames undistorted with the full-resolution calibration; they are downscaled, not re-made "
          "from the 1920-px calibration (the intrinsic differences above bound that effect).",
          f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("3c_browser_frames", "\n".join(L), {"times_s": tm, "scale": scale, "calib_rms": r["rms_px"], "dz_mean": float(np.abs(dW[..., 2]).mean()), "dxy_mean": float(np.linalg.norm(dW[..., :2], axis=2).mean()),
                                                     "pose_rms": pose_rms, "plan": pl, "heldout_iou_0p25": iou, "k": k, "per_video": per, "grooves": [{"L": o["L"], "S": o["S"], "views": o["n_views"]} for o in ops]})
    print("\n".join(L))


def cached_sam_adapter(scale, boxes, sizes):
    """Stand-in for live SAM in timing runs: the v1 cached full-resolution SAM masks (same frame, same concept key), mapped into the new
    downscaled crop. Returns sam(jobs) for pipeline.masks_from_crops; counts lookups."""
    import cv2
    import numpy as np
    sam_dir = V2R.CACHE / "sam"

    def sam(jobs):
        out = {}
        for key, crop, concept, box in jobs:
            base = key.split("__")[0]
            n, i = base.rsplit("_", 1)
            meta = sam_dir / f"{key}.json"
            if not meta.exists() or (n, i) not in boxes:
                out[key] = None
                continue
            bf = boxes[(n, i)]
            res = []
            for j, c in enumerate(json.loads(meta.read_text())["confs"]):
                m = cv2.imread(str(sam_dir / f"{key}_{j}.png"), 0)
                canvas = np.zeros((sizes[n][1], sizes[n][0]), np.uint8)
                canvas[bf[1]:bf[1] + m.shape[0], bf[0]:bf[0] + m.shape[1]] = m
                small = cv2.resize(canvas, None, fx=scale[n], fy=scale[n], interpolation=cv2.INTER_AREA)
                res.append((c, small[box[1]:box[3], box[0]:box[2]].copy()))
            out[key] = res
        return out
    return sam


def stage3(per_video=40, long_side=1920):
    """End-to-end browser path (pipeline.run_frames) on 40 frames per video at 1920 px, SAM from the cache mapped into the new crops;
    SAM time from the measured throughput (stage 3b: 40 calls at 16 parallel)."""
    import cv2
    import numpy as np
    from capture import carve, pipeline as P
    from spike_common import ROOT
    t0, d0 = time.time(), date()
    res2 = json.loads((V2R.V1 / "results.json").read_text())["stage2"]["frames"]
    boxes = {(r["video"], str(r["idx"])): r["box"] for r in res2}
    frames_by, scale, sizes = {}, {}, {}
    tdec = time.perf_counter()
    for n in NAMES:
        sel = json.loads((V2R.CACHE / f"sel_{n}.json").read_text())
        idx = sorted(int(k) for k in sel["frames"])[::2][:per_video]
        s = long_side / max(sel["size"])
        scale[n], sizes[n] = s, sel["size"]
        cap = cv2.VideoCapture(str(ROOT / "assets" / "private" / "lock_set" / "Video" / f"{n}.mov"))
        want, i, fr = set(idx), 0, {}
        while want:
            if i in want:
                ok, img = cap.read()
                if not ok:
                    break
                fr[i] = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                want.discard(i)
            elif not cap.grab():
                break
            i += 1
        frames_by[n] = fr
    tdec = time.perf_counter() - tdec
    logf = V3 / "pipeline_run.log"
    logf.write_text(f"browser-path run start {d0}\n")

    def log(m):
        with logf.open("a") as fh:
            fh.write(m + "\n")
        print(m, flush=True)
    roles = {"ref": NAMES[0], "flat": NAMES[2], "stand": NAMES[1]}
    rep = P.run_frames(frames_by, roles, V3 / "pipeline", cached_sam_adapter(scale, {k: v for k, v in boxes.items()}, sizes), log=log,
                       answers=PB.answers(), concepts=("brown plastic buckle", "brown plastic part", "brown object"))
    st = json.loads((V3 / "sam_test" / "stats.json").read_text())["summary"]
    per_call = st["per_call_wall_s"]
    n_main = sum(len(v) for v in frames_by.values())
    n_fb = rep["sam_calls"] - n_main
    sam_s = n_main * per_call + (max(st["latency_p90_s"], n_fb * per_call) if n_fb else 0)
    sec = dict(rep["seconds"])
    server = sum(v for k, v in sec.items() if k != "SAM")
    total = server + sam_s
    v1, f1 = P.run_frames.mesh
    pl, cen = carve.obb_plan(v1), carve.edge_census(v1, f1)
    feats = rep["features"]
    L = [f"STAGE 3: one user run, browser path (backend/capture/pipeline.run_frames). start {d0}. {per_video} frames per video at {long_side} px long side "
         f"({n_main} frames). Log: out/v3/pipeline_run.log.",
         "wall clock per stage (s):"] + [f"  {k}: {v:.1f}" + ("  (cache adapter here; real SAM below)" if k == "SAM" else "") for k, v in sec.items()] + [
         f"  SAM, measured rate: {rep['sam_calls']} calls ({n_main} main + {n_fb} fallback); stage 3b measured 40 calls at 16 parallel in {st['wall_s']:.1f} s "
         f"({per_call:.2f} s per call, latency median {st['latency_median_s']:.1f} s, p90 {st['latency_p90_s']:.1f} s, {st['n429']} x 429) -> {sam_s:.0f} s "
         f"({n_main} x {per_call:.2f} + fallback round {'max(p90, n x rate)' if n_fb else '0'}); the 120-call rate is extrapolated from the 40-call test",
         f"  TOTAL per user run: {server:.0f} s server stages + {sam_s:.0f} s SAM = {total:.0f} s ({total / 60:.1f} min). Target 5 min: {'REACHED' if total < 300 else 'NOT REACHED'}.",
         f"  not counted: decoding + downscaling the frames from the video here ({tdec:.0f} s); in the app the browser sends the frames.",
         f"  slowest: " + ", ".join(f"{k} {v:.0f} s" for k, v in sorted({**{k: v for k, v in sec.items() if k != 'SAM'}, 'SAM (measured rate)': sam_s}.items(), key=lambda kv: -kv[1])[:3]),
         f"result: footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f} (caliper 51.6 x 41.9), height {pl['z_max'] - pl['z_min']:.2f} (11.8); watertight {cen['watertight']}; "
         f"views {rep['views']}; mat calibration rms {rep['calib_rms_px']:.2f} px",
         f"  grooves {len(feats.get('grooves', []))}: " + ", ".join(f"{g['long_mm']:.2f} x {g['short_mm']:.2f} ({g['views']} views)" for g in feats.get("grooves", [])),
         f"  bump: " + (f"detected, length {feats['bump']['detected_length_mm']:.2f} mm, {feats['bump']['agreeing_views']} agreeing views ({feats['bump']['source_video']})" if feats.get("bump") else "not detected"),
         f"  plate faces: " + (", ".join(f"{k} z={v['z']:.2f} ({v['views']} views)" for k, v in feats["plate"]["faces"].items()) + f"; separation {feats['plate']['measured_separation_mm']:.2f} mm" if feats.get("plate") else "not measured"),
         "Speed attempts: (a) registration 1.0 mm grid (24 s, within 0.125 mm of the 0.5 mm grid); coarse-to-fine shell 113 s, dropped. (b) SAM 16 parallel measured. "
         "(c) 40 frames/video at 1920 px: detection 3.5 s instead of a 77 s 4K scan; k scaled to the view count (k=12 kept the hull fat: IoU 0.916; k=6: 0.936). "
         "(d) grooves: RANSAC on 40k instead of 200k samples (35 -> 8 s, same face, openings within 0.05 mm).",
         f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("3_speed", "\n".join(L), {"seconds": sec, "sam_estimate_s": sam_s, "sam_calls": rep["sam_calls"], "total_s": total, "server_s": server, "decode_s": tdec,
                                         "plan": pl, "watertight": cen["watertight"], "views": rep["views"]})
    log("\n".join(L))


def stage4(reuse_iou=True):
    """report.json for part B from the saved model (v1 views + registration), owner answers, folds held-out IoU plain vs with all features,
    and a rebuild() check against B_plate.stl."""
    import numpy as np
    from capture import carve, config as C, dataset, holdout as H, model as Mo, pipeline as P, report as Rp
    t0, d0 = time.time(), date()
    m = Mo.load(V3 / "model_B")
    field, g, feats = m["field"], m["grid"], m["features"]
    answers = PB.answers()
    user_given = list(answers)                                    # every value came from the owner's calipers
    v0, f0 = Mo.mesh(field, g)
    v1, f1 = Mo.rebuild(m, answers)
    vp, fp = V2R.load_mesh(SEND / "B_plate.stl")
    same = len(v1) == len(vp) and float(np.abs(np.sort(v1.round(4), axis=0) - np.sort(vp, axis=0)).max()) < 1e-3
    views = {n: dataset.load_views(n) for n in NAMES}
    T = V2R.load_T()
    vm = [x.moved(T[n]) for n in NAMES for x in views[n]]
    # held-out IoU: same 3 folds as holdout.joint, plain carve vs carve + all features (answers)
    allv = [(n, i) for n in NAMES for i in range(len(views[n]))]
    plain, full = [], []
    prev = json.loads((SEND / "results.json").read_text()).get("4_report") if reuse_iou else None
    if prev:                                                     # measured in the first stage 4 run (366 s); not recomputed
        plain, full = [prev["held_out_plain"]], [prev["held_out_features"]]
    for fo in range(0 if prev else 3):
        hold = set()
        for n in NAMES:
            share = max(round(10 * len(views[n]) / len(allv)), 1)
            step = max(len(views[n]) // share, 1)
            hold |= {(n, i) for i in list(range(fo % step, len(views[n]), step))[:share]}
        train = {n: [x for i, x in enumerate(views[n]) if (n, i) not in hold] for n in NAMES}
        _, gg, ff = P.carve_joint(train, T, NAMES[0], k=C.K_ROBUST)
        feats_f = dict(feats)
        if feats.get("plate"):
            p = dict(feats["plate"])
            X = gg.lo[0] + np.arange(ff.shape[0]) * gg.vs
            Y = gg.lo[1] + np.arange(ff.shape[1]) * gg.vs
            x0, x1, y0, y1 = p["region_xy"]
            p["mask"] = np.zeros(ff.shape[:2], bool)
            p["mask"][np.ix_((X >= x0) & (X <= x1), (Y >= y0) & (Y <= y1))] = True
            feats_f["plate"] = p
        ma, mb = Mo.mesh(ff, gg), Mo.mesh(Mo.apply(ff, gg, feats_f, answers), gg)
        for n, i in sorted(hold):
            mv = views[n][i].moved(T[n])
            plain.append(carve.view_iou(*ma, mv))
            full.append(carve.view_iou(*mb, mv))
    v2res = json.loads((V2R.SEND / "results.json").read_text())
    dz = [o["by_video"]["A_orbit_02"]["centre"][2] - o["by_video"]["A_orbit_03"]["centre"][2] for o in v2res["1_grooves"]["openings"] if len(o["by_video"]) > 1]
    st3 = json.loads((SEND / "results.json").read_text()).get("3_speed", {})
    s1 = json.loads((SEND / "results.json").read_text())["1_bump"]
    warns = [f"video 2 vs video 3: the same four slot openings triangulate {np.mean(dz):+.2f} mm apart in height (part frame z) between the two videos "
             f"(range {min(dz):+.2f}..{max(dz):+.2f}); positions use the video with more views (video 3).",
             f"lock bump: the block is user-supplied in size; {100 * s1['outside_hull_mm3'] / s1['block_volume_mm3']:.0f} % of it lies outside the carved hull, because the hull fills the recess around it.",
             "lock bump position: the owner says it starts at the top end of the slot column; the detection puts it in the middle (~8 mm free at both ends)."]
    rep = Rp.build((v0, f0), feats, {**answers}, user_given, caliper={"length": 51.6, "width": 41.9, "height": 11.8}, held_out_iou=float(np.mean(plain)),
                   views_moved=vm, timings={"per_stage_s": st3.get("seconds"), "sam_s_measured_rate": st3.get("sam_estimate_s"), "total_s": st3.get("total_s")}, warnings=warns)
    rep["held_out_iou_with_features"] = float(np.mean(full))
    rep["rebuild_reproduces_B_plate"] = bool(same)
    Rp.write(SEND / "report.json", rep)
    L = [f"STAGE 4: report card. start {d0}. backend/capture/report.py -> out/v3/SEND_TO_CLAUDE/report.json (part B, owner answers).",
         f"  held-out IoU (3 folds x 10 views, 0.15 mm, k=12): plain carve {np.mean(plain):.4f}; with grooves + bump + plate (answers) {np.mean(full):.4f} ({np.mean(full) - np.mean(plain):+.4f})",
         "  dims (plain carve vs caliper): " + ", ".join(f"{k} {d['measured']} vs {d['caliper']} ({d['error']:+.2f})" for k, d in rep["dims_mm"].items()),
         "  features: " + "; ".join(f"{f['id']}={f['value']} [{f['status']}{', ' + str(f['views']) + ' views' if f['views'] else ''}]" for f in rep["features"]),
         f"  questions ({len(rep['questions'])}): " + "; ".join(f"{q['id']} (default {q['default']} {q['unit']})" for q in rep["questions"]),
         f"  low-confidence surface (< 3 views): {100 * rep['low_confidence_regions']['area_share']:.1f} % of the area; largest cells " + ", ".join(f"{r['centre_mm']} {r['area_mm2']} mm2" for r in rep["low_confidence_regions"]["regions"][:4]),
         "  warnings: " + " | ".join(rep["warnings"]),
         f"  rebuild(model_B, answers) reproduces B_plate.stl (same vertices within 1e-3 mm): {same}",
         f"end {date()}; elapsed {time.time() - t0:.0f} s"]
    save_stage("4_report", "\n".join(L), {"held_out_plain": float(np.mean(plain)), "held_out_features": float(np.mean(full)), "rebuild_same": bool(same)})
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"1": stage1, "2": stage2, "3b": stage3b, "3c": stage3c, "3": stage3, "4": stage4}[a.stage]()
