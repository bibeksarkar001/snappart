"""Geometry fidelity pass 1 (cached masks, no API calls): height profiles from the edge photos, stepped solids, watertight STLs.

Edge photos = the part stood on its long edge and photographed from above; the silhouette's extent across the long axis is the
height profile. Plan outline from acc_coin_top_front_01. Writes out/fidelity/ (STLs) and out/fidelity/SEND_TO_CLAUDE/.
Usage: python backend/spikes/fidelity_test.py   (run solid_check.py separately for the 10-part hole/watertight check)
"""
import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cadquery as cq
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accuracy_set_test as acc  # noqa: E402
from cad.extrude import export_stl, extrude_part  # noqa: E402
from cad.section import mesh_edges, read_stl  # noqa: E402
from measure.outline import Params, extract_outline, sample_primitives  # noqa: E402
from measure.profile import RES, height_profile  # noqa: E402
from outline_run import load  # noqa: E402
from spike_common import OUT  # noqa: E402

DEST = OUT / "fidelity"
SEND = DEST / "SEND_TO_CLAUDE"
SEND.mkdir(parents=True, exist_ok=True)
ACC_EDGES = ["acc_coin_edge_01", "acc_coin_edge_02", "acc_coin_edge_03", "acc_card_edge_01"]
LOCK_EDGES = ["lock_edge_01", "lock_edge_02", "lock_edge_03", "lock_edge_04"]
B_HEIGHTS = (14.4, 11.8)            # caliper values for B
STAND_W = {"B": 41.9, "A": (34.7 + 34.0) / 2}   # plan dimension that becomes the standing height (assumption: the plan's short side)
SOLID_PLAN = "acc_coin_top_front_01"
SOLID_EDGE = "acc_coin_edge_01"


def long_axis_mm(res, prims):
    """Unit long-axis direction in the mm frame, pointing to image +x (so profile and plan share a direction)."""
    pts = sample_primitives(prims, 0.05)
    px = res.frame.to_px(pts)
    (cx, cy), (w, h), ang = cv2.minAreaRect(px.astype(np.float32))
    a = math.radians(ang if w >= h else ang + 90)
    u_px = np.array([math.cos(a), math.sin(a)])
    if u_px[0] < 0:
        u_px = -u_px
    c = np.array([cx, cy])
    d = (res.frame.to_mm(c + 50 * u_px) - res.frame.to_mm(c - 50 * u_px)).ravel()
    return d / np.linalg.norm(d)


def profile_row(photo, label, res, part, rec, plan_long, ref_note):
    u = long_axis_mm(res, part.primitives)
    prof = height_profile(part.primitives, u)
    f = plan_long / prof["length_mm"] if plan_long else None
    row = {"photo": photo, "blob": label, "ref": ref_note, "length_mm": prof["length_mm"], "max_height_mm": prof["max_height_mm"],
           "x_at_max_mm": prof["x_at_max_mm"], "p95_height_mm": float(np.percentile(prof["height_mm"][prof["height_mm"] > 0], 95)),
           "steps": prof["steps"], "barrels": prof["barrels"], "plan_long_mm": plan_long, "length_match_factor": f,
           "max_height_length_matched_mm": prof["max_height_mm"] * f if f else None}
    if rec and "distance_mm" in rec and label in ("p1", "p2"):
        part_lab = "B" if label == "p1" else "A"
        d = rec["distance_mm"]
        k = d / (d - STAND_W[part_lab])
        row.update(distance_mm=d, parallax_k_assumed=k, standing_height_assumed_mm=STAND_W[part_lab], max_height_parallax_model_mm=prof["max_height_mm"] / k)
    return row, prof, u


def draw_profile(img, res, parts, profs, rows, path):
    out = img.copy()
    t = 4
    pts_all = []
    for part, prof, col in zip(parts, profs, [(0, 0, 255), (0, 160, 0)]):
        px = np.round(res.frame.to_px(sample_primitives(part.primitives, 0.05))).astype(np.int32)
        pts_all.append(px)
        cv2.polylines(out, [px.reshape(-1, 1, 2)], True, col, t, cv2.LINE_AA)
        Ri, shift = prof["rotation"], prof["origin_shift"]
        for s in prof["steps"]:
            for xs in (s["x0_mm"], s["x1_mm"]):
                a = np.array([xs, prof["base_mm"][min(int(xs / RES), len(prof["base_mm"]) - 1)]])
                b = np.array([xs, prof["top_mm"][min(int(xs / RES), len(prof["top_mm"]) - 1)]])
                seg = np.array([((q + shift) @ Ri) for q in (a, b)])     # back to the mm frame (R is orthonormal)
                p2 = np.round(res.frame.to_px(seg)).astype(np.int32)
                cv2.line(out, tuple(p2[0]), tuple(p2[1]), (255, 160, 0), 2, cv2.LINE_AA)
            c = np.round(res.frame.to_px(((np.array([(s["x0_mm"] + s["x1_mm"]) / 2, prof["top_mm"][min(int((s["x0_mm"] + s["x1_mm"]) / 2 / RES), len(prof["top_mm"]) - 1)] + 1.0]) + shift) @ Ri)[None])).astype(int)[0]
            cv2.putText(out, f"{s['height_mm']:.1f}", (c[0] - 40, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(out, f"{s['height_mm']:.1f}", (c[0] - 40, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    allp = np.vstack(pts_all)
    x0, y0 = np.maximum(allp.min(0) - 150, 0)
    x1, y1 = np.minimum(allp.max(0) + 150, [out.shape[1], out.shape[0]])
    crop = out[y0:y1, x0:x1]
    s = min(1600 / crop.shape[1], 1.0)
    cv2.imwrite(str(path), cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))


def rotate_prims(prims, R, shift):
    f = lambda p: (np.asarray(p, float) @ R.T - shift).tolist()
    out = []
    for p in prims:
        q = dict(p)
        for k in ("p0", "p1", "c"):
            if k in q:
                q[k] = f(q[k])
        out.append(q)
    return out


def stepped_solid(part, res_top, prof, f, path):
    """Plan outline (rotated so the long axis is x, starting at 0) extruded, then cut down to the height profile steps (scaled by f)."""
    u = long_axis_mm(res_top, part.primitives)
    R = np.array([[u[0], u[1]], [-u[1], u[0]]])
    pts = sample_primitives(part.primitives, 0.05)
    q = pts @ R.T
    shift = q.min(0)
    rp = SimpleNamespace(primitives=rotate_prims(part.primitives, R, shift),
                         holes=[SimpleNamespace(primitives=rotate_prims(h.primitives, R, shift)) for h in part.holes])
    steps = prof["steps"]
    hmax = max(s["height_mm"] for s in steps) * f
    solid = extrude_part(rp, hmax + 1.0)
    L = float((q - shift)[:, 0].max())
    for i, s in enumerate(steps):
        x0 = -1.0 if i == 0 else steps[i]["x0_mm"] * f
        x1 = L + 1.0 if i == len(steps) - 1 else steps[i + 1]["x0_mm"] * f
        h = s["height_mm"] * f
        box = cq.Workplane("XY").box(x1 - x0, 1000, hmax + 2, centered=(False, True, False)).translate((x0, 0, h))
        solid = solid.cut(box)
    export_stl(solid, path)
    return mesh_edges(path), [{"x0_mm": (-0.0 if i == 0 else s["x0_mm"] * f), "x1_mm": (steps[i + 1]["x0_mm"] * f if i < len(steps) - 1 else L), "height_mm": s["height_mm"] * f}
                              for i, s in enumerate(steps)]


def render_views(stl, path, size=520):
    tris = read_stl(stl)
    mn, mx = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    ext = mx - mn
    s = (size - 60) / max(ext)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, -0.45, 0.82])
    panels = []
    for name, (ax, ay, az, sign) in {"top (x, y)": (0, 1, 2, 1), "front (x, z)": (0, 2, 1, -1), "side (y, z)": (1, 2, 0, 1)}.items():
        img = np.full((size, size, 3), 245, np.uint8)
        depth = tris[:, :, az].mean(1) * sign
        for i in np.argsort(depth):
            if n[i, az] * sign < 0 and name != "top (x, y)":
                pass
            p = tris[i][:, [ax, ay]]
            px = np.c_[30 + (p[:, 0] - mn[ax]) * s, size - 30 - (p[:, 1] - mn[ay]) * s]
            shade = int(60 + 170 * abs(float(n[i] @ light)))
            cv2.fillPoly(img, [px.astype(np.int32)], (shade * 0.75, shade * 0.8, shade))
            cv2.polylines(img, [px.astype(np.int32)], True, (shade * 0.75 - 20, shade * 0.8 - 20, shade - 20), 1, cv2.LINE_AA)
        cv2.putText(img, f"{name}  {ext[ax]:.1f} x {ext[ay]:.1f} mm", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(img)
    cv2.imwrite(str(path), np.hstack(panels))


async def main():
    meta = json.loads((acc.PHOTOS / "meta.json").read_text())
    cache_n = len(list(acc.CACHE.iterdir()))
    top_rec, top_ctx = await acc.process(acc.PHOTOS / f"{SOLID_PLAN}.jpg", meta[f"{SOLID_PLAN}.jpg"], False)
    plan = sorted(top_ctx["res"].parts, key=lambda p: -p.area_mm2)
    plan_long = {"p1": plan[0].long_mm, "p2": plan[1].long_mm}
    rows, profs, ctxs, edge_imgs = [], {}, {}, {}
    for n in ACC_EDGES:
        rec, ctx = await acc.process(acc.PHOTOS / f"{n}.jpg", meta[f"{n}.jpg"], False, Params(card_frame="similarity"))
        res = ctx["res"]
        edge_imgs[n] = ctx["img"]
        ctxs[n] = (res, rec)
        for lab, part in zip(("p1", "p2"), sorted(res.parts, key=lambda p: -p.area_mm2)):
            row, prof, _ = profile_row(n, lab, res, part, rec, plan_long[lab], f"{rec['ref_kind']} {rec['reference'].get('conf')}")
            rows.append(row)
            profs[(n, lab)] = (part, prof)
    lock_plan = load("lock_pair_sep_front_01")
    lres = extract_outline(lock_plan[0], lock_plan[1], lock_plan[2], 25.0)
    lock_plan_long = sorted(lres.parts, key=lambda p: -p.area_mm2)[0].long_mm
    for n in LOCK_EDGES:
        img, part_mask, coin, _ = load(n)
        res = extract_outline(img, part_mask, coin, 25.0)
        ctxs[n] = (res, None)
        edge_imgs[n] = img
        ps = sorted(res.parts, key=lambda p: -p.area_mm2)
        row, prof, _ = profile_row(n, "p1", res, ps[0], None, lock_plan_long, "coin 25.0 (unverified)")
        rows.append(row)
        profs[(n, "p1")] = (ps[0], prof)

    # B max height vs the caliper values
    cmp_rows = []
    for r in rows:
        if r["blob"] != "p1":
            continue
        for key in ("max_height_mm", "max_height_length_matched_mm", "max_height_parallax_model_mm"):
            v = r.get(key)
            if v is not None:
                d = {h: abs(v - h) for h in B_HEIGHTS}
                cmp_rows.append({"photo": r["photo"], "variant": key, "value": v, "diff_to_14.4": d[14.4], "diff_to_11.8": d[11.8], "closer_to": min(d, key=d.get)})

    # images and stepped solids from SOLID_PLAN + SOLID_EDGE
    acc.compare_image(top_rec, top_ctx, SEND / f"{SOLID_PLAN}_outline_compare.png")
    res_e, rec_e = ctxs[SOLID_EDGE]
    sort_e = sorted(res_e.parts, key=lambda p: -p.area_mm2)
    draw_profile(edge_imgs[SOLID_EDGE], res_e, sort_e, [profs[(SOLID_EDGE, "p1")][1], profs[(SOLID_EDGE, "p2")][1]],
                 [r for r in rows if r["photo"] == SOLID_EDGE], SEND / f"{SOLID_EDGE}_profile.png")
    res_l, _ = ctxs["lock_edge_01"]
    draw_profile(edge_imgs["lock_edge_01"], res_l, sorted(res_l.parts, key=lambda p: -p.area_mm2)[:1], [profs[("lock_edge_01", "p1")][1]],
                 [], SEND / "lock_edge_01_profile.png")
    solids = []
    for lab, label in (("p1", "B"), ("p2", "A")):
        prof = profs[(SOLID_EDGE, lab)][1]
        f = plan_long[lab] / prof["length_mm"]
        stl = DEST / f"stepped_{label}_{SOLID_PLAN}__{SOLID_EDGE}.stl"
        mesh, regions = stepped_solid(plan[0 if lab == "p1" else 1], top_ctx["res"], prof, f, stl)
        render_views(stl, SEND / f"stepped_{label}_views.png")
        solids.append({"part": label, "stl": stl.name, "length_match_factor": f, "regions_scaled": regions, "mesh": mesh,
                       "bbox_mm": (read_stl(stl).reshape(-1, 3).max(0) - read_stl(stl).reshape(-1, 3).min(0)).tolist()})
    assert len(list(acc.CACHE.iterdir())) == cache_n, "a cache miss would have made an API call"

    def jsonable(r):
        return {k: (v if k != "steps" else v) for k, v in r.items()}
    (SEND / "results.json").write_text(json.dumps({"profiles": [jsonable(r) for r in rows], "B_height_vs_caliper": cmp_rows, "stepped_solids": solids,
                                                    "plan_long_mm": plan_long}, indent=1, default=float))
    write_summary(rows, cmp_rows, solids, plan_long)
    print((SEND / "summary.txt").read_text())


def write_summary(rows, cmp_rows, solids, plan_long):
    L = ["Geometry fidelity pass 1: edge-photo height profiles and stepped solids (cached masks, no API calls).",
         "Edge photo = part stood on its long edge, photographed from above. 'height' = silhouette extent across the long axis, at the reference plane's scale.",
         "A standing part's top is nearer the camera, so its silhouette is magnified; 3 versions of the height are given: raw; length-matched (profile scaled so its",
         "long axis equals the plan outline's long side from acc_coin_top_front_01); parallax model (raw / (d/(d-W)), W = assumed standing height = the plan's short side).",
         "Reference scale: coin 25.0 (lock photos: unverified) or card (similarity frame). The lock edge photos show one part; it is treated as B (an assumption).", ""]
    L.append("1. HEIGHT PROFILES (mm)")
    for r in rows:
        L.append(f"{r['photo']:20} {r['blob']:3} length {r['length_mm']:.1f}  max height {r['max_height_mm']:.2f} at x={r['x_at_max_mm']:.1f}  p95 {r['p95_height_mm']:.2f}"
                 + (f"  length-matched {r['max_height_length_matched_mm']:.2f} (f={r['length_match_factor']:.3f})" if r.get("max_height_length_matched_mm") else "")
                 + (f"  parallax-model {r['max_height_parallax_model_mm']:.2f} (d={r['distance_mm']:.0f}, W={r['standing_height_assumed_mm']:.1f})" if "max_height_parallax_model_mm" in r else ""))
        L.append("      steps: " + "; ".join(f"x {s['x0_mm']:.1f}-{s['x1_mm']:.1f}: {s['height_mm']:.2f}" for s in r["steps"]))
        L.append("      barrels: " + ("; ".join(f"diameter {b['diameter_mm']:.2f} (sweep {b['sweep_deg']:.0f} deg, centre x {b['centre_x_mm']:.1f})" for b in r["barrels"]) if r["barrels"] else "none found (no arc of >= 120 deg and 1.5-15 mm radius in the outline)"))
    L.append("")
    L.append("2. B MAX HEIGHT vs CALIPERS 14.4 / 11.8 (B = larger blob p1)")
    L.append(f"{'photo':20}{'version':32}{'value':>7}{'|d| 14.4':>10}{'|d| 11.8':>10}  closer")
    for c in cmp_rows:
        L.append(f"{c['photo']:20}{c['variant']:32}{c['value']:7.2f}{c['diff_to_14.4']:10.2f}{c['diff_to_11.8']:10.2f}  {c['closer_to']}")
    for var in ("max_height_mm", "max_height_length_matched_mm", "max_height_parallax_model_mm"):
        vs = [c for c in cmp_rows if c["variant"] == var]
        if vs:
            L.append(f"  {var}: closer to 14.4 in {sum(c['closer_to'] == 14.4 for c in vs)} of {len(vs)} photos; mean {sum(c['value'] for c in vs) / len(vs):.2f}")
    L.append("")
    L.append("3. STEPPED SOLIDS (plan outline from acc_coin_top_front_01, profile from acc_coin_edge_01, profile scaled by the length-match factor)")
    for s in solids:
        m = s["mesh"]
        L.append(f"  {s['part']}: {s['stl']}  bbox {s['bbox_mm'][0]:.1f} x {s['bbox_mm'][1]:.1f} x {s['bbox_mm'][2]:.1f} mm, f={s['length_match_factor']:.3f}, "
                 f"{m['triangles']} tris, {m['boundary_edges']} boundary, {m['non_manifold_edges']} non-manifold, {m['degenerate_triangles']} degenerate, volume {m['volume_mm3']:.0f} mm3")
        L.append("     regions: " + "; ".join(f"x {r['x0_mm']:.1f}-{r['x1_mm']:.1f}: {r['height_mm']:.2f}" for r in s["regions_scaled"]))
    sj = OUT / "solid" / "results.json"
    fails = []
    if sj.exists():
        sol = json.loads(sj.read_text())
        L.append("")
        L.append("4. SOLID CHECK, 10 parts (spikes/solid_check.py, 4.0 mm stand-in thickness): section holes vs detected holes, STL edge census")
        for r in sol:
            m = r["mesh"]
            L.append(f"  {r['photo']:24}{r['part']:>3}  holes detected {r['detected']}, section {r['section_counts']['0.5']}, extra {r['extra_in_solid']}  | {m['triangles']} tris, "
                     f"{m['boundary_edges']} boundary edges, {m['non_manifold_edges']} non-manifold, {m['degenerate_triangles']} degenerate  | {'PASS' if r['ok'] else 'FAIL'}"
                     + "  hole shape: " + ", ".join(h.get("shape", "?") for h in r["holes"]))
        shapes = [h.get("shape") for r in sol for h in r["holes"]]
        L.append(f"  {sum(r['ok'] for r in sol)} of {len(sol)} parts pass; hole shapes: " + ", ".join(f"{k} {shapes.count(k)}" for k in sorted(set(shapes))))
        fails.append(f"only {shapes.count('rect') + shapes.count('rect-from-clean-edges')} of {len(shapes)} holes became rectangles; the rest keep ragged fitted outlines (their edges deviate more than 0.3 mm or have no clean parallel pair)")
    withb = [f"{r['photo']} {r['blob']}" for r in rows if r["barrels"]]
    fails += [f"barrel (arc of >= 120 deg, 1.5-15 mm radius) found in {len(withb)} of {len(rows)} edge silhouettes ({', '.join(withb) or 'none'}); in the others the round end is split into shorter arcs or lines",
              "B height: the raw maximum, length-matched and parallax-model versions give 13.0 to 14.4 mm on the coin edge photos; they do not isolate one caliper value. The profile has two levels (about 13.8 and 11.7 raw) that are near 14.4 and 11.8, but the match is not tested",
              "the standing part's top is nearer the camera, so silhouette sizes are magnified; the correction used (length-match factor, or assumed standing height) is an assumption, not a measurement",
              "the lock_edge photos show another pose (a perpendicular piece, heights 13 to 29 mm); they are not comparable to the acc_coin_edge photos and give no B height",
              "acc_card_edge_01 gives 18.0 mm for B raw (60.0 mm long): that photo's pose differs or the card frame (similarity) is off; not resolved",
              "profile uses the silhouette extent only: walls hidden behind the top edge, undercuts and the standing part's base line are not modelled; the stepped solid is the plan extrusion cut by x-region heights only",
              "step labels in the profile PNGs overlap where steps are closer than about 2 mm; the plan/profile alignment (same left-right direction in both photos) is assumed, not checked",
              "(resolved) card photos now use the default card_frame='axes'; re-run solid_check and this script to refresh the numbers"]
    L.append("")
    L.append("5. WHAT STILL FAILS (honest list)")
    L += [f"  - {f}" for f in fails]
    (SEND / "summary.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
