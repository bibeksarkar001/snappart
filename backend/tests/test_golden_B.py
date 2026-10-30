"""Golden regression on part B (the A_orbit_* videos): carve + feature steps rerun from the private cache with the frozen settings.

Calipers: 51.6 x 41.9 mm footprint, 11.8 mm height; four end grooves. Owner answers (tests/fixtures/part_b.json) for groove depth, bump size, plate thickness.
Carves at 0.25 mm voxels (production: 0.15) to keep the run short; the time budget is 240 s; the 0.15 mm carve alone takes ~95 s on the 16 GB development laptop.
Skipped when the private cache is missing."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from capture import carve, config as C, dataset, features as F, holdout as H, model as Mo, pipeline as P

PART_B = json.loads((Path(__file__).parent / "fixtures" / "part_b.json").read_text())   # owner's caliper answers

NAMES = ["A_orbit_01", "A_orbit_02", "A_orbit_03"]
CACHE = dataset.CACHE
VS = 0.25
NEED = [CACHE / "registration.json", CACHE / "masks", CACHE / "crops", CACHE.parent / "SEND_TO_CLAUDE" / "results.json"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEED), reason="private capture cache missing")


def test_golden_B(tmp_path):
    t0 = time.time()
    last = [t0]

    def tick(name):
        print(f"[{name}] {time.time() - last[0]:.1f} s", flush=True)
        last[0] = time.time()
    views = {n: dataset.load_views(n) for n in NAMES}
    T = {n: np.array(m) for n, m in json.loads((CACHE / "registration.json").read_text()).items()}
    # held-out IoU: one fold (10 views spread over the videos, holdout.joint's rule); the full 3-fold value at 0.15 mm is 0.953
    hold = set()
    for n in NAMES:
        share = max(round(10 * len(views[n]) / sum(len(x) for x in views.values())), 1)
        hold |= {(n, i) for i in list(range(0, len(views[n]), max(len(views[n]) // share, 1)))[:share]}
    train = {n: [x for i, x in enumerate(views[n]) if (n, i) not in hold] for n in NAMES}
    _, gh, fh = P.carve_joint(train, T, NAMES[0], k=C.K_ROBUST, vs=VS)
    mh = P.mesh_of(fh, gh)
    iou = float(np.mean([carve.view_iou(*mh, views[n][i].moved(T[n])) for n, i in sorted(hold)]))
    tick('heldout')
    _, g, field = P.carve_joint(views, T, NAMES[0], vs=VS)
    tick('carve')
    v, fc = P.mesh_of(field, g)
    vv = [(n, x.moved(T[n])) for n in NAMES for x in views[n]]
    imgs = [cv2.imread(str(CACHE / "crops" / f"{x.name}.jpg"), 0) for n in NAMES for x in views[n]]
    face, ops, _, _ = P.find_grooves_parallel(v, fc, vv, imgs)
    tick('grooves')
    pl = carve.obb_plan(v)
    assert abs(pl["length_mm"] - 51.6) <= 0.5 and abs(pl["width_mm"] - 41.9) <= 0.5
    assert abs(pl["z_max"] - pl["z_min"] - 11.8) <= 0.3
    assert iou >= 0.93
    assert len(ops) == 4
    assert all(abs(o["L"] - 7.4) <= 0.5 and abs(o["S"] - 2.6) <= 0.5 for o in ops)
    # features: bump (detected position, owner size) and plate (measured upper face, owner thickness)
    feats = {"grooves": F.groove_features(ops, face)}
    b = F.detect_bump(vv, imgs, face, float(np.mean([o["dist_from_face_mm"] for o in ops])))
    tick('bump')
    assert b is not None and abs(b["length_mm"] - 19.7) <= 1.0
    feats["bump"] = F.bump_feature(b, feats["grooves"])
    region = F.plate_region(field, g, b["ends"])
    pf = F.plate_faces(vv, imgs, region, {"upper": NAMES[0]})
    tick('plate sweep')
    X = g.lo[0] + np.arange(field.shape[0]) * g.vs
    Y = g.lo[1] + np.arange(field.shape[1]) * g.vs
    mask = np.zeros(field.shape[:2], bool)
    mask[np.ix_((X >= region[0]) & (X <= region[1]), (Y >= region[2]) & (Y <= region[3]))] = True
    feats["plate"] = {"normal": [0, 0, 1.0], "d": -pf["upper"]["z"], "mask": mask, "region_xy": list(region)}
    model = {"field": field, "grid": g, "features": feats}
    ans = dict(PART_B["answers"])
    vb, fb = Mo.rebuild(model, {**ans, "plate_thickness_mm": None})            # grooves + bump, plate not snapped
    tick('rebuilds')
    va, fa = Mo.rebuild(model, ans)
    assert carve.edge_census(va, fa)["watertight"]
    assert va[:, 2].min() < v[:, 2].min() - 0.2                                 # the bump stands out of the face the grooves open in
    xc, yc = (region[0] + region[1]) / 2, (region[2] + region[3]) / 2
    col = va[(np.abs(va[:, 0] - xc) < 1.0) & (np.abs(va[:, 1] - yc) < 1.0)]
    assert abs((col[:, 2].max() - col[:, 2].min()) - PART_B["answers"]["plate_thickness_mm"]) <= 0.15
    for up in (False, True):                                                    # roughness of both plate faces: not worse than before the snap
        assert F.face_rms(va, fa, region, up, 50000)[0] <= F.face_rms(vb, fb, region, up, 50000)[0]
    tick('rms')
    Mo.save(tmp_path / "model", field, g, feats)                                # rebuild() from the saved model + answers gives the same mesh
    tick('save/load')
    v2, f2 = Mo.rebuild(Mo.load(tmp_path / "model"), ans)
    assert np.array_equal(v2, va) and np.array_equal(f2, fa)
    tick('rebuild')
    v4_checks(vv)
    tick('v4 checks')
    dt = time.time() - t0
    print(f"footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f}, height {pl['z_max'] - pl['z_min']:.2f}, held-out IoU {iou:.4f}, grooves {len(ops)}, "
          f"bump {b['length_mm']:.2f} mm ({b['agree']} views), plate upper face z {pf['upper']['z']:.2f}, {dt:.0f} s")
    assert dt < 240                                                             # owner: 240 s budget (alone ~115 s; slower inside the full suite)


V4 = CACHE.parents[1] / "v4"


def v4_checks(vv):
    """v4 (owner review): the owner's corrections rebuild the v4 model from the v3 model; the lock ridge is a hill on the slot column's inner
    wall; the review_4 region is flat on the bottom plane; the anomaly finder flags the owner's marks on the unfixed v3 model; a card exists
    for every question. Skipped piece by piece when the v4 outputs are missing."""
    if not (V4 / "corrections_v4.json").exists():
        return
    from capture import corrections as Cr, review as Rv, snap
    send = V4 / "SEND_TO_CLAUDE"
    corr = json.loads((V4 / "corrections_v4.json").read_text())
    m3 = Mo.load(CACHE.parents[1] / "v3" / "model_B")
    for c in corr:
        Cr.validate(c, list(m3["features"]) + ["bump"], [])
    ans = dict(PART_B["answers"])
    v, fc = Mo.rebuild(m3, ans, corr)                                            # corrections reproduce the v4 model
    assert carve.edge_census(v, fc)["watertight"]
    from cad.section import read_stl
    from scipy.spatial import cKDTree
    vb = np.unique(np.round(read_stl(send / "B_v4.stl").reshape(-1, 3), 4), axis=0)
    va = np.unique(np.round(v, 4), axis=0)
    assert len(va) == len(vb) and cKDTree(vb).query(va)[0].max() < 1e-3          # same vertices (STL stores float32)
    # ridge: a hill standing out of the slot column's inner wall toward the plate (outward normal -x), peak 2.3 mm out, narrower at the top
    r = next(c["feature"] for c in corr if c["op"] == "add_feature")
    assert r["kind"] == "ridge" and r["shape"] == "hill" and r["normal"][0] < -0.99
    f4 = Mo.apply(m3["field"], m3["grid"], m3["features"], ans, corr)
    c0, n, w = (np.asarray(r[k]) for k in ("base_mid", "normal", "across"))
    out = lambda h, u: Mo.field_at(f4, m3["grid"], c0 + n * h + w * u)[0] > 0
    assert out(2.1, 0) and not out(2.6, 0)                                        # peak at 2.3 mm
    assert out(0.3, 1.0) and not out(1.9, 0.9)                                    # wide at the base, narrow near the top
    # review_4 region: pocket side flat, outer face on the bottom plane (within 0.15 mm), measured short of the ridge and the end bevel
    s2 = json.loads((send / "results.json").read_text())["2_flat_plate"]
    bp, poly = s2["bottom_plane"], np.asarray(s2["flatness"]["review_4"]["polygon_xy"])
    from matplotlib.path import Path as MPath
    P, N = snap.oriented_points(v, fc, 200000)
    ins = MPath(poly).contains_points(P[:, :2]) & (P[:, 0] < 162.0)
    for side in (N[:, 2] < -0.5, N[:, 2] > 0.5):
        q = P[ins & side]
        assert np.sqrt((((q - q.mean(0)) @ np.linalg.svd(q - q.mean(0))[2][2]) ** 2).mean()) <= 0.15
    up = P[ins & (N[:, 2] > 0.5)]
    assert np.sqrt(((up @ np.asarray(bp["n"]) + bp["d"]) ** 2).mean()) <= 0.15
    # anomaly finder on the unfixed v3 model flags the owner's marks (frozen thresholds; marks saved by the v4 run)
    marks = np.load(V4 / "owner_marks.npz")
    v3, f3 = Mo.mesh(Mo.apply(m3["field"], m3["grid"], m3["features"], ans), m3["grid"])
    v2res = json.loads((CACHE.parents[1] / "v2" / "SEND_TO_CLAUDE" / "results.json").read_text())
    box = ("lock_bump", Mo.bump_op(m3["features"]["bump"], ans["bump_length_mm"], ans["bump_width_mm"], ans["bump_height_mm"]),
           {k: x["agree"] for k, x in v2res["2_bump"]["per_video"].items()})
    items = Rv.find(v3, f3, vv, [box])
    from scipy.spatial import cKDTree
    for k in marks.files:
        tm = cKDTree(marks[k])
        assert any((tm.query(v3[np.unique(f3[it["triangles"]])], distance_upper_bound=1.0)[0] < 1.0).mean() >= 0.2 or
                   (cKDTree(v3[np.unique(f3[it["triangles"]])]).query(marks[k], distance_upper_bound=1.0)[0] < 1.0).mean() >= 0.2 for it in items), k
    # a card for every question in the v4 report
    rep = json.loads((send / "report.json").read_text())
    for q in rep["questions"]:
        assert (send / "cards" / q["id"] / "card.png").exists() and (send / "cards" / q["id"] / "card.json").exists()
