"""Golden regression on part B (the A_orbit_* videos): carve + feature steps rerun from the private cache with the frozen settings.

Calipers: 51.6 x 41.9 mm footprint, 11.8 mm height; four end grooves. Owner answers (config.py) for groove depth, bump size, plate thickness.
Carves at 0.25 mm voxels (production: 0.15) to stay under 2 minutes; the 0.15 mm carve alone takes ~95 s on the 16 GB development laptop.
Skipped when the private cache is missing."""
import json
import time

import cv2
import numpy as np
import pytest

from capture import carve, config as C, dataset, features as F, holdout as H, model as Mo, pipeline as P

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
    ans = Mo.default_answers()
    vb, fb = Mo.rebuild(model, {**ans, "plate_thickness_mm": None})            # grooves + bump, plate not snapped
    tick('rebuilds')
    va, fa = Mo.rebuild(model, ans)
    assert carve.edge_census(va, fa)["watertight"]
    assert va[:, 2].min() < v[:, 2].min() - 0.2                                 # the bump stands out of the face the grooves open in
    xc, yc = (region[0] + region[1]) / 2, (region[2] + region[3]) / 2
    col = va[(np.abs(va[:, 0] - xc) < 1.0) & (np.abs(va[:, 1] - yc) < 1.0)]
    assert abs((col[:, 2].max() - col[:, 2].min()) - C.PLATE_THICKNESS_USER_MM) <= 0.15
    for up in (False, True):                                                    # roughness of both plate faces: not worse than before the snap
        assert F.face_rms(va, fa, region, up, 50000)[0] <= F.face_rms(vb, fb, region, up, 50000)[0]
    tick('rms')
    Mo.save(tmp_path / "model", field, g, feats)                                # rebuild() from the saved model + answers gives the same mesh
    tick('save/load')
    v2, f2 = Mo.rebuild(Mo.load(tmp_path / "model"), ans)
    assert np.array_equal(v2, va) and np.array_equal(f2, fa)
    dt = time.time() - t0
    print(f"footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f}, height {pl['z_max'] - pl['z_min']:.2f}, held-out IoU {iou:.4f}, grooves {len(ops)}, "
          f"bump {b['length_mm']:.2f} mm ({b['agree']} views), plate upper face z {pf['upper']['z']:.2f}, {dt:.0f} s")
    assert dt < 120
