"""Golden regression on part B (the A_orbit_* videos): carve + groove steps rerun from the private cache with the frozen settings.

Calipers: 51.6 x 41.9 mm footprint, 11.8 mm height; four end grooves. Skipped when the private cache is missing."""
import json
import time

import cv2
import numpy as np
import pytest

from capture import carve, config as C, dataset, features as F, holdout as H, pipeline as P

NAMES = ["A_orbit_01", "A_orbit_02", "A_orbit_03"]
CACHE = dataset.CACHE
NEED = [CACHE / "registration.json", CACHE / "masks", CACHE / "crops", CACHE.parent / "SEND_TO_CLAUDE" / "results.json"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEED), reason="private capture cache missing")


def test_golden_B():
    t0 = time.time()
    views = {n: dataset.load_views(n) for n in NAMES}
    T = {n: np.array(m) for n, m in json.loads((CACHE / "registration.json").read_text()).items()}
    # held-out IoU: one fold (10 views spread over the videos) at 0.25 mm voxels to stay fast; the full 3-fold value at 0.15 mm is 0.953
    iou, per, _ = H.joint(views, T, C.K_ROBUST, 0.25, n_hold=10, n_folds=1, ref=NAMES[0], grow=C.GROW_MM)
    _, g, field = P.carve_joint(views, T, NAMES[0])
    v, fc = P.mesh_of(field, g)
    vv = [(n, x.moved(T[n])) for n in NAMES for x in views[n]]
    imgs = [cv2.imread(str(CACHE / "crops" / f"{x.name}.jpg"), 0) for n in NAMES for x in views[n]]
    face, ops, _, _ = F.find_grooves(v, fc, vv, imgs)
    pl = carve.obb_plan(v)
    dt = time.time() - t0
    print(f"footprint {pl['length_mm']:.2f} x {pl['width_mm']:.2f}, height {pl['z_max'] - pl['z_min']:.2f}, held-out IoU {iou:.4f}, grooves {len(ops)}, {dt:.0f} s")
    assert abs(pl["length_mm"] - 51.6) <= 0.5 and abs(pl["width_mm"] - 41.9) <= 0.5
    assert abs(pl["z_max"] - pl["z_min"] - 11.8) <= 0.3
    assert iou >= 0.93
    assert len(ops) == 4
    assert all(abs(o["L"] - 7.4) <= 0.5 and abs(o["S"] - 2.6) <= 0.5 for o in ops)
    assert dt < 120
