"""Corrections engine on a synthetic slab: each operation does what it says, rebuild() applies a list of them with no model call, and the
sentence adapter only lets schema-valid corrections through."""
import json

import numpy as np
import pytest

from capture import carve, corrections as Cr, model as Mo


def slab_model():
    """20 x 16 x 6 mm slab (z 0..6) with a 1 mm lump on its top over x 2..6, and a ridge on the bottom face."""
    g = carve.Grid([-12, -12, -5], [12, 12, 10], 0.2)
    X = g.points()
    top = 6 + ((X[:, 0] > 2) & (X[:, 0] < 6) & (np.abs(X[:, 1]) < 3))
    f = np.clip(np.minimum.reduce([10 - np.abs(X[:, 0]), 8 - np.abs(X[:, 1]), X[:, 2], top - X[:, 2]]), -3, 3).astype(np.float32).reshape(g.shape)
    ridge = {"kind": "ridge", "shape": "hill", "base_mid": [-4.0, 0, 0], "axis": [0, 1.0, 0], "across": [1.0, 0, 0], "normal": [0, 0, -1.0]}
    return {"field": f, "grid": g, "features": {"bump": ridge, "planes": {"top": {"n": [0, 0, 1.0], "d": -6.0}}}}


ANS = {"groove_depth_mm": None, "bump_length_mm": 8.0, "bump_width_mm": 2.3, "bump_height_mm": 2.0, "bump_top_radius_mm": 0.5, "plate_thickness_mm": None}
SQUARE = {"polygon_xy": [[1, -4], [7, -4], [7, 4], [1, 4]]}


def vol(m):
    return carve.edge_census(*m)["volume_mm3"]


def test_each_operation():
    m = slab_model()
    base = Mo.rebuild(m, ANS)
    assert carve.edge_census(*base)["watertight"]
    flat = Mo.rebuild(m, ANS, [{"op": "flatten_region", "region": SQUARE, "to_plane": "top", "thickness_mm": 6.0}])
    sel = (flat[0][:, 0] > 2.5) & (flat[0][:, 0] < 5.5) & (np.abs(flat[0][:, 1]) < 2.5) & (flat[0][:, 2] > 3)
    assert np.abs(flat[0][sel, 2] - 6.0).max() < 0.1                                  # the lump is gone: the top is the plane
    assert abs((vol(base) - vol(flat)) - 4 * 6 * 1.0) < 3.0                            # 4 x 6 x 1 mm lump removed
    cut = Mo.rebuild(m, ANS, [{"op": "remove_region", "region": {**SQUARE, "z_mm": [3.0, 20.0]}}])
    assert abs((vol(base) - vol(cut)) - (6 * 8 * 3 + 4 * 6 * 1)) < 10.0                # prism 6 x 8 above z 3 (+ the lump) removed (0.2 mm voxels, smoothed)
    shapes = {s: vol(Mo.rebuild(m, ANS, [{"op": "set_feature_shape", "feature_id": "bump", "shape": s}])) for s in ("box", "rounded", "hill")}
    assert shapes["box"] > shapes["rounded"] > shapes["hill"] > vol(Mo.rebuild(m, {**ANS, "bump_height_mm": None}))
    moved = Mo.rebuild(m, ANS, [{"op": "move_feature_to_opposite_face", "feature_id": "bump"}])
    assert base[0][:, 2].min() < -1.8 and moved[0][:, 2].min() > -0.1                 # off the bottom face ...
    assert moved[0][(np.abs(moved[0][:, 0] + 4) < 0.5) & (np.abs(moved[0][:, 1]) < 3), 2].max() > 7.8   # ... onto the top (z 6 + 2)
    added = Mo.rebuild(m, {**ANS, "bump_height_mm": 1.0}, [{"op": "add_feature", "feature_id": "bump", "feature": {**m["features"]["bump"], "base_mid": [4.0, 0, 0]}}])
    assert added[0][(np.abs(added[0][:, 0] - 4) < 0.3) & (np.abs(added[0][:, 1]) < 3), 2].min() < -0.8
    assert all(carve.edge_census(*x)["watertight"] for x in (flat, cut, moved, added))
    assert np.array_equal(Mo.rebuild(m, ANS, [{"op": "remove_region", "region": SQUARE}])[0], Mo.rebuild(m, ANS, [{"op": "remove_region", "region": SQUARE}])[0])


def test_hill_profile():
    p = Mo.profile("hill", 2.3, 2.3, 0.5)
    assert np.isclose(p[:, 1].max(), 2.3) and np.isclose(p[:, 0].max(), 1.15)
    assert 60 < Mo.side_slope_deg(p, 2.3) < 80                                          # steep sides rising to a rounded peak
    assert Mo.sd_poly(np.array([0.0]), np.array([1.0]), p)[0] < 0 < Mo.sd_poly(np.array([0.0]), np.array([2.5]), p)[0]


def test_schema_and_sentence_hook():
    ok = {"op": "flatten_region", "region": SQUARE, "to_plane": "bottom", "thickness_mm": 2.6}
    assert Cr.validate(ok, ["bump"], ["bottom"]) is ok
    for bad in ({"op": "explode"}, {"op": "set_feature_shape", "feature_id": "bump", "shape": "star"},
                {"op": "flatten_region", "region": {"polygon_xy": [[0, 0], [1, 1]]}, "to_plane": "bottom"},
                {"op": "move_feature_to_opposite_face", "feature_id": "nope"}):
        with pytest.raises(ValueError):
            Cr.validate(bad, ["bump"], ["bottom"])
    item = {"id": "R01", "polygon_xy": SQUARE["polygon_xy"]}
    seen = []
    fake = lambda prompt: (seen.append(prompt), json.dumps({"op": "flatten_region", "region": {"polygon_xy": item["polygon_xy"]}, "to_plane": "bottom"}))[1]
    c = Cr.from_sentence("this part is flat like the bottom", item, ["bump"], ["bottom"], fake)
    assert c["op"] == "flatten_region" and c["review_item"] == "R01" and "flat like the bottom" in seen[0]
    assert Cr.from_sentence("looks fine", item, ["bump"], ["bottom"], lambda p: '{"op": "none"}') is None
    with pytest.raises(ValueError):
        Cr.from_sentence("make it a star", item, ["bump"], ["bottom"], lambda p: '{"op": "set_feature_shape", "feature_id": "bump", "shape": "star"}')
