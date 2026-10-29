"""rebuild(model, answers) on a synthetic slab: answers change only the features they control, with no model call."""
import numpy as np

from capture import carve, model as Mo, report


def synth_model():
    g = carve.Grid([-12, -12, -4], [12, 12, 8], 0.25)
    X = g.points()
    field = np.clip(np.minimum.reduce([10 - np.abs(X[:, 0]), 8 - np.abs(X[:, 1]), X[:, 2], 6 - X[:, 2]]), -3, 3).astype(np.float32).reshape(g.shape)
    ax = [[0, 1.0, 0], [1.0, 0, 0], [0, 0, -1.0]]                          # long along y, short along x, outward = -z (bottom face)
    feats = {"grooves": [{"centre": [6.0, 0.0, 0.0], "axes": ax, "long_mm": 6.0, "short_mm": 2.0, "views": 9}],
             "bump": {"edge_mid": [2.0, 0.0, 0.0], "axis": [0, 1.0, 0], "toward_openings": [1.0, 0, 0], "normal": [0, 0, -1.0], "detected_length_mm": 8.0,
                      "ends": [[2, -4, 0], [2, 4, 0]], "source_video": "synthetic", "agreeing_views": 5}}
    return {"field": field, "grid": g, "features": feats}


def volume(m):
    return carve.edge_census(*m)["volume_mm3"]


def test_rebuild_applies_answers():
    m = synth_model()
    base = {"groove_depth_mm": 2.0, "bump_length_mm": 8.0, "bump_width_mm": 2.0, "bump_height_mm": 1.0, "plate_thickness_mm": None}
    a = Mo.rebuild(m, base)
    b = Mo.rebuild(m, {**base, "groove_depth_mm": 3.0})
    c = Mo.rebuild(m, {**base, "bump_height_mm": 2.0})
    assert carve.edge_census(*a)["watertight"] and carve.edge_census(*b)["watertight"] and carve.edge_census(*c)["watertight"]
    # 1 mm deeper pocket of 6 x 2 removes ~12 mm3; 1 mm taller bump of 8 x 2 adds ~16 mm3 (voxel 0.25 mm, light smoothing)
    assert abs((volume(a) - volume(b)) - 12.0) < 2.5
    assert abs((volume(c) - volume(a)) - 16.0) < 3.0
    assert c[0][:, 2].min() < a[0][:, 2].min() - 0.8                         # the bump reaches 1 mm further out of the bottom face
    assert np.array_equal(Mo.rebuild(m, base)[0], a[0])                       # same answers -> same model
    try:
        Mo.rebuild(m, {**base, "bump_height_mm": 5.0})                        # would leave the grid: refused, not clipped
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    qs = report.questions(m["features"], {"groove_depth_mm": 2.0})
    assert {q["id"] for q in qs} >= {"groove_depth_mm", "bump_height_mm"} and all(q["hint"] and q["unit"] == "mm" for q in qs)
