"""Generalisation on SYNTHETIC parts (CadQuery shapes, noisy three-video capture on the rendered mat). Bounds are regression limits around
the measured table in docs/accuracy.md ("Synthetic parts"), not accuracy claims. 45 views at 0.3 mm keep this short."""
import pytest

from capture import carve, report as Rp, synth

PARTS = synth.parts()


def run(name):
    truth = synth._at_centre(PARTS[name])
    res = synth.build_part(*truth, n=45, vs=0.3)
    return truth, res, synth.part_metrics(truth, res)


@pytest.mark.parametrize("name", ["box", "plate with holes", "hill ridge part"])
def test_synthetic_part(name):
    truth, res, m = run(name)
    assert carve.edge_census(*res["mesh"])["watertight"]
    assert m["camera_centre_err_max_mm"] < 0.3
    assert abs(m["error_mm"][2]) < 0.2                                         # height: fixed by the two mat planes
    assert all(0.0 <= e < 1.8 for e in m["error_mm"][:2])                      # footprint: the robust carve's known overshoot (see learnings)
    assert m["held_out_iou"] >= 0.95


@pytest.mark.xfail(strict=True, reason="known gap: a cup's interior opens onto a face that rests on the mat in the flipped video, so the "
                                       "filled hull there looks floor-fixed and is not reported not_detected (docs/learnings.md)")
def test_thin_walled_interior_is_reported():
    truth, res, m = run("thin-walled part")
    assert m["volume_ratio"] > 2                                               # the hull fills the cup
    low = Rp.low_confidence(res["train_views"], *res["mesh"], floors=res["floors"])
    _, recesses = Rp.coverage_requests(low)
    assert sum(r["area_mm2"] for r in recesses) > 200
