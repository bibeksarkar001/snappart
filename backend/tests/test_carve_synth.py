"""Stage 0: carve a known part from 60 virtual cameras whose poses come from detected mat markers."""
from capture import synth


def test_synthetic_carve_matches_truth():
    r = synth.report(*synth.build())
    print(r)
    b = r["carved_pts_by_z_band"]
    assert r["census"]["watertight"] and r["census"]["volume_mm3"] > 0
    assert r["camera_centre_err_mm"]["max"] < 0.3 and r["reproj_rms_px_max"] < 1.0
    # side walls and slot are fixed by the silhouettes; the top face and above are inflated by the elevation coverage (a hull limit)
    assert b["z 1..5.5 (side walls, slot)"]["mean_mm"] < 0.25 and b["z 1..5.5 (side walls, slot)"]["max_mm"] < 1.5
