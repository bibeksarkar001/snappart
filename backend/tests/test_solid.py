"""Outline -> solid -> STL section: the holes in the solid must be the detected holes. Also scale_correction."""
import pytest

from cad.extrude import export_stl, extrude_part
from measure.outline import Params, extract_outline
from cad.section import section_holes
from test_outline import COIN_MM, scene


def test_holes_survive_into_the_stl(tmp_path):
    img, m = scene()
    p = extract_outline(img, m["part"], m["ref"], COIN_MM).parts[0]
    export_stl(extrude_part(p, 4.0), tmp_path / "p.stl")
    cut = section_holes(tmp_path / "p.stl", 2.0)
    assert len(cut) == len(p.holes) == 2
    assert sorted(h["long"] for h in cut) == pytest.approx(sorted(h.long_mm for h in p.holes), abs=0.3)   # solid is built from the fitted primitives


def test_scale_correction_scales_mm_results():
    img, m = scene()
    a = extract_outline(img, m["part"], m["ref"], COIN_MM).parts[0]
    b = extract_outline(img, m["part"], m["ref"], COIN_MM, params=Params(scale_correction=0.97)).parts[0]
    assert b.long_mm == pytest.approx(a.long_mm * 0.97, abs=0.03) and b.short_mm == pytest.approx(a.short_mm * 0.97, abs=0.03)
    assert [h.long_mm for h in b.holes] == pytest.approx([h.long_mm * 0.97 for h in a.holes], abs=0.05)
    with pytest.raises(ValueError):
        extract_outline(img, m["part"], m["ref"], COIN_MM, params=Params(scale_correction=0.0))


def test_check_notices_a_missing_hole(tmp_path):
    img, m = scene()
    p = extract_outline(img, m["part"], m["ref"], COIN_MM).parts[0]
    p.holes.clear()                                       # solid without holes: the section must show none
    export_stl(extrude_part(p, 4.0), tmp_path / "p.stl")
    assert section_holes(tmp_path / "p.stl", 2.0) == []


def test_height_profile_of_a_stepped_silhouette():
    import numpy as np
    from measure.profile import height_profile
    poly = [(0, 0), (50, 0), (50, 10), (20, 10), (20, 6), (0, 6)]                    # 6 mm high for x 0..20, 10 mm for x 20..50
    prims = [{"type": "line", "p0": list(poly[i]), "p1": list(poly[(i + 1) % len(poly)])} for i in range(len(poly))]
    r = height_profile(prims, direction=np.array([1.0, 0.0]))
    assert r["max_height_mm"] == pytest.approx(10.0, abs=0.1) and r["length_mm"] == pytest.approx(50.0, abs=0.1)
    assert [round(s["height_mm"]) for s in r["steps"]] == [6, 10]
    assert r["steps"][1]["x0_mm"] == pytest.approx(20.0, abs=0.7)
