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
