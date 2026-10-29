"""Question cards: implausible answers are rejected, every question type has a well-formed diagram, and the highlight colour is the reserved
colour that stands out most from the pixels it is drawn on."""
import xml.etree.ElementTree as ET

import numpy as np

from capture import question_cards as Q

PART = {"height_mm": 11.8, "length_mm": 51.6, "width_mm": 41.9}


def test_answer_sanity():
    assert Q.check_answer("groove_depth_mm", 3.7, PART) is None
    assert "deeper than the part" in Q.check_answer("groove_depth_mm", 12.0, PART)          # deeper than the part is tall
    assert Q.check_answer("plate_thickness_mm", 2.6, PART) is None
    assert Q.check_answer("plate_thickness_mm", 15, PART)
    for bad in (0, -1, "abc", float("nan"), None):
        assert Q.check_answer("bump_height_mm", bad, PART)
    assert Q.check_answer("bump_length_mm", 60, PART)
    assert Q.check_answer("bump_top_shape", "slightly rounded", PART) is None
    assert Q.check_answer("bump_top_shape", "pointy", PART)


def test_diagrams_and_colour():
    for qid in list(Q.TEXT) + ["review"]:
        root = ET.fromstring(Q.diagram(qid))
        assert root.tag.endswith("svg") and len(root) > 2
    magenta_crop = np.full((40, 40, 3), Q.HIGHLIGHTS["magenta"], np.uint8)
    band = np.ones((40, 40), np.uint8)
    assert Q.pick_colour(magenta_crop, band)[0] != "magenta"                                   # never the colour already under the outline
    brown = np.full((40, 40, 3), (40, 70, 140), np.uint8)
    assert Q.pick_colour(brown, band)[0] in Q.HIGHLIGHTS
