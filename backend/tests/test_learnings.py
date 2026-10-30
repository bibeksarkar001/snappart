"""Each learnings rule (capture/learnings.py) with synthetic inputs, plus the generic per-job answers (no part-B constants)."""
import numpy as np

from capture import config as C, learnings as L, masks as M, model as Mo


def test_k_and_voxel_scale_with_views():
    assert L.k_for_views(229) == 12 and L.k_for_views(114) == 6 and L.k_for_views(10) == 2
    assert L.voxel_mm(229) == L.voxel_mm(40) == C.VOXEL_MM


def test_recesses_are_not_detected():
    low = {"regions": [{"centre_mm": [1, 2, 3], "area_mm2": 31.2}, {"centre_mm": [4, 5, 6], "area_mm2": 9.8}]}
    nd = L.not_detected_regions(low)
    assert len(nd) == 1 and nd[0]["status"] == "not_detected" and "too few camera views" in nd[0]["reason"]


def test_registration_disagreement_uses_more_views():
    r = L.registration_disagreement({"v2": {"point": [0, 0, 1.65], "views": 48}, "v3": {"point": [0, 0, 0], "views": 105}})
    assert r["use"] == "v3" and r["gap_mm"] == 1.65 and r["warn"] and "1.65 mm" in r["message"]
    assert not L.registration_disagreement({"a": {"point": [0, 0, 0], "views": 5}, "b": {"point": [0.2, 0, 0], "views": 9}})["warn"]
    assert L.registration_disagreement({"a": {"point": [0, 0, 0], "views": 5}}) is None


def test_plate_conflict_asks_and_shows_both():
    c = L.plate_conflict(0.9, 2.6)
    assert [x["value"] for x in c["candidates"]] == [0.9, 2.6] and "0.90 mm" in c["message"]
    assert L.plate_conflict(2.5, 2.6) is None and L.plate_conflict(None, 2.6) is None


def test_location_agreement():
    one = L.location_agreement({"v3": {"ends": [[0, 0, 0], [0, 19, 0]], "views": 29}, "v2": {"ends": [[0, 0, 0], [0, 18, 0]], "views": 2}})
    assert one["ask"] and one["reason"] == "found in one video only"
    two = L.location_agreement({"a": {"ends": [[0, 0, 0], [0, 19, 0]], "views": 9}, "b": {"ends": [[0, 19.5, 0], [0.3, 0.2, 0]], "views": 7}})
    assert two["agree"] and not two["ask"]                                    # same ridge, ends listed in the other order
    far = L.location_agreement({"a": {"ends": [[0, 0, 0], [0, 19, 0]], "views": 9}, "b": {"ends": [[3, 0, 0], [3, 19, 0]], "views": 7}})
    assert far["ask"] and far["gap_mm"] == 3.0


def test_sub_millimetre_is_not_claimed():
    assert not L.claimable(0.9) and L.claimable(1.0) and not L.claimable(None)
    a, src = Mo.estimate_answers({"plate": {"measured_separation_mm": 0.9}}, 11.7)
    assert src["plate_thickness_mm"] == "estimated" and a["plate_thickness_mm"] >= 0.9
    a, src = Mo.estimate_answers({"plate": {"measured_separation_mm": 2.4}}, 11.7)
    assert src["plate_thickness_mm"] == "measured" and a["plate_thickness_mm"] == 2.4


def test_answers_come_from_the_job():
    a, src = Mo.estimate_answers({"bump": {"kind": "ridge", "detected_length_mm": 19.37, "extent_mm": 1.9}}, 11.7)
    assert a["bump_length_mm"] == 19.37 and src["bump_length_mm"] == "measured"
    assert a["bump_height_mm"] == 1.9 and src["bump_height_mm"] == "measured" and src["bump_width_mm"] == "estimated"
    assert src["groove_depth_mm"] == "estimated" and a["groove_depth_mm"] == round(0.5 * 11.7, 3)
    assert not any(hasattr(C, k) for k in ("GROOVE_DEPTH_USER_MM", "BUMP_LENGTH_USER_MM", "PLATE_THICKNESS_USER_MM"))   # no part-B answers


def test_colour_model_is_learned_or_off():
    rng = np.random.default_rng(0)
    img = np.full((80, 80, 3), (200, 200, 200), np.uint8)                    # grey background
    img[20:60, 20:60] = (40, 160, 40)                                          # a green part (BGR)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    m = np.zeros((80, 80), np.uint8)
    m[20:60, 20:60] = 255
    lo, hi = M.learn_colour([(img, m)])
    _, h, _ = M.hue_chroma(img)
    assert lo < np.median(h[25:55, 25:55]) < hi and not lo < 30 < hi            # green, not brown
    grey = np.full((80, 80, 3), 128, np.uint8)
    assert M.learn_colour([(grey, np.full((80, 80), 255, np.uint8))]) is None   # achromatic: no colour gating
