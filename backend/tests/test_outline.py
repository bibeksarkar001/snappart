"""Synthetic-scene tests for measure.outline. The scenes are drawn at a known scale, so the true sizes are known."""
import math

import cv2
import numpy as np
import pytest

from measure.outline import Params, extract_outline, sample_primitives

PAPER = (225, 205, 205)       # BGR, a cool lavender-grey like the lock_set photos
BROWN = (30, 55, 110)
W, H = 1600, 1200
COIN_MM, COIN_PX = 25.0, 340  # 13.6 px/mm
PPM = COIN_PX / COIN_MM


def scene(tilt_deg=0.0, shadow=False, glare=False, hand=False, card=False, seed=1):
    """Brown 30 x 22 mm plate with a 9 x 3 mm paper slot, a 3 mm hole and optional shadow / glare patch / hand."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), PAPER, np.uint8)
    masks = {k: np.zeros((H, W), np.uint8) for k in ("part", "ref", "hand")}
    x0, y0, pw, ph = 500, 600, int(30 * PPM), int(22 * PPM)
    if shadow:   # grey soft shadow on the right, 12 px wide
        cv2.rectangle(img, (x0 + pw, y0 + 8), (x0 + pw + 12, y0 + ph + 8), (150, 140, 140), -1)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BROWN, -1)
    cv2.rectangle(img, (x0 + 80, y0 + 90), (x0 + 80 + int(9 * PPM), y0 + 90 + int(3 * PPM)), PAPER, -1)          # slot
    cv2.circle(img, (x0 + pw - 90, y0 + ph - 80), int(1.5 * PPM), PAPER, -1)                                  # hole
    if glare:    # warm, light, grey-brown patch on the plastic that is NOT paper
        cv2.rectangle(img, (x0 + 150, y0 + 200), (x0 + 150 + int(8 * PPM), y0 + 200 + int(4 * PPM)), (140, 160, 185), -1)
    cv2.rectangle(masks["part"], (x0 - 3, y0 - 3), (x0 + pw + 3 + (10 if shadow else 0), y0 + ph + 3 + (6 if shadow else 0)), 255, -1)  # SAM-like: fills holes
    if hand:
        cv2.circle(img, (x0 - 10, y0 + ph // 2), 70, (110, 140, 190), -1)                                    # skin-like blob over the left edge
        cv2.circle(masks["hand"], (x0 - 10, y0 + ph // 2), 70, 255, -1)
    if card:
        cw, ch = int(85.6 * 5), int(53.98 * 5)                                                              # 5 px/mm here
        cv2.rectangle(img, (100, 100), (100 + cw, 100 + ch), (90, 160, 90), -1)
        masks["ref"][100:100 + ch, 100:100 + cw] = 255
    else:
        cv2.circle(img, (1100, 300), COIN_PX // 2, (60, 190, 210), -1)
        cv2.circle(masks["ref"], (1100, 300), COIN_PX // 2, 255, -1)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    if tilt_deg:   # a camera tilt about a horizontal axis squashes the whole scene along y
        f = math.cos(math.radians(tilt_deg))
        img = cv2.resize(img, (W, int(H * f)), interpolation=cv2.INTER_AREA)
        masks = {k: cv2.resize(m, (W, int(H * f)), interpolation=cv2.INTER_NEAREST) for k, m in masks.items()}
    return img, masks


def test_plate_size_slot_and_hole():
    img, m = scene()
    r = extract_outline(img, m["part"], m["ref"], COIN_MM)
    assert len(r.parts) == 1
    p = r.parts[0]
    assert p.long_mm == pytest.approx(30.0, abs=0.15) and p.short_mm == pytest.approx(22.0, abs=0.15)
    assert all(q["type"] == "line" for q in p.primitives) and len(p.primitives) <= 6
    kinds = sorted(h.kind for h in p.holes)
    assert kinds == ["hole", "slot"]
    hole = next(h for h in p.holes if h.kind == "hole")
    assert hole.long_mm == pytest.approx(3.0, abs=0.2) and hole.primitives[0]["type"] == "circle"
    assert p.max_dev_mm < 0.15 and r.quality > 0.8


def test_tilt_is_unstretched():
    img, m = scene(tilt_deg=15)
    r = extract_outline(img, m["part"], m["ref"], COIN_MM)
    assert r.tilt_deg == pytest.approx(15, abs=1.0)
    p = r.parts[0]
    assert p.long_mm == pytest.approx(30.0, abs=0.25) and p.short_mm == pytest.approx(22.0, abs=0.25)
    assert any("tilt" in w for w in r.warnings) is True


def test_shadow_is_not_part_of_the_outline():
    img, m = scene(shadow=True)
    r = extract_outline(img, m["part"], m["ref"], COIN_MM)
    p = r.parts[0]
    assert p.long_mm == pytest.approx(30.0, abs=0.2) and p.short_mm == pytest.approx(22.0, abs=0.2)


def test_glare_is_not_a_hole():
    img, m = scene(glare=True)
    r = extract_outline(img, m["part"], m["ref"], COIN_MM)
    assert sorted(h.kind for h in r.parts[0].holes) == ["hole", "slot"]   # the glare patch adds nothing


def test_hand_contact_is_reported_and_hand_is_excluded():
    img, m = scene(hand=True)
    r = extract_outline(img, m["part"], m["ref"], COIN_MM, exclude_mask=m["hand"])
    assert r.parts[0].hand_contact_mm > 1.0
    assert any("hand contact" in w for w in r.warnings)
    assert r.quality_parts["hand"] < 1.0


def test_card_reference():
    img, m = scene(card=True)
    # the part is drawn at 13.6 px/mm but the card at 5 px/mm: only check that the card scale is applied (6.8 mm per ... 5 px/mm)
    r = extract_outline(img, m["part"], m["ref"], (85.6, 53.98), ref_kind="card")
    assert r.mm_per_px == pytest.approx(1 / 5, rel=0.01) and r.tilt_deg is None
    assert r.parts[0].long_mm == pytest.approx(30 * PPM / 5, rel=0.02)


def test_primitives_stay_on_the_contour():
    img, m = scene()
    p = extract_outline(img, m["part"], m["ref"], COIN_MM).parts[0]
    pts = sample_primitives(p.primitives, 0.05)
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(p.long_mm, abs=0.2)


def test_empty_part_and_bad_reference():
    img, m = scene()
    assert extract_outline(img, np.zeros_like(m["part"]), m["ref"], COIN_MM).parts == []
    with pytest.raises(ValueError):
        extract_outline(img, m["part"], np.zeros_like(m["ref"]), COIN_MM)
    with pytest.raises(ValueError):
        extract_outline(img, m["part"], m["ref"], COIN_MM, ref_kind="ruler")
