"""Part B's owner answers for the research spikes (single source: backend/tests/fixtures/part_b.json)."""
import json
from pathlib import Path

_F = json.loads((Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "part_b.json").read_text())
ANSWERS = _F["answers"]
CALIPERS = _F["calipers_mm"]
GROOVE_DEPTH_USER_MM = ANSWERS["groove_depth_mm"]
BUMP_LENGTH_USER_MM = ANSWERS["bump_length_mm"]
BUMP_WIDTH_USER_MM = ANSWERS["bump_width_mm"]
BUMP_HEIGHT_USER_MM = ANSWERS["bump_height_mm"]
BUMP_TOP_RADIUS_MM = ANSWERS["bump_top_radius_mm"]
PLATE_THICKNESS_USER_MM = ANSWERS["plate_thickness_mm"]


def answers() -> dict:
    return dict(ANSWERS)
