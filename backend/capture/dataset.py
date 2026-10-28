"""Views for carving, built from the stage 1 selection and the stage 2 masks. Each view works on a crop of its frame: the principal point is shifted by the crop origin."""
import json
from pathlib import Path

import cv2
import numpy as np

from . import carve

CACHE = Path(__file__).resolve().parents[1] / "spikes" / "out" / "carve" / "cache"


def load_views(name: str, cache: Path = CACHE) -> list[carve.View]:
    sel = json.loads((cache / f"sel_{name}.json").read_text())
    boxes = {}
    stage = json.loads((cache.parent / "SEND_TO_CLAUDE" / "results.json").read_text())["stage2"]["frames"]
    for r in stage:
        if r["video"] == name and r["status"].startswith("ok"):
            boxes[str(r["idx"])] = r["box"]
    out = []
    for k, box in boxes.items():
        fr = sel["frames"][k]
        m = cv2.imread(str(cache / "masks" / f"{name}_{k}.png"), 0)
        K = np.array(sel["K"], float).copy()
        K[0, 2] -= box[0]
        K[1, 2] -= box[1]
        R = cv2.Rodrigues(np.array(fr["rvec"], float))[0]
        v = carve.View(K, R, np.array(fr["tvec"], float), carve.mask_sdf(m), f"{name}_{k}", name)
        v.az, v.el = fr["az"], fr["el"]
        out.append(v)
    return sorted(out, key=lambda v: int(v.name.rsplit("_", 1)[1]))
