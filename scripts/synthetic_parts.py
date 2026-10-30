"""Generalisation table on SYNTHETIC parts (CadQuery shapes, noisy three-video capture rendered on the mat): size errors, held-out IoU,
volume ratio, unseen (not_detected) area. Usage (repo root): python scripts/synthetic_parts.py [--views 60] [--voxel 0.25] > table.md"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from capture import report as Rp, synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--voxel", type=float, default=0.25)
    a = ap.parse_args()
    rows = []
    for name, wp in synth.parts().items():
        t0 = time.time()
        truth = synth._at_centre(wp)
        res = synth.build_part(*truth, n=a.views, vs=a.voxel)
        m = synth.part_metrics(truth, res)
        low = Rp.low_confidence(res["train_views"], *res["mesh"], floors=res["floors"])
        req, rec = Rp.coverage_requests(low)
        m.update(name=name, seconds=round(time.time() - t0, 1), not_detected_mm2=round(sum(r["area_mm2"] for r in rec), 1), film_requests=len(req))
        rows.append(m)
        print(json.dumps(m), file=sys.stderr, flush=True)
    print("| Part (synthetic) | Truth L x W x H (mm) | Error L / W / H (mm) | Held-out IoU | Volume ratio | Not detected (mm2) |")
    print("|---|---|---|---|---|---|")
    for m in rows:
        t, e = m["truth_mm"], m["error_mm"]
        print(f"| {m['name']} | {t[0]:.1f} x {t[1]:.1f} x {t[2]:.1f} | {e[0]:+.2f} / {e[1]:+.2f} / {e[2]:+.2f} | {m['held_out_iou']:.3f} | "
              f"{m['volume_ratio']:.2f} | {m['not_detected_mm2']:.0f} |")


if __name__ == "__main__":
    main()
