"""Calibrated multi-view carving of part A from three videos. Usage: python backend/spikes/carve_run.py --stage 0..6

Writes out/carve/SEND_TO_CLAUDE/ after every stage: stageN.txt (measured numbers), summary.txt (all stages so far),
results.json (merged), PNGs and STLs. Videos and frames are private: caches live under out/ (git-ignored).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from spike_common import OUT, ROOT  # noqa: E402

CARVE = OUT / "carve"
SEND = CARVE / "SEND_TO_CLAUDE"
VIDEOS = ROOT / "assets" / "private" / "lock_set" / "Video"
NAMES = ["A_orbit_01", "A_orbit_02", "A_orbit_03"]
for d in (CARVE, SEND):
    d.mkdir(parents=True, exist_ok=True)


def save_stage(n: int, text: str, results: dict):
    """Write stageN.txt, rebuild summary.txt from all stage files, merge results into results.json."""
    (SEND / f"stage{n}.txt").write_text(text)
    (SEND / "summary.txt").write_text("\n\n".join(p.read_text() for p in sorted(SEND.glob("stage*.txt"))))
    f = SEND / "results.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[f"stage{n}"] = results
    f.write_text(json.dumps(cur, indent=1, default=float))


def stage0():
    from capture import synth
    r = synth.report(*synth.build())
    b = r["carved_pts_by_z_band"]
    L = ["STAGE 0: synthetic test (plate 34 x 30 x 6 mm, through-slot 22 x 12, 4.0 mm bar; 60 virtual cameras, elevation 15-89 deg, 1920x1080, f 1900 px)",
         "Poses come from detected mat markers (not ground truth); masks are exact anti-aliased silhouettes; k=2; voxel 0.25 mm; Taubin 5 iterations.",
         f"surface distance carve<->truth: mean {r['overall']['mean_mm']:.3f} mm, p95 {r['overall']['p95_mm']:.3f}, MAX {r['overall']['max_mm']:.3f}",
         "carved surface points by height band (distance to truth):"]
    L += [f"  {k}: mean {v['mean_mm']:.3f} mm, max {v['max_mm']:.3f}, n={v['n']}" for k, v in b.items()]
    L += [f"extent carve {[round(x, 2) for x in r['carved_extent_mm']]} vs truth {r['truth_extent_mm']}; volume carve {r['census']['volume_mm3']:.0f} vs truth {r['truth_volume_mm3']:.0f} mm3",
          f"watertight {r['census']['watertight']}, boundary edges {r['census']['boundary_edges']}, non-manifold {r['census']['non_manifold_edges']}",
          f"camera centre error from marker PnP: mean {r['camera_centre_err_mm']['mean']:.3f} mm, max {r['camera_centre_err_mm']['max']:.3f}; reprojection rms max {r['reproj_rms_px_max']:.2f} px",
          "NOTE: the error is concentrated above the top face (hull inflation from limited elevation coverage, 60 views); side walls and slot are far smaller."]
    save_stage(0, "\n".join(L), r)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    a = ap.parse_args()
    {0: stage0}[a.stage]()
