"""Dataset harness (skeleton): reads the private test set's manifest and writes examples/ledger.csv with numbers and labels only (never
video paths). Builds are not run yet: every row is written with status "not_run" and its caliper values, so the ledger's shape is fixed
before results arrive. Run from the repo root: `python scripts/run_dataset.py`.

assets/private/testset/manifest.csv columns: part_id, label, video_1, video_2, video_3, length_mm, width_mm, height_mm
(video_* are paths relative to the testset folder; caliper values may be empty)."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "assets" / "private" / "testset"
MANIFEST = TESTSET / "manifest.csv"
LEDGER = ROOT / "examples" / "ledger.csv"
NEED = ["part_id", "label", "video_1", "video_2", "video_3", "length_mm", "width_mm", "height_mm"]
OUT = ["part_id", "label", "status", "caliper_length_mm", "caliper_width_mm", "caliper_height_mm", "measured_length_mm", "measured_width_mm",
       "measured_height_mm", "held_out_iou", "quality"]


def number(x: str):
    x = (x or "").strip()
    if not x:
        return ""
    v = float(x)                                                         # a malformed number stops the run with the row named
    return f"{v:.2f}"


def main() -> int:
    if not MANIFEST.exists():
        print(f"No test set found ({MANIFEST.relative_to(ROOT)} is missing). Nothing to do.")
        return 0
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in NEED if rows and c not in rows[0]]
    if not rows or missing:
        print(f"manifest.csv has no rows or lacks columns: {missing or NEED}")
        return 1
    out = []
    for i, r in enumerate(rows, 2):
        try:
            cal = [number(r[k]) for k in ("length_mm", "width_mm", "height_mm")]
        except ValueError:
            print(f"manifest.csv line {i}: a caliper value is not a number")
            return 1
        present = all((TESTSET / r[k]).is_file() for k in ("video_1", "video_2", "video_3"))
        out.append({"part_id": r["part_id"], "label": r["label"], "status": "not_run" if present else "videos_missing",
                    "caliper_length_mm": cal[0], "caliper_width_mm": cal[1], "caliper_height_mm": cal[2],
                    **{k: "" for k in OUT[6:]}})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} parts -> {LEDGER.relative_to(ROOT)} (builds not run yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
