"""Outline -> extruded solid -> STL -> section at mid thickness: the through-holes in the solid must be the detected holes.

Photos: lock_pair_sep_front_01 (cached masks) and acc_{coin,card}_top_{front,back}_01 (accuracy cache; no API calls if cached).
Writes out/solid/ (git-ignored): one STL per part, cyan-hole compare PNGs, results.json, summary.txt. Exit code 1 if any detected hole is
missing in the solid (or the solid has extra holes). Thickness is a fixed 4.0 mm stand-in for the user's slider.
"""
import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accuracy_set_test as acc  # noqa: E402
from cad.extrude import export_stl, extrude_part  # noqa: E402
from cad.section import section_holes  # noqa: E402
from measure.outline import extract_outline, sample_primitives  # noqa: E402
from outline_run import load  # noqa: E402
from spike_common import OUT  # noqa: E402

DEST = OUT / "solid"
THICKNESS = 4.0
POS_TOL, SIZE_TOL = 0.5, 0.3          # mm: centre and long/short side agreement between detected hole and section hole
LEVELS = (0.5, 0.25, 0.75)            # section heights as fractions of the thickness; the first is the one that is matched
ACC_PHOTOS = ["acc_coin_top_front_01", "acc_coin_top_back_01", "acc_card_top_front_01", "acc_card_top_back_01"]
DEST.mkdir(exist_ok=True)


def match(part, stl):
    """Detected holes (mm, y down -> mirrored to CAD y up) against the section holes at LEVELS. Returns (rows, ok)."""
    det = []
    for h in part.holes:
        pts = sample_primitives(h.primitives, 0.05)
        mo = cv2.moments(pts.astype(np.float32))
        det.append({"kind": h.kind, "long": h.long_mm, "short": h.short_mm, "cx": mo["m10"] / mo["m00"], "cy": -mo["m01"] / mo["m00"], "area": h.area_mm2})
    sections = {f: section_holes(stl, THICKNESS * f) for f in LEVELS}
    cut = sections[LEVELS[0]]
    rows, used = [], set()
    for d in det:
        cand = [(np.hypot(c["cx"] - d["cx"], c["cy"] - d["cy"]), i) for i, c in enumerate(cut) if i not in used]
        dist, i = min(cand) if cand else (None, None)
        ok = i is not None and dist <= POS_TOL and abs(cut[i]["long"] - d["long"]) <= SIZE_TOL and abs(cut[i]["short"] - d["short"]) <= SIZE_TOL
        if ok:
            used.add(i)
        rows.append({**d, "found": ok, "section": cut[i] if i is not None else None, "dist_mm": dist})
    extra = [c for i, c in enumerate(cut) if i not in used]
    through = all(len(sections[f]) == len(cut) for f in LEVELS)
    return rows, extra, through, {str(f): len(v) for f, v in sections.items()}


def main():
    jobs = []   # (photo, rec, ctx)
    # lock photo: cached masks, coin 25.0 (unverified), no truth
    img, part, coin, _ = load("lock_pair_sep_front_01")
    res = extract_outline(img, part, coin, 25.0)
    ps = sorted(res.parts, key=lambda p: -p.area_mm2)
    rec = {"photo": "lock_pair_sep_front_01", "ref_kind": "coin",
           "parts": {f"p{i + 1}": {"long_mm": p.long_mm, "short_mm": p.short_mm} for i, p in enumerate(ps)}}
    jobs.append((rec, {"img": img, "ref": coin, "comps": [part], "res": res}))
    meta = json.loads((acc.PHOTOS / "meta.json").read_text())

    async def run_acc():
        return [await acc.process(acc.PHOTOS / f"{n}.jpg", meta[f"{n}.jpg"], False) for n in ACC_PHOTOS]
    for rec, ctx in asyncio.run(run_acc()):
        jobs.append((rec, ctx))

    out, lines, failed = [], [], False
    for rec, ctx in jobs:
        res = ctx["res"]
        for label, part in zip(rec["parts"], sorted(res.parts, key=lambda p: -p.area_mm2)):
            stl = DEST / f"{rec['photo']}_{label}.stl"
            export_stl(extrude_part(part, THICKNESS), stl)
            rows, extra, through, counts = match(part, stl)
            ok = all(r["found"] for r in rows) and not extra and through
            failed |= not ok
            out.append({"photo": rec["photo"], "part": label, "detected": len(rows), "section_counts": counts, "extra_in_solid": len(extra),
                        "through": through, "ok": ok, "holes": rows})
            lines.append(f"{rec['photo']:26} {label:3} detected {len(rows)} | section holes at 25/50/75% thickness {list(counts.values())} | "
                         f"extra {len(extra)} | {'PASS' if ok else 'FAIL'}")
            for r in rows:
                s = r["section"]
                lines.append(f"    {r['kind']:5} detected {r['long']:.2f} x {r['short']:.2f} mm -> section "
                             + (f"{s['long']:.2f} x {s['short']:.2f} mm, centre offset {r['dist_mm']:.2f} mm" if s else "MISSING") + ("" if r["found"] else "  <- MISMATCH"))
        acc.compare_image(rec, ctx, DEST / f"{rec['photo']}_compare.png")
    head = [f"Solid check: extrude outline {THICKNESS} mm (stand-in thickness), export STL, section holes vs detected holes.",
            f"Match rule: centre within {POS_TOL} mm, long and short side within {SIZE_TOL} mm; sections at {LEVELS} of thickness must agree in count.",
            "Compare PNGs: detected holes in cyan. lock_pair_sep_front_01 uses coin 25.0 (unverified); sizes only, hole check is scale-independent.",
            f"RESULT: {'FAIL' if failed else 'PASS'}", ""]
    (DEST / "summary.txt").write_text("\n".join(head + lines) + "\n")
    (DEST / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(head + lines))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
