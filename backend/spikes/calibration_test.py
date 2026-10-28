"""Calibration fixes on the accuracy_set (cached masks, no API calls): coin ellipse analysis, no un-stretch for a level camera,
card homography from four corners, parallax k = d/(d-h). Before = out/accuracy/SEND_TO_CLAUDE/results.json (previous module);
after = the current module run over the same cached masks. Writes out/calibration/SEND_TO_CLAUDE/.
Usage: python backend/spikes/calibration_test.py
"""
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accuracy_set_test as acc  # noqa: E402
from measure.outline import Params, _reference_contour, card_corners  # noqa: E402
from spike_common import OUT  # noqa: E402

SEND = OUT / "calibration" / "SEND_TO_CLAUDE"
SEND.mkdir(parents=True, exist_ok=True)
TOP = ["acc_coin_top_01", "acc_coin_top_back_01", "acc_coin_top_front_01", "acc_coin_top_front_02", "acc_coin_top_front_03",
       "acc_card_top_back_01", "acc_card_top_front_01"]
COIN_T_MM = 1.9           # coin thickness, stated by the user
H_B = (11.8, 14.4)        # B height candidates: caliper, and the larger value to try
CARD_MM = acc.CARD_MM
DIMS = (("B", "long"), ("B", "short"), ("A", "long"), ("A", "short"))


def ellipse_row(rec, ctx, size):
    c = _reference_contour(ctx["ref"])
    (cx, cy), (a, b), ang = cv2.fitEllipse(c.astype(np.float32))
    major, minor = max(a, b), min(a, b)
    th = math.radians(ang if a >= b else ang + 90)                       # major axis direction
    rad = np.array([cx - size[0] / 2, cy - size[1] / 2])                 # image centre (assumed principal point) -> coin
    rad_deg = math.degrees(math.atan2(rad[1], rad[0]))
    diff = abs((math.degrees(th) - rad_deg + 90) % 180 - 90)             # angle between two undirected lines, 0..90
    s = rec["mm_per_px"]
    d = rec["distance_mm"]
    r_mm = float(np.linalg.norm(rad)) * s
    return {"photo": rec["photo"], "major_px": major, "minor_px": minor, "ratio": major / minor,
            "tilt_equiv_deg": math.degrees(math.acos(minor / major)), "major_dir_deg": math.degrees(th) % 180,
            "radial_dir_deg": rad_deg % 180, "angle_to_radial_deg": diff, "offset_from_centre_mm": r_mm,
            "major_mm_at_card_scale": major * 0.0717, "minor_mm_at_card_scale": minor * 0.0717,
            "major_mm_at_445": major * 445 / rec["f_px"], "minor_mm_at_445": minor * 445 / rec["f_px"],
            "predicted_ratio_side_wall": (25.0 + COIN_T_MM * r_mm / d) / 25.0, "mm_per_px_after": s}


def card_row(rec, ctx):
    q = card_corners(ctx["ref"])
    n = lambda i, j: float(np.linalg.norm(q[i] - q[j]))
    sides = [n(0, 1), n(1, 2), n(2, 3), n(3, 0)]
    area = float(cv2.contourArea(q.astype(np.float32)))
    s = math.sqrt(CARD_MM[0] * CARD_MM[1] / area)                         # the scale the homography frame uses
    pair = (sides[0], sides[2]) if sides[0] >= sides[1] else (sides[1], sides[3])
    other = (sides[1], sides[3]) if sides[0] >= sides[1] else (sides[0], sides[2])
    wl, ws = statistics.mean(pair) * s, statistics.mean(other) * s
    return {"photo": rec["photo"], "corners_px": q.tolist(), "long_mm": wl, "short_mm": ws, "aspect": wl / ws,
            "aspect_true": CARD_MM[0] / CARD_MM[1], "long_vs_85.60_pct": 100 * (wl / CARD_MM[0] - 1),
            "short_vs_53.98_pct": 100 * (ws / CARD_MM[1] - 1),
            "opposite_side_diff_pct": [100 * (pair[0] / pair[1] - 1), 100 * (other[0] / other[1] - 1)], "mm_per_px": s}


def lsq_k(rows):
    """measured = k * truth, least squares."""
    return sum(t * m for t, m in rows) / sum(t * t for t, _ in rows)


def part_rows(recs, part):
    return [(r["parts"][part]["truth"][dim], r["parts"][part][f"{dim}_mm"]) for r in recs for dim in ("long", "short")]


def main():
    meta = json.loads((acc.PHOTOS / "meta.json").read_text())
    before = {r["photo"]: r for r in json.loads((OUT / "accuracy" / "SEND_TO_CLAUDE" / "results.json").read_text())["photos"]}
    cache_n = len(list(acc.CACHE.iterdir()))

    async def run():
        return [await acc.process(acc.PHOTOS / f"{n}.jpg", meta[f"{n}.jpg"], False) for n in TOP]
    runs = asyncio.run(run())
    assert len(list(acc.CACHE.iterdir())) == cache_n, "a cache miss would have made an API call"
    recs = {r["photo"]: r for r, _ in runs}
    card_names = [n for n in TOP if "card" in n]

    async def run_variant(mode):
        return [(await acc.process(acc.PHOTOS / f"{n}.jpg", meta[f"{n}.jpg"], False, Params(card_frame=mode)))[0] for n in card_names]
    variants = {"homography": [recs[n] for n in card_names]}
    for mode in ("similarity", "axes"):
        variants[mode] = asyncio.run(run_variant(mode))
    ctxs = {r["photo"]: c for r, c in runs}
    for r in recs.values():
        assert "parts" in r and set(r["parts"]) == {"A", "B"}, r.get("error")

    # 1. coin ellipse
    size = (3024, 4032)
    ell = [ellipse_row(recs[n], ctxs[n], size) for n in TOP if "coin" in n]
    cards = [card_row(recs[n], ctxs[n]) for n in TOP if "card" in n]

    # 2. before / after
    table = []
    for n in TOP:
        for lab, dim in DIMS:
            b, a = before[n]["parts"][lab], recs[n]["parts"][lab]
            t = a["truth"][dim]
            table.append({"photo": n, "ref": recs[n]["ref_kind"], "part": lab, "dim": dim, "truth": t,
                          "before": b[f"{dim}_mm"], "after": a[f"{dim}_mm"], "err_before": b[f"{dim}_mm"] - t, "err_after": a[f"{dim}_mm"] - t})
    agg = {}
    for ref in ("coin", "card"):
        rows = [t for t in table if t["ref"] == ref]
        agg[ref] = {k: {"mean_signed": statistics.mean(r[f"err_{k}"] for r in rows), "mean_abs": statistics.mean(abs(r[f"err_{k}"]) for r in rows),
                        "worst": max(abs(r[f"err_{k}"]) for r in rows)} for k in ("before", "after")}
        agg[ref]["mm_per_px_before_after"] = [statistics.mean(before[n]["mm_per_px"] for n in TOP if ref in n),
                                              statistics.mean(recs[n]["mm_per_px"] for n in TOP if ref in n)]

    # card frame variants: errors per part and side (mean over the two card photos)
    cv_tab = {m: {f"{lab} {dim}": statistics.mean(r["parts"][lab][f"err_{dim}_mm"] for r in rs) for lab, dim in DIMS} |
              {"mean_signed": statistics.mean(r["parts"][lab][f"err_{dim}_mm"] for r in rs for lab, dim in DIMS),
               "mm_per_px": statistics.mean(r["mm_per_px"] for r in rs), "distance_mm": statistics.mean(r["distance_mm"] for r in rs)}
              for m, rs in variants.items()}
    # 3. parallax: card variants fitted on the card photos; coin photos out-of-sample
    coin_recs = [recs[n] for n in TOP if "coin" in n]
    par, resid, summ = {}, [], {}
    for mode, card_recs in list(variants.items()) + [("coin (no un-stretch)", coin_recs)]:
        d = statistics.mean(r["distance_mm"] for r in card_recs)
        kA, kB = lsq_k(part_rows(card_recs, "A")), lsq_k(part_rows(card_recs, "B"))
        par[mode] = {"d_mm": d, "A_fit": {"k": kA, "h_mm": d * (1 - 1 / kA)}, "B_fit_for_consistency": {"k": kB, "h_mm": d * (1 - 1 / kB)}}
    for mode, card_recs in list(variants.items()) + [("coin (no un-stretch)", coin_recs)]:
        hA = par[mode]["A_fit"]["h_mm"]
        for hb in H_B:
            for r in card_recs:
                d = r["distance_mm"]
                for lab, h in (("B", hb), ("A", hA)):
                    k = d / (d - h)
                    for dim in ("long", "short"):
                        p = r["parts"][lab]
                        resid.append({"mode": mode, "photo": r["photo"], "h_B": hb, "h_A": hA, "part": lab, "dim": dim, "truth": p["truth"][dim], "k": k,
                                      "measured": p[f"{dim}_mm"], "corrected": p[f"{dim}_mm"] / k,
                                      "err_before_parallax": p[f"{dim}_mm"] - p["truth"][dim], "err_after": p[f"{dim}_mm"] / k - p["truth"][dim]})
            rows = [x for x in resid if x["mode"] == mode and x["h_B"] == hb]
            summ[f"{mode} / h_B={hb}"] = {f"{lab} {dim}": statistics.mean(x["err_after"] for x in rows if x["part"] == lab and x["dim"] == dim) for lab, dim in DIMS} | {
                "rms": math.sqrt(statistics.mean(x["err_after"] ** 2 for x in rows)),
                "rms_without_parallax": math.sqrt(statistics.mean(x["err_before_parallax"] ** 2 for x in rows)), "h_A": hA}
    # coin photos with the h_A fitted on the (similarity) card photos: out-of-sample
    hA_card = par["similarity"]["A_fit"]["h_mm"]
    oos = {}
    for hb in H_B:
        rows = []
        for r in coin_recs:
            d = r["distance_mm"]
            for lab, h in (("B", hb), ("A", hA_card)):
                for dim in ("long", "short"):
                    rows.append((lab, dim, r["parts"][lab][f"{dim}_mm"] / (d / (d - h)) - r["parts"][lab]["truth"][dim]))
        oos[f"hB={hb}, hA={hA_card:.1f} (fitted on card/similarity)"] = {f"{lab} {dim}": statistics.mean(e for l, dm, e in rows if (l, dm) == (lab, dim)) for lab, dim in DIMS} | {
            "rms": math.sqrt(statistics.mean(e * e for _, _, e in rows))}

    # compare images
    for n in ("acc_coin_top_front_01", "acc_card_top_front_01"):
        acc.compare_image(recs[n], ctxs[n], SEND / f"{n}_compare.png")

    results = {"ellipse": ell, "card_corners": cards, "before_after": table, "aggregate": agg, "card_variants": cv_tab, "parallax": par, "residuals": resid,
               "residual_summary": summ, "coin_out_of_sample": oos, "photos_after": {n: {k: v for k, v in recs[n].items() if k not in ("parts",)} | {
                   "parts": {l: {k: v for k, v in p.items() if k != "primitives"} for l, p in recs[n]["parts"].items()}} for n in TOP}}
    (SEND / "results.json").write_text(json.dumps(results, indent=1, default=float))
    write_summary(ell, cards, table, agg, cv_tab, par, summ, oos)
    print((SEND / "summary.txt").read_text())


def write_summary(ell, cards, table, agg, cv_tab, par, summ, oos):
    L = ["Calibration fixes, accuracy_set, cached masks (no API calls). Truth: A 34.7 x 34.0, B 51.6 x 41.9, coin 25.0, card 85.60 x 53.98 mm.",
         "'before' = previous module (coin tilt un-stretch, card scaled along its own axes); 'after' = current module. Errors = ours - truth (mm).",
         "A's two truths differ by 0.7 mm: long/short pairing is sorted, so A's per-side errors can swap (the mean is safe).", "",
         "1. COIN ELLIPSE (is the 'tilt' radial?)  principal point assumed = image centre; coin thickness 1.9 mm (stated)",
         f"{'photo':22}{'ratio':>7}{'tilt-eq':>8}{'major dir':>10}{'radial dir':>11}{'angle':>7}{'off-axis mm':>12}{'pred ratio (wall)':>18}{'major/minor px':>16}{'major/minor mm @445':>21}{'@card scale':>16}"]
    for e in ell:
        L.append(f"{e['photo']:22}{e['ratio']:7.4f}{e['tilt_equiv_deg']:7.1f}d{e['major_dir_deg']:9.1f}d{e['radial_dir_deg']:10.1f}d{e['angle_to_radial_deg']:6.1f}d"
                 f"{e['offset_from_centre_mm']:12.1f}{e['predicted_ratio_side_wall']:18.4f}{e['major_px']:9.1f}/{e['minor_px']:.1f}"
                 f"{e['major_mm_at_445']:12.2f}/{e['minor_mm_at_445']:.2f}{e['major_mm_at_card_scale']:9.2f}/{e['minor_mm_at_card_scale']:.2f}")
    ang = [e["angle_to_radial_deg"] for e in ell]
    L.append(f"   angle between the major axis and the radial line: {min(ang):.1f} to {max(ang):.1f} deg (0 = radial, 90 = tangential). "
             f"Observed ratios {min(e['ratio'] for e in ell):.4f}-{max(e['ratio'] for e in ell):.4f}; wall model predicts "
             f"{min(e['predicted_ratio_side_wall'] for e in ell):.4f}-{max(e['predicted_ratio_side_wall'] for e in ell):.4f}.")
    L.append("   mm columns: diameters along the two ellipse axes at an assumed scale (445 mm distance, or the card photos' mean mm/px 0.0717); only their ratio is scale-free.")
    L.append("")
    L.append("   Card corners (homography input), measured with the area-consistent scale vs 85.60 x 53.98:")
    for c in cards:
        L.append(f"   {c['photo']}: long {c['long_mm']:.2f} ({c['long_vs_85.60_pct']:+.2f} %), short {c['short_mm']:.2f} ({c['short_vs_53.98_pct']:+.2f} %), "
                 f"aspect {c['aspect']:.4f} vs {c['aspect_true']:.4f}; opposite sides differ {c['opposite_side_diff_pct'][0]:+.2f} % (long), {c['opposite_side_diff_pct'][1]:+.2f} % (short)")
    L.append("")
    L.append("2. BEFORE / AFTER GEOMETRY FIX (coin: no un-stretch when axis ratio < 1.02; card: homography from four corners)")
    L.append(f"{'photo':24}{'part':>5}{'dim':>7}{'truth':>7}{'before':>8}{'err':>7}{'after':>8}{'err':>7}")
    for t in table:
        L.append(f"{t['photo']:24}{t['part']:>5}{t['dim']:>7}{t['truth']:7.1f}{t['before']:8.2f}{t['err_before']:+7.2f}{t['after']:8.2f}{t['err_after']:+7.2f}")
    for ref, a in agg.items():
        L.append(f"  {ref}: mean signed err {a['before']['mean_signed']:+.2f} -> {a['after']['mean_signed']:+.2f} mm; mean abs {a['before']['mean_abs']:.2f} -> "
                 f"{a['after']['mean_abs']:.2f}; worst {a['before']['worst']:.2f} -> {a['after']['worst']:.2f}; mean mm/px {a['mm_per_px_before_after'][0]:.5f} -> {a['mm_per_px_before_after'][1]:.5f}")
    L.append("")
    L.append("   Card frame variants (mean error mm over the 2 card photos): homography = four corners (as requested), similarity = rotation + one scale from the corners, axes = previous module")
    L.append(f"   {'frame':12}{'B long':>8}{'B short':>9}{'A long':>8}{'A short':>9}{'mean':>8}{'mm/px':>9}{'dist mm':>9}")
    for m, v in cv_tab.items():
        L.append(f"   {m:12}{v['B long']:+8.2f}{v['B short']:+9.2f}{v['A long']:+8.2f}{v['A short']:+9.2f}{v['mean_signed']:+8.2f}{v['mm_per_px']:9.5f}{v['distance_mm']:9.0f}")
    L.append(f"   coin photos for scale comparison: mean mm/px {agg['coin']['mm_per_px_before_after'][1]:.5f}")
    L.append("")
    L.append("3. PARALLAX  corrected = measured / k, k = d/(d-h), d = f_px * mm_per_px of that photo. h_A fitted per row group (4 rows, 1 parameter).")
    for m, v in par.items():
        L.append(f"   {m:22} d={v['d_mm']:.0f} mm: A fit k={v['A_fit']['k']:.4f} -> h_A={v['A_fit']['h_mm']:.1f} mm; (consistency: B fit k={v['B_fit_for_consistency']['k']:.4f} -> h_B={v['B_fit_for_consistency']['h_mm']:.1f}; given 11.8 / 14.4)")
    L.append("   Mean remaining error (mm) after geometry fix + parallax, per part and side; rows fitted in-sample for A except where noted:")
    L.append(f"   {'case':36}{'B long':>8}{'B short':>9}{'A long':>8}{'A short':>9}{'rms':>7}{'rms before parallax':>21}")
    for k, v in summ.items():
        L.append(f"   {k:36}{v['B long']:+8.2f}{v['B short']:+9.2f}{v['A long']:+8.2f}{v['A short']:+9.2f}{v['rms']:7.2f}{v['rms_without_parallax']:21.2f}")
    L.append("   Coin photos, out-of-sample (h_A fitted on the card photos, not on these):")
    for k, v in oos.items():
        L.append(f"   {k:60}{v['B long']:+8.2f}{v['B short']:+9.2f}{v['A long']:+8.2f}{v['A short']:+9.2f}{v['rms']:7.2f}")
    L.append("")
    L.append("Notes: truth values come from calipers; only 4 distinct numbers exist. Heights are not measured by us (B from calipers, A fitted).")
    (SEND / "summary.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
