"""Phase 1 spike: run the production module backend/measure/outline.py over the private lock_set photos (cached masks only).

Reads out/lock_set/*, out/lock_set/refined/hand_cache/* and the previous run's out/lock_set/refined/run2_previous_SEND/results.json
(the "before" numbers). Writes only out/lock_set/refined/ (git-ignored). No API calls.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.outline import Params, extract_outline, sample_primitives  # noqa: E402
from spike_common import OUT, ROOT  # noqa: E402

SRC = OUT / "lock_set"
R = SRC / "refined"
SEND = R / "SEND_TO_CLAUDE"
PHOTOS = ROOT / "assets" / "private" / "lock_set"
COINS = {"25.0": 25.0, "23.0": 23.0}
COMPARE = ["lock_pair_sep_front_01", "lock_pair_clipped_top_01", "lock_single_top_03", "lock_edge_01", "lock_edge_03", "lock_hand_01"]
PAIRS = [("lock_single_top_03", "lock_single_top_04"), ("lock_single_top_05", "lock_single_top_06")]
COL = {"line": (0, 0, 255), "arc": (0, 200, 0), "circle": (255, 0, 255)}
HOLE_COL = (255, 200, 0)


def load(stem):
    img = cv2.imread(str(PHOTOS / f"{stem}.jpg"))
    part = cv2.imread(str(SRC / f"{stem}_part_sam.png"), 0)
    hand = None
    if "hand" in stem:
        coin = cv2.imread(str(R / "hand_cache" / f"{stem}_coin.png"), 0)
        hand = cv2.imread(str(SRC / f"{stem}_hand.png"), 0)
    else:
        coin = cv2.imread(str(SRC / f"{stem}_coin.png"), 0)
    return img, part, coin, hand


def filled(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def iou(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def fit_iou(res, sam):
    ras = np.zeros_like(sam)
    for p in res.parts:
        cv2.fillPoly(ras, [np.round(res.frame.to_px(sample_primitives(p.primitives, 0.05))).astype(np.int32)], 255)
    return iou(ras, filled(sam))


def draw(img, res, coin, width):
    """Crop to part + coin, scale to `width` px wide, draw primitives (red line, green arc, magenta circle) and holes (cyan)."""
    f = res.frame
    pts = [f.to_px(sample_primitives(p.primitives, 0.2)) for p in res.parts]
    cy, cx = np.nonzero(coin)
    allp = np.vstack(pts + [np.array([[cx.min(), cy.min()], [cx.max(), cy.max()]])])
    h, w = img.shape[:2]
    x0, y0 = np.maximum(allp.min(0) - 120, 0).astype(int)
    x1, y1 = np.minimum(allp.max(0) + 120, [w, h]).astype(int)
    left = img[y0:y1, x0:x1].copy()
    right = left.copy()
    scale = width / (x1 - x0)
    th = max(2, int(round(2.5 / scale)))

    def poly(P, col):
        cv2.polylines(right, [np.round((P - [x0, y0]) * 16).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=4)
    for part in res.parts:
        for q in part.primitives:
            poly(f.to_px(sample_primitives([q], 0.05)), COL[q["type"]])
            for k in ("p0", "p1"):
                if k in q:
                    cv2.circle(right, tuple(np.round(f.to_px([q[k]])[0] - [x0, y0]).astype(int)), th * 2, (255, 255, 255), -1)
        for hole in part.holes:
            for q in hole.primitives:
                poly(f.to_px(sample_primitives([q], 0.03)), HOLE_COL)
    sz = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return sz(left), sz(right)


def main():
    old = json.loads((R / "run2_previous_SEND" / "results.json").read_text())
    stems = sorted(p.stem for p in PHOTOS.glob("lock_*.jpg"))
    recs = {}
    for stem in stems:
        img, part, coin, hand = load(stem)
        rec = {"image": stem, "by_coin": {}}
        for name, mm in COINS.items():
            t = time.perf_counter()
            res = extract_outline(img, part, coin, mm, exclude_mask=hand)
            dt = time.perf_counter() - t
            rec["by_coin"][name] = {**res.to_dict(), "runtime_s": dt}
            if name == "25.0":
                res25 = res
        rec["iou_fit_vs_sam"] = fit_iou(res25, part)
        recs[stem] = rec
        _, right = draw(img, res25, coin, 1200)
        cv2.imwrite(str(R / f"{stem}_fit.png"), right)
        if stem in COMPARE:
            l2, r2 = draw(img, res25, coin, 800)
            cv2.imwrite(str(SEND / f"{stem}_compare.png"), np.hstack([l2, r2]))
        a = rec["by_coin"]["25.0"]
        nh = sum(h["kind"] == "hole" for p in a["parts"] for h in p["holes"])
        ns = sum(h["kind"] == "slot" for p in a["parts"] for h in p["holes"])
        print(f"{stem}: prims={sum(len(p['primitives']) for p in a['parts'])} dev={max(p['max_dev_mm'] for p in a['parts']):.3f} h/s={nh}/{ns} q={a['quality']:.2f}", flush=True)
    (SEND / "results.json").write_text(json.dumps(recs, indent=1, default=float))
    write_summary(recs, old)


VARIANTS = [
    ("fit 0.07, no clean-up (old settings)", Params(fit_tol_mm=0.07, min_segment_mm=0.0, collinear_deg=0.0, snap_deg=2.0, snap_tol_mm=0.07)),
    ("fit 0.10, no clean-up", Params(min_segment_mm=0.0, collinear_deg=0.0, snap_deg=2.0, snap_tol_mm=0.07)),
    ("CHOSEN: fit 0.10 + short/collinear/snap", Params()),
    ("chosen + drop arcs r<0.5 mm", Params(min_arc_radius_mm=0.5)),
    ("chosen + sharp corners <1.2 mm", Params(min_arc_radius_mm=0.5, corner_mm=1.2)),
]


def variant_table(L):
    stems = [s for s in sorted(p.stem for p in PHOTOS.glob("lock_*.jpg")) if "hand" not in s]
    data = {s: load(s) for s in stems}
    L.append(f"\n== 2. primitive reduction, 16 measuring photos, coin 25.0 (per-photo max deviation: median / p90 / worst) ==")
    for label, prm in VARIANTS:
        n, devs = 0, []
        for s in stems:
            img, part, coin, _ = data[s]
            res = extract_outline(img, part, coin, 25.0, params=prm)
            n += sum(len(p.primitives) for p in res.parts)
            devs.append(max(p.max_dev_mm for p in res.parts))
        L.append(f"{label:<44}{n:>5} primitives   dev {np.median(devs):.3f} / {np.percentile(devs, 90):.3f} / {max(devs):.3f} mm")


def counts(by):
    parts = by["parts"]
    return (sum(len(p["primitives"]) for p in parts), max(p["max_dev_mm"] for p in parts),
            sum(h["kind"] == "hole" for p in parts for h in p["holes"]), sum(h["kind"] == "slot" for p in parts for h in p["holes"]),
            [h for p in parts for h in p["holes"]])


def write_summary(recs, old):
    L, P = [], None
    P = L.append
    P(f"{'image':<26}{'prims before>after':>20}{'maxdev before>after':>22}{'h/s before>after':>18}{'LxS mm @25':>13}{'quality':>9}{'hand mm':>9}{'s':>6}")
    tot_b = tot_a = 0
    for k, r in recs.items():
        a = r["by_coin"]["25.0"]
        n, dev, nh, ns, _ = counts(a)
        o = old.get(k, {}).get("by_coin", {}).get("25.0")
        before = f"{o['n_primitives']:>4} > {n:<4}" if o else f"{'-':>4} > {n:<4}"
        devs = f"{o['max_dev_mm']:.3f} > {dev:.3f}" if o else f"{'-':>5} > {dev:.3f}"
        hs = f"{o['n_holes']}h/{o['n_slots']}s > {nh}h/{ns}s" if o else f"- > {nh}h/{ns}s"
        if o:
            tot_b, tot_a = tot_b + o["n_primitives"], tot_a + n
        L25 = max(a["parts"], key=lambda p: p["area_mm2"]) if False else None
        long_ = max(p["long_mm"] for p in a["parts"])
        hand = sum(p["hand_contact_mm"] for p in a["parts"])
        P(f"{k:<26}{before:>20}{devs:>22}{hs:>18}{long_:>13.1f}{a['quality']:>9.2f}{hand:>9.1f}{a['runtime_s']:>6.1f}")
    P(f"\nprimitives, 16 measuring photos: {tot_b} before, {tot_a} after. 'before' = outline_refine.py run (commit 9f12160, fit tol 0.07, snap 2 deg/0.07 mm);")
    P("'after' = backend/measure/outline.py (fit tol 0.10, segments <0.3 mm dropped, collinear merge, snap 3 deg within 0.15 mm).")
    P("maxdev = largest distance (mm) between the final primitives and the smoothed sub-pixel contour, both directions.")

    variant_table(L)
    P("\n== 1. glare false holes (hole/slot detail at coin 25.0; size = long x short mm) ==")
    for k in ("lock_edge_01", "lock_single_top_03", "lock_single_top_04", "lock_pair_sep_front_01"):
        o = old[k]["by_coin"]["25.0"]
        _, _, nh, ns, hs = counts(recs[k]["by_coin"]["25.0"])
        P(f"{k}: before {o['n_holes']}h/{o['n_slots']}s -> now {nh}h/{ns}s  " + "; ".join(f"{h['kind']} {h['long_mm']:.2f}x{h['short_mm']:.2f}" for h in hs))
    P("losses from the stricter rule: rejected on colour (warmer than paper / too dark) - see per-photo h/s columns above.")

    P("\n== repeat-shot consistency (holes/slots must match, sizes within 0.15 mm) ==")
    for a, b in PAIRS:
        for coin in COINS:
            _, _, ha, sa, la = counts(recs[a]["by_coin"][coin])
            _, _, hb, sb, lb = counts(recs[b]["by_coin"][coin])
            diffs = [max(abs(x["long_mm"] - y["long_mm"]), abs(x["short_mm"] - y["short_mm"])) for x, y in zip(la, lb)]
            ok = (ha, sa) == (hb, sb) and diffs and max(diffs) <= 0.15
            P(f"{a[5:]} vs {b[5:]} coin {coin}: {ha}h/{sa}s vs {hb}h/{sb}s; worst size diff {max(diffs) if diffs else float('nan'):.2f} mm -> {'OK' if ok else 'NOT OK'}")

    P("\n== hand photos (hand mask subtracted with a 0.7 mm margin; chroma refinement may not re-add hand pixels) ==")
    for k, r in recs.items():
        if "hand" not in k:
            continue
        for coin in ("25.0",):
            a = r["by_coin"][coin]
            n, dev, nh, ns, hs = counts(a)
            perim = sum(sum(_plen(q) for q in p["primitives"]) for p in a["parts"])
            contact = sum(p["hand_contact_mm"] for p in a["parts"])
            P(f"{k}: parts={len(a['parts'])} prims={n} maxdev={dev:.3f} perimeter={perim:.1f} mm, hand contact={contact:.1f} mm ({100 * contact / max(perim, 1e-6):.0f}% of the outline), "
              f"quality={a['quality']:.2f} {a['quality_parts']}; LxS={a['parts'][0]['long_mm']:.1f}x{a['parts'][0]['short_mm']:.1f} mm; holes/slots {nh}h/{ns}s; warnings: {a['warnings']}")
    P("Checked by eye on hand_01 and hand_02 (hand_03 not viewed): the outline follows the part wherever no finger touches it, and follows the finger")
    P("edge where a finger covers the part (hand_01: thumb at top right, finger at bottom; hand_02: top left and bottom). Those stretches are not part")
    P("edges, so the outline is NOT clean there; the module reports them as hand contact and lowers the quality score. Sizes from hand photos")
    P("(LxS above) include the cut-off stretches and are not part sizes.")
    text = "\n".join(L)
    (SEND / "summary.txt").write_text(text)
    print("\n" + text)


def _plen(q):
    import math
    if q["type"] == "line":
        return math.dist(q["p0"], q["p1"])
    return q["r"] * math.radians(q["sweep_deg"]) if q["type"] == "arc" else 2 * math.pi * q["r"]


if __name__ == "__main__":
    main()
