"""Real-photo accuracy test of backend/measure/outline.py against caliper truth, with the coin and with the bank card.

Photos: assets/private/accuracy_set/ (git-ignored, never committed) + meta.json. Live Muse Spark + SAM calls, cached in
out/accuracy/cache/ (git-ignored). Writes out/accuracy/SEND_TO_CLAUDE/{summary.txt, results.json, 4 compare PNGs}.
Usage: python backend/spikes/accuracy_set_test.py [--refresh]   (one process at a time)
"""
import argparse
import asyncio
import contextlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.outline import extract_outline, sample_primitives  # noqa: E402
from lock_set_test import spark  # noqa: E402  (Muse Spark photo check, JSON schema)
from sam_coin import segment, slug  # noqa: E402
from spike_common import OUT, ROOT, full_mask, tint  # noqa: E402

PHOTOS = ROOT / "assets" / "private" / "accuracy_set"
ACC = OUT / "accuracy"
CACHE = ACC / "cache"
SEND = ACC / "SEND_TO_CLAUDE"
CARD_MM = (85.60, 53.98)
COIN_MM = 25.0
TRUTH = {"A": (34.7, 34.0), "B": (51.6, 41.9)}      # (length, width) in mm, calipers; A is the smaller part, B the larger
COMPARE = ["acc_coin_top_front_01", "acc_coin_top_back_01", "acc_card_top_front_01", "acc_card_top_back_01"]
REF_LADDER = {"coin": ["coin", "gold coin", "round metal coin"],
              "card": ["credit card", "bank card", "dark rectangular card", "rectangular card"]}
MIN_PART_PX = 6000
for d in (CACHE, SEND):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- API stage (cached)

async def sam(path, concept, refresh=False):
    """[(confidence, full-frame 0/255 mask)]; empty = miss. Cached per photo and concept."""
    key = f"{path.stem}__{slug(concept)}"
    meta = CACHE / f"{key}.json"
    if meta.exists() and not refresh:
        return [(c, cv2.imread(str(CACHE / f"{key}_{i}.png"), 0)) for i, c in enumerate(json.loads(meta.read_text())["confs"])]
    with contextlib.redirect_stdout(io.StringIO()):
        _, w, h, boxes, masks, _ = await segment(path, concept, f"accuracy/cache/{key}_raw.txt")
    out = []
    for i, b in enumerate(boxes):
        m = masks[b.object_id]
        raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
        out.append((b.confidence or 0.0, full_mask(b, raster, w, h)))
        cv2.imwrite(str(CACHE / f"{key}_{i}.png"), out[-1][1])
    meta.write_text(json.dumps({"concept": concept, "confs": [c for c, _ in out]}))
    return out


def spark_cached(path, refresh):
    f = CACHE / f"{path.stem}__spark.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    desc, tokens, t = spark(path)
    f.write_text(json.dumps(desc))
    return desc


def ref_ok(kind, mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    if kind == "coin":
        (_, _), (a, b), _ = cv2.fitEllipse(c)
        return min(a, b) / max(a, b) > 0.85
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return 1.4 < max(w, h) / min(w, h) < 1.8


async def find_ref(path, kind, refresh):
    log = []
    for concept in REF_LADDER[kind]:
        found = [(c, m) for c, m in await sam(path, concept, refresh) if c >= 0.5 and ref_ok(kind, m)]
        log.append({"concept": concept, "usable_matches": len(found)})
        if found:
            conf, mask = max(found, key=lambda f: f[0])
            return mask, {"concept": concept, "conf": round(conf, 3), "ladder": log}
    return None, {"ladder": log}


def pick_components(found, ref):
    """Union of confident matches that do not overlap the reference, split into blobs; up to 2 largest, none touching the border."""
    keep = [m for c, m in found if c >= 0.5 and (m[ref > 0] > 0).sum() < 0.3 * (m > 0).sum()]
    if not keep:
        return []
    u = np.maximum.reduce(keep)
    n, lab, st, _ = cv2.connectedComponentsWithStats((u > 0).astype(np.uint8))
    h, w = u.shape
    good = [i for i in range(1, n) if st[i, 4] >= MIN_PART_PX and st[i, 0] > 0 and st[i, 1] > 0
            and st[i, 0] + st[i, 2] < w and st[i, 1] + st[i, 3] < h]
    good = sorted(good, key=lambda i: -st[i, 4])[:2]
    return [((lab == i) * 255).astype(np.uint8) for i in good]


async def find_parts(path, desc, ref, need, refresh):
    best, log = ("", []), []
    for concept in dict.fromkeys([desc["part_description"], f"{desc['colour']} plastic part", "brown plastic part", "dark object"]):
        comps = pick_components(await sam(path, concept, refresh), ref)
        log.append({"concept": concept, "components": len(comps)})
        if len(comps) > len(best[1]):
            best = (concept, comps)
        if len(comps) >= need:
            break
    return best[0], best[1], log


async def find_hand(path, refresh):
    for concept in ("hand", "human hand"):
        found = [m for c, m in await sam(path, concept, refresh) if c >= 0.5]
        if found:
            return np.maximum.reduce(found), concept
    return None, None


# ---------------------------------------------------------------- measuring

async def process(path, meta, refresh):
    stem = path.stem
    kind = "coin" if "_coin_" in stem else "card"
    view = "top" if "_top" in stem else "edge" if "_edge" in stem else "hand"
    img = cv2.imread(str(path))
    rec = {"photo": stem, "ref_kind": kind, "view": view}
    desc = spark_cached(path, refresh)
    rec["spark"] = desc
    ref, ref_info = await find_ref(path, kind, refresh)
    rec["reference"] = ref_info
    if ref is None:
        rec["error"] = "reference not found"
        return rec, None
    hand = hand_concept = None
    if view == "hand":
        hand, hand_concept = await find_hand(path, refresh)
        rec["hand"] = {"found": hand is not None, "concept": hand_concept,
                       "area_frac": None if hand is None else float((hand > 0).mean())}
    concept, comps, log = await find_parts(path, desc, ref, 1 if view == "hand" else 2, refresh)
    rec["part_ladder"], rec["part_concept"], rec["n_components"] = log, concept, len(comps)
    if not comps:
        rec["error"] = "part not found"
        return rec, {"img": img, "ref": ref, "comps": [], "hand": hand}
    union = np.maximum.reduce(comps)
    h, w = img.shape[:2]
    diag = math.hypot(w, h)
    try:
        t = time.perf_counter()
        res = extract_outline(img, union, ref, COIN_MM if kind == "coin" else CARD_MM, ref_kind=kind, exclude_mask=hand)
        rec["t_outline_s"] = round(time.perf_counter() - t, 2)
    except ValueError as e:
        rec["error"] = f"extract_outline: {e}"
        return rec, {"img": img, "ref": ref, "comps": comps, "hand": hand}
    f35 = meta["focal_35mm_equiv"]
    f_px = f35 / 43.27 * diag
    rec.update(mm_per_px=res.mm_per_px, tilt_deg=res.tilt_deg, f_px=f_px, distance_mm=f_px * res.mm_per_px,
               quality=res.quality, quality_parts=res.quality_parts, warnings=res.warnings)
    parts = sorted(res.parts, key=lambda p: -p.area_mm2)
    rec["parts"] = {}
    for i, p in enumerate(parts):
        label = ("B", "A")[i] if view == "top" and len(parts) == 2 else f"p{i + 1}"
        d = {"long_mm": p.long_mm, "short_mm": p.short_mm, "area_mm2": p.area_mm2, "max_dev_mm": p.max_dev_mm,
             "n_primitives": len(p.primitives), "holes": [h_.__dict__ | {"primitives": None} for h_ in p.holes],
             "hand_contact_mm": p.hand_contact_mm, "primitives": p.primitives}
        if label in TRUTH:
            tl, ts = TRUTH[label]
            ls, ss = sorted(TRUTH[label], reverse=True)
            d["truth"] = {"long": ls, "short": ss}
            d["err_long_mm"], d["err_short_mm"] = p.long_mm - ls, p.short_mm - ss
            d["err_long_pct"], d["err_short_pct"] = 100 * (p.long_mm - ls) / ls, 100 * (p.short_mm - ss) / ss
            d["err_mean_mm"] = (p.long_mm + p.short_mm) / 2 - (ls + ss) / 2
        rec["parts"][label] = d
    if view == "top" and len(parts) != 2:
        rec["error"] = f"expected 2 parts, got {len(parts)}"
    return rec, {"img": img, "ref": ref, "comps": comps, "hand": hand, "res": res}


# ---------------------------------------------------------------- analysis

def fit_models(rows):
    """rows: [(photo, part, dim, truth, measured)]. Models predict measured from truth; every fit is least squares on mm."""
    def fit(group_of):
        keys = sorted({group_of(r) for r in rows})
        k = {g: sum(r[3] * r[4] for r in rows if group_of(r) == g) / sum(r[3] ** 2 for r in rows if group_of(r) == g) for g in keys}
        return k
    models = {"none (k=1)": (0, lambda r: "-"), "uniform k": (1, lambda r: "all"),
              "per part (A, B)": (2, lambda r: r[1]), "per dimension (long, short)": (2, lambda r: r[2])}
    out = {}
    photos = sorted({r[0] for r in rows})
    for name, (npar, g) in models.items():
        k = {"-": 1.0} if npar == 0 else fit(g)
        rms = math.sqrt(sum((r[4] - k[g(r)] * r[3]) ** 2 for r in rows) / len(rows))
        loo = []
        for ph in photos:
            tr = [r for r in rows if r[0] != ph]
            kk = {"-": 1.0} if npar == 0 else {gg: sum(r[3] * r[4] for r in tr if g(r) == gg) / sum(r[3] ** 2 for r in tr if g(r) == gg)
                                               for gg in {g(r) for r in tr}}
            loo += [(r[4] - kk.get(g(r), kk[next(iter(kk))]) * r[3]) ** 2 for r in rows if r[0] == ph]
        out[name] = {"n_params": npar, "k": k, "rms_mm": rms, "loo_rms_mm": math.sqrt(sum(loo) / len(loo))}
    return out


def top_rows(recs):
    rows = []
    for r in recs:
        for lab, p in r.get("parts", {}).items():
            if "truth" in p:
                rows += [(r["photo"], lab, "long", p["truth"]["long"], p["long_mm"]), (r["photo"], lab, "short", p["truth"]["short"], p["short_mm"])]
    return rows


# ---------------------------------------------------------------- compare images

def pixelate(img, mask, block=40):
    small = cv2.resize(img, (img.shape[1] // block, img.shape[0] // block), interpolation=cv2.INTER_AREA)
    mos = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
    out = img.copy()
    out[m] = mos[m]
    return out


def compare_image(rec, ctx, path):
    img, res = ctx["img"], ctx["res"]
    base = pixelate(img, ctx["ref"]) if rec["ref_kind"] == "card" else img       # card face carries a name: pixelated in outputs
    left = tint(base, ctx["ref"], (255, 128, 0))
    for m, col in zip(ctx["comps"], [(0, 200, 0), (0, 128, 255)]):
        left = tint(left, m, col)
    right = base.copy()
    t = 3
    for lab, p in zip(rec["parts"], sorted(res.parts, key=lambda p: -p.area_mm2)):
        pts = sample_primitives(p.primitives, 0.05)
        px = np.round(res.frame.to_px(pts)).astype(np.int32)
        cv2.polylines(right, [px.reshape(-1, 1, 2)], False, (0, 0, 255), t, cv2.LINE_AA)
        x, y = px.min(0)
        d = rec["parts"][lab]
        txt = f"{lab} {d['long_mm']:.1f} x {d['short_mm']:.1f}" + (f" (truth {d['truth']['long']} x {d['truth']['short']})" if "truth" in d else "")
        cv2.putText(right, txt, (max(int(x) - 40, 10), max(int(y) - 25, 60)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(right, txt, (max(int(x) - 40, 10), max(int(y) - 25, 60)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    s = 800 / img.shape[1]
    both = np.hstack([cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for v in (left, right)])
    cv2.imwrite(str(path), both)


# ---------------------------------------------------------------- summary

def fmt(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def write_summary(recs, models, rep, vs, par, meta_note):
    L = []
    top = [r for r in recs if r["view"] == "top"]
    L.append("SnapPart accuracy test, production module backend/measure/outline.py, real photos (iPhone 15), caliper truth.")
    L.append("Measured numbers only. Truth (calipers): A length 34.7 width 34.0; B length 51.6 width 41.9; coin 25.0 mm; card 85.60 x 53.98 mm.")
    L.append("Our 'long/short' = minimum-area rectangle of the fitted outline. A's two truths differ by 0.7 mm, so which of ours is 'length' can swap:")
    L.append("the 'mean' column (average of both sides, pairing-free) is the safe one for A.")
    L.append(f"A = smaller part by area, B = larger. {meta_note}")
    L.append("")
    L.append("1. TOP PHOTOS: per part, ours vs truth (mm, error mm, error %)")
    L.append(f"{'photo':24}{'ref':5}{'mm/px':>8}{'tilt':>6}{'q':>5} | {'part':4}{'long':>7}{'err':>7}{'%':>6}{'short':>7}{'err':>7}{'%':>6}{'mean err':>9}{'holes':>6}{'prims':>6}{'maxdev':>7}")
    for r in top:
        if "parts" not in r:
            L.append(f"{r['photo']:24}{r['ref_kind']:5} ERROR {r.get('error')}")
            continue
        for lab, p in r["parts"].items():
            L.append(f"{r['photo']:24}{r['ref_kind']:5}{r['mm_per_px']:8.5f}{fmt(r['tilt_deg'], 1):>6}{r['quality']:5.2f} | {lab:4}"
                     f"{p['long_mm']:7.2f}{fmt(p.get('err_long_mm')):>7}{fmt(p.get('err_long_pct'), 1):>6}{p['short_mm']:7.2f}"
                     f"{fmt(p.get('err_short_mm')):>7}{fmt(p.get('err_short_pct'), 1):>6}{fmt(p.get('err_mean_mm')):>9}"
                     f"{len(p['holes']):6}{p['n_primitives']:6}{p['max_dev_mm']:7.3f}")
        for w in r["warnings"]:
            L.append(f"    warning: {w}")
        if r.get("error"):
            L.append(f"    ERROR: {r['error']}")
    L.append("")
    L.append("Reference and part ladders (SAM concept that worked; Spark part description):")
    for r in recs:
        L.append(f"  {r['photo']:24} ref '{r['reference'].get('concept')}' conf {r['reference'].get('conf')} | parts '{r.get('part_concept')}' "
                 f"({r.get('n_components')} blobs) | spark: {r['spark']['part_description']!r}, issues {r['spark']['image_quality_issues']}")
    L.append("")
    L.append("2. REPEATABILITY, coin_top_front_01..03 (same pose, three shots)")
    for lab, d in rep.items():
        L.append(f"  {lab}: " + "  ".join(f"{k} {v['vals']} range {v['range']:.2f} sd {v['sd']:.2f}" for k, v in d.items()))
    L.append("")
    L.append("3. COIN vs CARD (all numbers: mean over that reference's top photos; signed error = ours - truth)")
    for k, v in vs["agg"].items():
        L.append(f"  {k:6} n={v['n']}: mean signed error {v['mean_signed']:+.2f} mm ({v['mean_signed_pct']:+.1f} %), mean abs error {v['mean_abs']:.2f} mm, "
                 f"worst {v['worst']:.2f} mm | mm/px mean {v['mm_per_px']:.5f}, distance mean {v['distance_mm']:.0f} mm")
    L.append("  same-view pairs (different shots, parts re-placed):")
    for a, b, rows in vs["pairs"]:
        L.append(f"    {a} vs {b}")
        for row in rows:
            L.append(f"      {row['part']:2} {row['dim']:5} truth {row['truth']:5.1f}  coin {row['coin']:6.2f} ({row['coin_err']:+.2f})  card {row['card']:6.2f} ({row['card_err']:+.2f})")
    L.append("")
    L.append("4. CAMERA DISTANCE (f_px = f35/43.27 * diagonal_px; distance = f_px * mm_per_px; stated 440-450 mm)")
    L.append(f"   f35 = {meta_note.split('f35=')[1].split(';')[0]}, image diagonal 5040 px.")
    for k, v in vs["agg"].items():
        L.append(f"  {k}: " + ", ".join(f"{r['photo'][4:]} {r['distance_mm']:.0f}" for r in top if r["ref_kind"] == k and "distance_mm" in r))
    L.append("")
    L.append("5. PARALLAX / SCALE-CORRECTION MODELS (measured = k * truth, fitted to the photo x part x dimension rows; rms in mm)")
    L.append("   k = d/(d-h) for a part of height h at camera distance d, so extra size = h/(d-h) and h = d*(1-1/k).")
    for ref, mm in models.items():
        L.append(f"  {ref} photos ({len({r[0] for r in top_rows([x for x in top if x['ref_kind'] == ref])})} photos, {len(top_rows([x for x in top if x['ref_kind'] == ref]))} rows):")
        for name, m in mm.items():
            ks = ", ".join(f"{g}={v:.4f}" for g, v in m["k"].items())
            L.append(f"    {name:30} params {m['n_params']}  k: {ks:40} rms {m['rms_mm']:.3f}  leave-one-photo-out rms {m['loo_rms_mm']:.3f}")
        L.append("    implied height h (mm) from k, at the measured distance: " + par[ref])
    L.append("   Limits: only 4 distinct truth values exist (A and B, length and width), so at most 2 parameters are supportable; the rms above is")
    L.append("   against repeat noise of about 0.02-0.05 mm (section 2), so any rms above that is a systematic error the model did not remove.")
    L.append("   Edge-photo heights are silhouette short sides, not measured thickness. The coin ellipse gives 7-8.5 deg 'tilt' (card mode gives none).")
    L.append("   Reasoning, NOT tested here: a flat plane facing the sensor images a coin as a circle, so an ellipse means the paper is tilted against the sensor;")
    L.append("   a tilt changes the scale across the field (parts sit tens of mm from the coin), and the ellipse cannot tell the sign of that gradient.")
    L.append("")
    L.append("6. EDGE PHOTOS: silhouette sizes only (not accuracy; scale from the reference seen face-on)")
    for r in recs:
        if r["view"] == "edge":
            if "parts" not in r:
                L.append(f"  {r['photo']}: {r.get('error')}")
                continue
            L.append(f"  {r['photo']}: " + "; ".join(f"{lab} {p['long_mm']:.1f} x {p['short_mm']:.1f} mm (area {p['area_mm2']:.0f} mm2)" for lab, p in r["parts"].items())
                     + f"  [mm/px {r['mm_per_px']:.5f}, ref conf {r['reference'].get('conf')}]")
    L.append("")
    L.append("7. HAND PHOTOS: hand check only")
    for r in recs:
        if r["view"] == "hand":
            h = r.get("hand", {})
            c = sum(p["hand_contact_mm"] for p in r.get("parts", {}).values())
            L.append(f"  {r['photo']}: SAM hand found {h.get('found')} ({h.get('concept')}, {100 * (h.get('area_frac') or 0):.1f} % of frame); "
                     f"part blobs {r.get('n_components')}; hand contact along outline {c:.1f} mm; "
                     f"{r.get('error') or 'quality ' + format(r['quality'], '.2f')}")
    (SEND / "summary.txt").write_text("\n".join(L) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    refresh = ap.parse_args().refresh
    meta = json.loads((PHOTOS / "meta.json").read_text())
    recs, ctxs = [], {}
    for path in sorted(PHOTOS.glob("*.jpg")):
        t = time.perf_counter()
        rec, ctx = await process(path, meta[path.name], refresh)
        rec["t_total_s"] = round(time.perf_counter() - t, 1)
        recs.append(rec)
        ctxs[path.stem] = ctx
        print(f"{path.stem:24} {rec.get('error') or 'ok':30} parts {rec.get('n_components')} {rec['t_total_s']}s", flush=True)

    top = [r for r in recs if r["view"] == "top" and "parts" in r]
    # repeatability
    front = [r for r in top if r["ref_kind"] == "coin" and r["photo"].startswith("acc_coin_top_front")]
    rep = {}
    for lab in ("B", "A"):
        rep[lab] = {}
        for dim in ("long_mm", "short_mm"):
            v = [r["parts"][lab][dim] for r in front if lab in r["parts"]]
            rep[lab][dim] = {"vals": [round(x, 2) for x in v], "range": max(v) - min(v), "sd": statistics.pstdev(v)}
        v = [r["mm_per_px"] for r in front]
        rep.setdefault("mm_per_px", {})["mm_per_px"] = {"vals": [round(x, 5) for x in v], "range": max(v) - min(v), "sd": statistics.pstdev(v)}
    # coin vs card
    agg = {}
    for kind in ("coin", "card"):
        rs = [r for r in top if r["ref_kind"] == kind]
        errs = [(p[f"err_{d}_mm"], p[f"err_{d}_pct"]) for r in rs for p in r["parts"].values() if "truth" in p for d in ("long", "short")]
        agg[kind] = {"n": len(rs), "mean_signed": statistics.mean(e for e, _ in errs), "mean_signed_pct": statistics.mean(e for _, e in errs),
                     "mean_abs": statistics.mean(abs(e) for e, _ in errs), "worst": max(abs(e) for e, _ in errs),
                     "mm_per_px": statistics.mean(r["mm_per_px"] for r in rs), "distance_mm": statistics.mean(r["distance_mm"] for r in rs)}
    pairs = []
    for view in ("front", "back"):
        a = next(r for r in top if r["photo"] == f"acc_coin_top_{view}_01")
        b = next(r for r in top if r["photo"] == f"acc_card_top_{view}_01")
        rows = [{"part": lab, "dim": dim, "truth": a["parts"][lab]["truth"][dim], "coin": a["parts"][lab][f"{dim}_mm"], "card": b["parts"][lab][f"{dim}_mm"],
                 "coin_err": a["parts"][lab][f"err_{dim}_mm"], "card_err": b["parts"][lab][f"err_{dim}_mm"]}
                for lab in ("B", "A") for dim in ("long", "short")]
        pairs.append((a["photo"], b["photo"], rows))
    # parallax
    models, par = {}, {}
    for kind in ("coin", "card"):
        rows = top_rows([r for r in top if r["ref_kind"] == kind])
        models[kind] = fit_models(rows)
        edge = [r for r in recs if r["view"] == "edge" and r["ref_kind"] == kind and len(r.get("parts", {})) == 2]
        d = statistics.mean(r["distance_mm"] for r in top if r["ref_kind"] == kind)
        if edge:    # zero fitted parameters: h = short side of the edge-photo silhouette (larger blob = B, smaller = A)
            h = {"B": statistics.mean(r["parts"]["p1"]["short_mm"] for r in edge), "A": statistics.mean(r["parts"]["p2"]["short_mm"] for r in edge)}
            k = {g: d / (d - v) for g, v in h.items()}
            rms = math.sqrt(sum((r[4] - k[r[1]] * r[3]) ** 2 for r in rows) / len(rows))
            models[kind]["edge-photo heights (0 fitted)"] = {"n_params": 0, "k": k, "rms_mm": rms, "loo_rms_mm": rms, "h_mm": h}
        d = statistics.mean(r["distance_mm"] for r in top if r["ref_kind"] == kind)
        k = models[kind]["per part (A, B)"]["k"]
        ku = models[kind]["uniform k"]["k"]["all"]
        par[kind] = (f"d={d:.0f} mm; per part " + ", ".join(f"{p}: k={v:.4f} -> h={d * (1 - 1 / v):.1f} mm" for p, v in k.items())
                     + f"; uniform k={ku:.4f} -> h={d * (1 - 1 / ku):.1f} mm")
    meta_note = f"f35={next(iter(meta.values()))['focal_35mm_equiv']}; focal and size come from meta.json (the jpgs carry no EXIF); stated height 44-45 cm"
    vs = {"agg": agg, "pairs": pairs}
    write_summary(recs, models, rep, vs, par, meta_note)
    for stem in COMPARE:
        ctx = ctxs.get(stem)
        rec = next(r for r in recs if r["photo"] == stem)
        if ctx and "res" in ctx:
            compare_image(rec, ctx, SEND / f"{stem}_compare.png")
    (SEND / "results.json").write_text(json.dumps({"truth": TRUTH, "photos": recs, "repeatability": rep, "coin_vs_card": vs,
                                                   "models": models, "implied_height": par}, indent=1, default=float))
    print((SEND / "summary.txt").read_text())


if __name__ == "__main__":
    asyncio.run(main())
