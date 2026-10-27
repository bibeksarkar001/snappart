"""Phase 1 spike: run the measuring pipeline over the private lock_set photos (assets/private/lock_set, git-ignored).

Per measuring image: Muse Spark description + SAM coin + SAM part ladder (API results are cached in out/lock_set/),
ellipse-fit the coin, un-stretch the tilt, OpenCV colour segmentation of the part, mm outline + holes/slots, all for a
25.0 mm and a 23.0 mm coin. hand_* images only get the hand checks. Prints one summary table; writes out/lock_set/*.

Usage: python backend/spikes/lock_set_test.py [--refresh]    (--refresh ignores the API cache)
Photos are only ever sent to our own Meta API calls. Nothing here is an accuracy claim.
"""
import argparse
import asyncio
import contextlib
import io
import json
import math
import statistics
import time

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

import spark_describe
from sam_coin import segment
from spike_common import OUT, ROOT, full_mask, save_overlay, tint, unstretch_fn

PHOTOS = ROOT / "assets" / "private" / "lock_set"
OUTD = OUT / "lock_set"
OUTD.mkdir(exist_ok=True)
COINS = {"25.0": 25.0, "23.0": 23.0}
MIN_PART_PX = 6000           # ignore specks smaller than this (about 0.05% of the frame)
MIN_HOLE_MM2 = 2.0           # inner contours smaller than this are treated as noise (at either coin size)
SLOT_ELONGATION = 2.0        # inner contour long/short side above this = slot, else hole
CORNER = (0.88, 0.80)        # bottom-right region (x>, y>) where the dark object sits, as fractions of the frame
# Which single_top shots show which part. A visual call by Claude, NOT from the user: ribbed curved face visible (A) vs
# frame with crossbar and no ribbed face visible (B). Confirm with whoever shot the photos.
GROUPS = {
    "A (ribbed face visible)": ["lock_single_top_02", "lock_single_top_03", "lock_single_top_04"],
    "B (frame with crossbar)": ["lock_single_top_01", "lock_single_top_05", "lock_single_top_06"],
}


# ---------- API stage (cached) ----------

async def sam(path, concept, tag):
    """Quiet SAM call. Returns ([(confidence, full-frame mask)], seconds). Empty list = miss."""
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _, w, h, boxes, masks, _ = await segment(path, concept, f"lock_set/{path.stem}_{tag}_raw.txt")
    out = []
    for b in boxes:
        m = masks[b.object_id]
        raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
        out.append((b.confidence or 0.0, full_mask(b, raster, w, h)))
    return out, time.perf_counter() - t


def spark(path):
    t = time.perf_counter()
    r = spark_describe.client.chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": spark_describe.PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{spark_describe.jpeg_b64(path)}"}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "part_check", "schema": spark_describe.SCHEMA}},
    )
    return json.loads(r.choices[0].message.content), r.usage.total_tokens, time.perf_counter() - t


def union(found):
    keep = [m for c, m in found if c >= 0.5]
    return np.maximum.reduce(keep) if keep else None


async def api_stage(path, refresh):
    meta_path = OUTD / f"{path.stem}.meta.json"
    if meta_path.exists() and not refresh:
        return json.loads(meta_path.read_text())
    is_hand = "hand" in path.stem
    desc, tokens, t_spark = await asyncio.to_thread(spark, path)
    side, t_side = await sam(path, "hand" if is_hand else "coin", "hand" if is_hand else "coin")
    meta = {"spark": desc, "spark_tokens": tokens, "t_spark": t_spark, "ladder": []}
    if side:
        best = max(side, key=lambda f: f[0])
        cv2.imwrite(str(OUTD / f"{path.stem}_{'hand' if is_hand else 'coin'}.png"), best[1])
        meta["side_conf"] = best[0]
    meta["side_found"] = bool(side)
    meta["t_side"] = t_side
    t = time.perf_counter()
    for concept in [desc["part_description"], f"{desc['colour']} part", "dark object"]:
        found, _ = await sam(path, concept, "part")
        meta["ladder"].append({"concept": concept, "matches": len(found)})
        if found:
            cv2.imwrite(str(OUTD / f"{path.stem}_part_sam.png"), union(found))
            meta["part_confs"] = [round(c, 3) for c, _ in found]
            break
    meta["t_part"] = time.perf_counter() - t
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


# ---------- OpenCV stage ----------

def cv_part_mask(img, coin):
    """Brown part(s): saturated pixels (coin removed), reflective patches filled from inside, border-touching blobs dropped."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (0, 0), 3)
    v = cv2.GaussianBlur(hsv[..., 2], (0, 0), 3)
    thr = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    core = (s > thr).astype(np.uint8) * 255
    if coin is not None:
        core[cv2.dilate(coin, np.ones((25, 25), np.uint8)) > 0] = 0
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside = np.zeros_like(core)
    cv2.drawContours(inside, cnts, -1, 255, cv2.FILLED)
    near = cv2.dilate(inside, np.ones((81, 81), np.uint8)) > 0
    if coin is not None:
        near |= cv2.dilate(coin, np.ones((25, 25), np.uint8)) > 0
    paper_v = float(np.median(v[~near]))
    mask = core | (((inside > 0) & (v < 0.8 * paper_v)).astype(np.uint8) * 255)  # grey glints on the part, not slots (paper is bright)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a >= MIN_PART_PX and x > 0 and y > 0 and x + bw < w and y + bh < h:  # blobs touching the frame edge are not parts
            keep[lab == i] = 255
    return keep, thr, paper_v


def iou(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def filled(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def poly_area(p):
    return abs(cv2.contourArea(p.astype(np.float32)))


def rect_dims(p):
    (_, _), (rw, rh), _ = cv2.minAreaRect(p.astype(np.float32))
    return max(rw, rh), min(rw, rh)


def measure(mask, fix, major, coin_mm):
    """mm outline of every part in a 0/255 mask. Returns the per-image metrics and the outline polygons (mm)."""
    mm_per_px = coin_mm / major
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    parts, all_pts = [], []
    for i, c in enumerate(cnts):
        if hier[0][i][3] >= 0:
            continue
        pts = fix(c) * mm_per_px
        all_pts.append(pts)
        holes, slots, hole_polys = 0, 0, []
        j = hier[0][i][2]
        while j >= 0:
            hp = fix(cnts[j]) * mm_per_px
            if poly_area(hp) >= MIN_HOLE_MM2:
                (rl, rs) = rect_dims(hp)
                slots, holes = (slots + 1, holes) if rl / max(rs, 1e-6) > SLOT_ELONGATION else (slots, holes + 1)
                hole_polys.append(hp)
            j = hier[0][j][0]
        l, s = rect_dims(pts)
        parts.append({"long_mm": l, "short_mm": s, "area_mm2": poly_area(pts), "holes": holes, "slots": slots,
                      "outline": cv2.approxPolyDP(pts.astype(np.float32), 0.15, True).reshape(-1, 2).tolist(),
                      "hole_outlines": [cv2.approxPolyDP(h.astype(np.float32), 0.05, True).reshape(-1, 2).tolist() for h in hole_polys]})
    if not parts:
        return {"n_parts": 0}
    pts = np.vstack(all_pts)
    l, s = rect_dims(cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2))
    return {"n_parts": len(parts), "long_mm": l, "short_mm": s, "aabb_mm": [float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))],
            "area_mm2": sum(p["area_mm2"] for p in parts), "holes": sum(p["holes"] for p in parts),
            "slots": sum(p["slots"] for p in parts), "parts": parts}


def corner_px(mask):
    h, w = mask.shape
    return int((mask[int(CORNER[1] * h):, int(CORNER[0] * w):] > 0).sum())


def process_measuring(path, meta):
    t = time.perf_counter()
    img = cv2.imread(str(path))
    rec = {"image": path.stem, "spark": meta["spark"], "ladder": meta["ladder"], "part_confs": meta.get("part_confs"),
           "coin_found": meta["side_found"], "part_found": "part_confs" in meta}
    coin = cv2.imread(str(OUTD / f"{path.stem}_coin.png"), 0) if meta["side_found"] else None
    sam_part = cv2.imread(str(OUTD / f"{path.stem}_part_sam.png"), 0) if rec["part_found"] else None
    if coin is None:
        rec["error"] = "no coin found"
        return rec, img
    fix, major, minor, tilt, ell = unstretch_fn(coin)
    cvm, thr, paper_v = cv_part_mask(img, coin)
    rec.update(coin_major_px=major, coin_minor_px=minor, tilt_deg=tilt, tilt_flag=tilt > 5.0, sat_otsu=float(thr), paper_v=paper_v)
    rec["by_coin"] = {k: {"px_per_mm": major / mm, **measure(cvm, fix, major, mm)} for k, mm in COINS.items()}
    rec["iou_raw"] = iou(cvm, sam_part) if sam_part is not None else None
    rec["iou_filled"] = iou(filled(cvm), filled(sam_part)) if sam_part is not None else None
    rec["corner_px_cv"] = corner_px(cvm)
    rec["corner_px_sam"] = corner_px(sam_part) if sam_part is not None else None
    rec["cv_runtime_s"] = time.perf_counter() - t
    # overlay: SAM tint green, CV outline red, holes blue, coin ellipse orange, corner box yellow
    ov = img if sam_part is None else tint(img, sam_part, (0, 200, 0), 0.35)
    cnts, hier = cv2.findContours(cvm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(cnts):
        cv2.drawContours(ov, [c], -1, (0, 0, 255) if hier[0][i][3] < 0 else (255, 0, 0), 4)
    cv2.ellipse(ov, ell, (0, 160, 255), 5)
    h, w = img.shape[:2]
    cv2.rectangle(ov, (int(CORNER[0] * w), int(CORNER[1] * h)), (w - 1, h - 1), (0, 255, 255), 5)
    save_overlay(OUTD / f"{path.stem}_overlay.png", ov)
    out = {k: {kk: vv for kk, vv in v.items() if kk != "parts"} | {"parts": v.get("parts", [])} for k, v in rec["by_coin"].items()}
    (OUTD / f"{path.stem}_outline_mm.json").write_text(json.dumps(out))
    return rec, img


async def process_hand(path, meta):
    rec = {"image": path.stem, "spark": meta["spark"], "ladder": meta["ladder"]}
    hand = cv2.imread(str(OUTD / f"{path.stem}_hand.png"), 0) if meta["side_found"] else None
    part = cv2.imread(str(OUTD / f"{path.stem}_part_sam.png"), 0) if "part_confs" in meta else None
    rec["sam_hand_found"], rec["sam_hand_conf"] = meta["side_found"], meta.get("side_conf")
    rec["part_found"] = part is not None
    if hand is not None and part is not None:
        rec["hand_part_overlap_px"] = int(((cv2.dilate(hand, np.ones((31, 31), np.uint8)) > 0) & (part > 0)).sum())
    return rec


# ---------- reporting ----------

def stats(vals):
    return {"n": len(vals), "min": min(vals), "max": max(vals), "range": max(vals) - min(vals), "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    refresh = ap.parse_args().refresh
    photos = sorted(PHOTOS.glob("lock_*.jpg"))
    recs = {}

    async def run():
        for p in photos:
            t = time.perf_counter()
            meta = await api_stage(p, refresh)
            if "hand" in p.stem:
                recs[p.stem] = await process_hand(p, meta)
            else:
                recs[p.stem], _ = process_measuring(p, meta)
            recs[p.stem]["api_runtime_s"] = meta["t_spark"] + meta["t_side"] + meta["t_part"]
            recs[p.stem]["wall_s"] = time.perf_counter() - t
            print(f"done {p.stem} ({recs[p.stem]['wall_s']:.1f}s)", flush=True)
    asyncio.run(run())
    (OUTD / "results.json").write_text(json.dumps(recs, indent=1, default=float))

    lines = []
    P = lines.append
    P(f"{'image':<24}{'tilt':>6}{'px/mm25':>9}{'px/mm23':>9}  {'LxS mm @25':<13}{'LxS mm @23':<13}{'holes/slots':>12}{'IoU raw/fill':>14}  {'rung':<5}{'sam+spark s':>12}{'cv s':>6}")
    for k, r in recs.items():
        if "hand" in k:
            continue
        if "error" in r or r.get("by_coin", {}).get("25.0", {}).get("n_parts", 0) == 0:
            P(f"{k:<24}  ERROR: {r.get('error', 'no part found by OpenCV')}")
            continue
        a, b = r["by_coin"]["25.0"], r["by_coin"]["23.0"]
        rung = next((i + 1 for i, s in enumerate(r["ladder"]) if s["matches"]), "miss")
        iou_txt = f"{r['iou_raw']:.3f}/{r['iou_filled']:.3f}" if r["iou_raw"] is not None else "-"
        P(f"{k:<24}{r['tilt_deg']:>5.1f}°{a['px_per_mm']:>9.2f}{b['px_per_mm']:>9.2f}  {a['long_mm']:>5.1f}x{a['short_mm']:<6.1f}  {b['long_mm']:>5.1f}x{b['short_mm']:<6.1f}"
          f"{a['holes']:>6}h/{a['slots']}s{iou_txt:>14}  {rung!s:<5}{r['api_runtime_s']:>12.1f}{r['cv_runtime_s']:>6.2f}")
    P("\nparts found by OpenCV per image (outlines): " + ", ".join(f"{k.replace('lock_', '')}={r['by_coin']['25.0']['n_parts']}"
                                                                 for k, r in recs.items() if "by_coin" in r and "hand" not in k))

    P("\n== repeatability (single_top, long x short side of the minimum-area rectangle, mm) ==")
    for g, names in GROUPS.items():
        for coin in COINS:
            for field in ("long_mm", "short_mm", "area_mm2"):
                s = stats([recs[n]["by_coin"][coin][field] for n in names if n in recs and "by_coin" in recs[n]])
                P(f"group {g} coin {coin}: {field:<8} n={s['n']} min={s['min']:.2f} max={s['max']:.2f} range={s['range']:.2f} std={s['std']:.2f} mean={s['mean']:.2f}")
    if not GROUPS:
        P("no groups defined yet")

    P("\n== failure cases ==")
    for k, r in recs.items():
        if "pair_clipped" in k and "by_coin" in r:
            a = r["by_coin"]["25.0"]
            P(f"{k}: OpenCV outlines={a['n_parts']} area={a['area_mm2']:.0f} mm2 rect={a['long_mm']:.1f}x{a['short_mm']:.1f} mm, SAM matches={(r['part_confs'] or [])}")
    sep = [recs[k]["by_coin"]["25.0"] for k in recs if "pair_sep" in k and "by_coin" in recs[k]]
    if sep:
        P("pair_sep (separate parts) outlines: " + ", ".join(f"{x['n_parts']} outlines, total area {x['area_mm2']:.0f} mm2" for x in sep) + "  (coin 25.0)")
    P("corner object (mask pixels inside the bottom-right box, CV / SAM): "
      + ", ".join(f"{k.replace('lock_', '')}={r['corner_px_cv']}/{r['corner_px_sam']}" for k, r in recs.items() if "corner_px_cv" in r))
    for k, r in recs.items():
        if "hand" in k:
            P(f"{k}: Spark hand_touching_part={r['spark']['hand_touching_part']} issues={r['spark']['image_quality_issues']}; "
              f"SAM hand found={r['sam_hand_found']} conf={r['sam_hand_conf']}; part found={r['part_found']} "
              f"hand/part overlap (hand dilated 15px)={r.get('hand_part_overlap_px')} px")

    P("\n== Spark 'camera tilted' vs measured coin tilt ==")
    for k, r in recs.items():
        if "tilt_deg" in r:
            P(f"{k}: measured {r['tilt_deg']:.1f} deg (flag>5: {r['tilt_flag']}), Spark issues={r['spark']['image_quality_issues']}")

    P("\n== edge_* silhouette (min-area rectangle short side = 'height'; same coin scale, not corrected for perspective) ==")
    for k, r in recs.items():
        if "edge" in k and "by_coin" in r:
            a, b = r["by_coin"]["25.0"], r["by_coin"]["23.0"]
            P(f"{k}: outlines={a['n_parts']} rect {a['long_mm']:.1f}x{a['short_mm']:.1f} mm @25 | {b['long_mm']:.1f}x{b['short_mm']:.1f} mm @23; "
              f"aabb {a['aabb_mm'][0]:.1f}x{a['aabb_mm'][1]:.1f} @25")
    text = "\n".join(lines)
    (OUTD / "summary.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
