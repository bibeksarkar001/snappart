"""Phase 1 step 2, spike 2: SAM prompt ladder for the part, plus the coin mask.

Ladder: [Spark part_description, "<colour> part", "dark object"]; stops at the first match.
Reads out/spark_describe.json (run spark_describe.py first). Saves full-frame masks for later spikes.
"""
import asyncio
import json
import time

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

from sam_coin import segment
from spike_common import OUT, PHOTO, full_mask, save_overlay, tint


def components(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]), reverse=True)


async def first_mask(concept, tag):
    t = time.perf_counter()
    img, w, h, boxes, masks, _ = await segment(PHOTO, concept, f"sam_{tag}_raw.txt")
    dt = time.perf_counter() - t
    found = []
    for b in boxes:
        m = masks[b.object_id]
        raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
        found.append((b, full_mask(b, raster, w, h)))
    return img, found, dt


async def main():
    d = json.loads((OUT / "spark_describe.json").read_text())["test_01.jpg"]
    ladder = [d["part_description"], f"{d['colour']} part", "dark object"]
    print("ladder:", ladder)
    matched = None
    for rung, concept in enumerate(ladder, 1):
        img, found, dt = await first_mask(concept, f"ladder{rung}")
        print(f"rung {rung} {concept!r}: {len(found)} match(es) in {dt:.1f}s" + ("  -> MISS (empty response)" if not found else ""))
        for b, m in found:
            print(f"   box=({b.left},{b.top})-({b.right},{b.bottom}) conf={b.confidence:.3f} area={int((m > 0).sum())} px")
        if found:
            matched = (rung, concept, found)
            break
    if not matched:
        print("RESULT: no rung matched")
        return
    rung, concept, found = matched
    print(f"RESULT: matched on rung {rung} ({concept!r})")
    b, mask = max(found, key=lambda f: f[0].confidence or 0)
    comps = components(mask)
    print(f"connected components in chosen mask: {len(comps)}  areas(px)={comps[:5]}"
          f"  -> {'ONE connected part' if len(comps) == 1 else 'NOT a single connected part'}")
    # coin from the same photo
    _, coin_found, dt = await first_mask("coin", "coin")
    print(f"coin: {len(coin_found)} match(es) in {dt:.1f}s")
    cb, cmask = max(coin_found, key=lambda f: f[0].confidence or 0)
    print(f"   box=({cb.left},{cb.top})-({cb.right},{cb.bottom}) conf={cb.confidence:.3f}")
    overlap = int(((mask > 0) & (cmask > 0)).sum())
    print(f"part/coin mask overlap: {overlap} px")
    cv2.imwrite(str(OUT / "part_mask_sam.png"), mask)
    cv2.imwrite(str(OUT / "coin_mask_sam.png"), cmask)
    save_overlay(OUT / "ladder_overlay.png", tint(tint(img, mask, (0, 200, 0)), cmask, (0, 128, 255)))
    print("saved part_mask_sam.png, coin_mask_sam.png, ladder_overlay.png")


if __name__ == "__main__":
    asyncio.run(main())
