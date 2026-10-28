"""Phase 1 spike: one SAM call per hand_* photo for the coin; masks cached in out/lock_set/refined/hand_cache/ (git-ignored)."""
import asyncio
import time

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

from sam_coin import segment
from spike_common import OUT, ROOT, full_mask

PHOTOS = ROOT / "assets" / "private" / "lock_set"
CACHE = OUT / "lock_set" / "refined" / "hand_cache"


async def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for path in sorted(PHOTOS.glob("lock_hand_*.jpg")):
        out = CACHE / f"{path.stem}_coin.png"
        if out.exists():
            print(f"{path.stem}: cached")
            continue
        t = time.perf_counter()
        _, w, h, boxes, masks, _ = await segment(path, "coin", f"lock_set/refined/hand_cache/{path.stem}_coin_raw.txt")
        if not boxes:
            print(f"{path.stem}: NO COIN FOUND")
            continue
        b = max(boxes, key=lambda b: b.confidence or 0)
        m = masks[b.object_id]
        raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
        cv2.imwrite(str(out), full_mask(b, raster, w, h))
        print(f"{path.stem}: coin box=({b.left},{b.top})-({b.right},{b.bottom}) conf={b.confidence:.3f} {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
