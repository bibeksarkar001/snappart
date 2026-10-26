"""Spike 3: SAM 3.1 via Responses API (streaming), parsed with meta_sam_parser.

Usage: python backend/spikes/sam_coin.py [--part "name of the part"]
"""
import argparse
import asyncio
import base64
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import decode_mask_to_raster, image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
GOLDEN = ROOT / "assets" / "golden"

load_dotenv(ROOT / ".env")
client = AsyncOpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])


def slug(text):
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_")


async def segment(image_path: Path, concept: str, raw_name: str):
    """Run one concept on one image. Returns (frame_w, frame_h, boxes, masks, raw_text)."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    print(f"\n=== concept={concept!r} image={image_path.name} ({w}x{h}) ===")
    t0 = time.perf_counter()
    events = await client.responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
        ]}],
        stream=True,
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    )
    parsed = parse_responses_stream(events, image_segmentation_format())
    first = None
    async for _ in parsed:
        first = first or time.perf_counter() - t0
    result = await parsed.final_result
    total = time.perf_counter() - t0
    (OUT / raw_name).write_text(result.raw_output)
    print(f"outcome={result.outcome.status} first_snapshot={first or 0:.2f}s total={total:.2f}s "
          f"raw_chars={len(result.raw_output)} diagnostics={len(result.diagnostics)}")
    for d in result.diagnostics:
        print(f"  diagnostic {d.severity} {d.code}: {d.message}")
    boxes = [r for r in result.records if r.kind == "box"]
    masks = {r.object_id: r for r in result.records if r.kind == "mask"}
    return img, w, h, boxes, masks, result.raw_output


def report(img, w, h, boxes, masks, raw, tag):
    print(f"objects found: {len(boxes)}")
    # wire x/y fields of each mask token, e.g. <|mask;x=0;y=0;data=H,W,...|>
    wire_xy = re.findall(r"<\|mask;x=(-?\d+);y=(-?\d+)", raw)
    for i, b in enumerate(boxes):
        m = masks.get(b.object_id)
        bw, bh = b.right - b.left, b.bottom - b.top
        print(f"  [{b.object_id}] box=({b.left},{b.top})-({b.right},{b.bottom}) size={bw}x{bh} conf={b.confidence}")
        if not m:
            print("    no mask record")
            continue
        raster = np.frombuffer(decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
        xy = wire_xy[i] if i < len(wire_xy) else None
        print(f"    mask raster={m.mask.width}x{m.mask.height} (box={bw}x{bh}) wire x,y={xy} "
              f"fg={raster.mean():.1%} conf={m.confidence} bounds==box:"
              f"{(m.bounds.left, m.bounds.top, m.bounds.right, m.bounds.bottom) == (b.left, b.top, b.right, b.bottom)}")
        # paste: stretch the box-local raster to the box and place it at the box's top-left
        canvas = np.zeros((h, w), np.uint8)
        canvas[b.top:b.bottom, b.left:b.right] = cv2.resize(raster * 255, (bw, bh), interpolation=cv2.INTER_NEAREST)
        out = OUT / f"sam_{tag}_{b.object_id}_mask.png"
        cv2.imwrite(str(out), canvas)
        print(f"    saved {out.name}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part", help="noun phrase naming the part in test_01.jpg")
    args = ap.parse_args()

    photo = GOLDEN / "test_01.jpg"
    runs = [(photo, "coin", "sam_coin_raw.txt"),
            (photo, args.part, f"sam_{slug(args.part)}_raw.txt"),
            (photo, "credit card", "sam_credit_card_raw.txt")]
    hand = GOLDEN / "test_02_hand.jpg"
    if hand.exists():
        runs.append((hand, "hand", "sam_hand_raw.txt"))
    else:
        print(f"skipping hand test: {hand} not found")

    for path, concept, raw_name in runs:
        tag = slug(concept) if path == photo else f"{slug(concept)}_{path.stem}"
        report(*await segment(path, concept, raw_name), tag)


if __name__ == "__main__":
    asyncio.run(main())
