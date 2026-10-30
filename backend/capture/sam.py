"""SAM 3.1 for the capture pipeline, without any network code: the Responses API request for one concept on one crop, turning the parsed
records (box + box-local mask raster per object, with confidence) into crop-space masks, and an adapter that gives
pipeline.masks_from_crops its `sam(jobs)` callable from an async `segment(jpeg, concept)` function (app/meta_client.py provides it).
The Lab-colour pick / fusion / edge refinement stays in masks.py and runs on these masks."""
import asyncio
import base64

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster


def request(jpeg: bytes, concept: str) -> dict:
    """Responses API body: one short noun phrase and one image (docs: one concept per request), compact masks with confidence."""
    return {"model": "sam-3.1",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}]}],
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}


def masks(result, shape) -> list[tuple[float, np.ndarray]]:
    """[(confidence, 0/255 mask of the image's shape)] from a parsed image-segmentation result: each object's box-local raster pasted into
    its box (stretched to the box, as the spikes measured)."""
    h, w = shape[:2]
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    out = []
    for r in result.records:
        if r.kind != "mask" or r.object_id not in boxes:
            continue
        b = boxes[r.object_id]
        x0, y0, x1, y1 = max(b.left, 0), max(b.top, 0), min(b.right, w), min(b.bottom, h)
        if x1 <= x0 or y1 <= y0:
            continue
        raster = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        full = cv2.resize(raster * 255, (b.right - b.left, b.bottom - b.top), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((h, w), np.uint8)
        canvas[y0:y1, x0:x1] = full[y0 - b.top:y1 - b.top, x0 - b.left:x1 - b.left]
        conf = r.confidence if r.confidence is not None else (b.confidence if b.confidence is not None else 0.0)
        out.append((float(conf), canvas))
    return out


def adapter(segment, jpeg_quality: int = 92, log=lambda m: None):
    """sam(jobs) for pipeline.masks_from_crops: jobs [(key, crop BGR, concept, box)] -> {key: [(conf, mask)] or None}. All jobs of a call run
    concurrently (segment() applies the shared concurrency, rate and retry limits); a job that fails or finds nothing maps to None, so the
    pipeline tries its fallback concept for it."""
    def sam(jobs):
        async def one(key, crop, concept):
            ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            try:
                res = await segment(enc.tobytes(), concept)
            except Exception as e:                                    # noqa: BLE001  (reported, then treated as a miss)
                log(f"SAM failed for {key}: {getattr(e, 'message', type(e).__name__)}")
                return key, None
            m = masks(res, crop.shape)
            return key, m or None

        async def run():
            return dict(await asyncio.gather(*[one(k, crop, concept) for k, crop, concept, _ in jobs]))
        return asyncio.run(run())
    return sam
