"""Phase 1 step 2, spike 1: Muse Spark describes the part and checks photo quality (JSON schema output)."""
import base64
import json
import os
import time

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from spike_common import HAND_PHOTO, OUT, PHOTO, ROOT

load_dotenv(ROOT / ".env")
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

SCHEMA = {
    "type": "object",
    "properties": {
        "part_description": {"type": "string", "description": "2 to 4 words: colour + material + shape. Never a generic noun like 'part' or 'object'."},
        "colour": {"type": "string", "description": "one word, the part's main colour"},
        "reference_object": {"type": "string", "description": "the coin or card used for scale, with your best guess of what it is, or 'none'"},
        "hand_touching_part": {"type": "boolean"},
        "image_quality_issues": {
            "type": "array",
            "items": {"type": "string", "enum": ["camera tilted", "glare", "blurry", "shadow", "low light", "part cut off", "reference cut off", "cluttered background"]},
            "description": "empty list if none",
        },
    },
    "required": ["part_description", "colour", "reference_object", "hand_touching_part", "image_quality_issues"],
    "additionalProperties": False,
}
PROMPT = ("This photo is for an app that measures a small part lying next to a coin or card of known size. "
          "Describe the main part and check the photo quality.")


def jpeg_b64(path, long_side=2048):
    """Resize to 2048 px on the long side, JPEG 0.85, as camera-spec.md says the app will upload."""
    img = cv2.imread(str(path))
    s = long_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def main():
    results = {}
    for path in (PHOTO, HAND_PHOTO):
        t = time.perf_counter()
        r = client.chat.completions.create(
            model="muse-spark-1.3",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(path)}"}},
            ]}],
            response_format={"type": "json_schema", "json_schema": {"name": "part_check", "schema": SCHEMA}},
        )
        raw = r.choices[0].message.content
        print(f"\n=== {path.name} ({time.perf_counter() - t:.2f}s) ===")
        print(raw)
        print(f"usage: {r.usage.prompt_tokens} prompt + {r.usage.completion_tokens} completion = {r.usage.total_tokens}")
        results[path.name] = json.loads(raw)
    (OUT / "spark_describe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
