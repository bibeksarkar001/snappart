"""Spike 4: Muse Spark photo check with JSON-schema structured output (Chat Completions).

Docs: image input = image_url object with a data URL; structured output = response_format json_schema.
"""
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

SCHEMA = {
    "type": "object",
    "properties": {
        "coin_visible": {"type": "boolean"},
        "card_visible": {"type": "boolean"},
        "part_visible": {"type": "boolean"},
        "part_name": {"type": "string"},
        "reference_guess": {"type": "string", "description": "what the reference object is, e.g. '1 euro coin', 'bank card', 'none'"},
        "camera_tilted": {"type": "boolean"},
        "glare": {"type": "boolean"},
    },
    "required": ["coin_visible", "card_visible", "part_visible", "part_name", "reference_guess", "camera_tilted", "glare"],
    "additionalProperties": False,
}

PROMPT = ("Check this photo for a measuring app. A flat part should lie next to a coin or bank card "
          "of known size, photographed from directly above. Report what you see.")


def main():
    b64 = base64.b64encode((ROOT / "assets" / "golden" / "test_01.jpg").read_bytes()).decode()
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "photo_check", "schema": SCHEMA}},
    )
    print(f"time: {time.perf_counter() - t0:.2f}s")
    print(json.dumps(json.loads(r.choices[0].message.content), indent=2))
    print("usage:", r.usage.prompt_tokens, "prompt +", r.usage.completion_tokens, "completion =", r.usage.total_tokens)


if __name__ == "__main__":
    main()
