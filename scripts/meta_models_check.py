"""Which Meta models this API key can use: GET {base}/models, printing model ids only (never the key).

Usage (repo root): python scripts/meta_models_check.py   (the key is read from MODEL_API_KEY, loaded from .env by python-dotenv)"""
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0")


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        print("MODEL_API_KEY is not set")
        return 2
    r = httpx.get(os.environ.get("SNAPPART_META_BASE_URL", "https://api.meta.ai/v1") + "/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=20)
    print(f"GET /models -> HTTP {r.status_code}")
    if r.status_code != 200:
        return 1
    body = r.json()
    rows = body.get("data", body.get("models", [])) if isinstance(body, dict) else body
    ids = sorted(m.get("id") if isinstance(m, dict) else str(m) for m in rows)
    print("model ids:", ", ".join(ids))
    for m in NEEDED:
        print(f"  {m}: {'available' if m in ids else 'NOT AVAILABLE to this key'}")
    return 0 if all(m in ids for m in NEEDED) else 1


if __name__ == "__main__":
    sys.exit(main())
