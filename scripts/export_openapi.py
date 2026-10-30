"""Write the API's machine-readable contract to docs/openapi.json: `python scripts/export_openapi.py` from the repo root."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("SNAPPART_DATA_DIR", tempfile.mkdtemp(prefix="snappart-openapi-"))   # no real data dir needed

from app.main import create_app  # noqa: E402

spec = create_app().openapi()
(ROOT / "docs" / "openapi.json").write_text(json.dumps(spec, indent=1) + "\n")
print(f"docs/openapi.json: {len(spec['paths'])} paths")
