"""SnapPart backend app: job API, storage, queue and workers around the capture pipeline in backend/capture/.

The pipeline is imported as the top-level package `capture` (as the tests do), so the backend directory is put on sys.path here; this lets
both `python -m backend.app.<module>` from the repo root and `import app` from backend/ work."""
import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
