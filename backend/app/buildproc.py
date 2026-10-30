"""One build, run by the worker in an isolated subprocess (rlimits + wall-clock watchdog are applied by the worker).

`python -m app.buildproc <job_id>` (from backend/) runs the build step for a job the worker has claimed and writes its outcome to
work/result.json: {"state", "reason", "error": [code, message] | null}. The worker, which holds the lease, applies the outcome. The build
emits its own progress events and keeps stage checkpoints under work/, so a requeued build resumes instead of restarting."""
import json
import sys

from . import db, mock
from .settings import get_settings
from .storage import LocalStorage


class BuildFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def real_build(conn, storage, settings, job_id):
    """Real pipeline run from the uploaded frames. Not wired yet: it needs the Meta client for SAM 3.1 (backend stage 4)."""
    raise BuildFailure("real_build_not_available", "Building from your frames is not switched on yet on this server.")


def run(job_id: str) -> dict:
    s = get_settings()
    st = LocalStorage(s.data_dir)
    with db.session(s.db_path) as conn:
        try:
            state, reason = (mock.run_build if s.mock else real_build)(conn, st, s, job_id)
            out = {"state": state, "reason": reason, "error": None}
        except BuildFailure as e:
            out = {"state": "failed", "reason": e.message, "error": [e.code, e.message]}
    st.put(job_id, "work/result.json", json.dumps(out).encode())
    return out


if __name__ == "__main__":
    run(sys.argv[1])
