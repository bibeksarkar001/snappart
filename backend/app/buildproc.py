"""One build, run by the worker in an isolated subprocess (rlimits + wall-clock watchdog are applied by the worker).

`python -m app.buildproc <job_id>` (from backend/) runs the build step for a job the worker has claimed and writes its outcome to
work/result.json: {"state", "reason", "error": [code, message] | null}. The worker, which holds the lease, applies the outcome. The build
emits its own progress events and keeps stage checkpoints under work/, so a requeued build resumes instead of restarting."""
import json
import sys

from . import db, mock
from .settings import get_settings
from .storage import LocalStorage


from .errors import BuildFailure  # noqa: F401  (one class: this module also runs as __main__)


def real_build(conn, storage, settings, job_id):
    """Real pipeline run from the uploaded frames (app/realbuild.py)."""
    from . import realbuild
    return realbuild.run(conn, storage, settings, job_id)


def run(job_id: str) -> dict:
    s = get_settings()
    st = LocalStorage(s.data_dir)
    with db.session(s.db_path) as conn:
        try:
            if s.mock:
                state, reason = mock.run_build(conn, st, s, job_id)
            elif s.replay:
                from . import replay
                state, reason = replay.run_build(conn, st, s, job_id)
            else:
                state, reason = real_build(conn, st, s, job_id)
            out = {"state": state, "reason": reason, "error": None}
        except BuildFailure as e:
            out = {"state": "failed", "reason": e.message, "error": [e.code, e.message]}
    st.put(job_id, "work/result.json", json.dumps(out).encode())
    return out


if __name__ == "__main__":
    run(sys.argv[1])
