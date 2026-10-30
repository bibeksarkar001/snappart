"""Deleting jobs: DELETE /v1/jobs/{id} and the retention sweep share `purge`. The sweep (run by every worker each sweep_every_s, or
`python -m app.retention` from backend/) deletes every job not changed for retention_days, optionally the frames of finished parts, and
stale rate-limit and budget rows. Voice audio is never stored (app/voice.py), so there is nothing to sweep for it."""
import hashlib

from . import db

FINISHED = ("done", "failed", "cancelled", "expired")


def purge(c, st, job_id: str):
    """Frames, outputs, answers, corrections, events and the job row; keeps only an anonymous tombstone (hash of the id, time)."""
    st.delete_job(job_id)
    with db.tx(c):
        c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))           # uploads, events, answers, corrections cascade
        c.execute("INSERT OR REPLACE INTO tombstones (id_hash, deleted_at) VALUES (?, ?)", (hashlib.sha256(job_id.encode()).hexdigest(), db.now()))
    st.delete_job(job_id)                                               # again: a build may have written meanwhile


def sweep(c, st, s) -> dict:
    t = db.now()
    old = [r[0] for r in c.execute("SELECT id FROM jobs WHERE updated_at < ?", (t - s.retention_days * 86400,))]
    for jid in old:
        purge(c, st, jid)
    frames = 0
    if s.delete_frames_when_done:
        for (jid,) in c.execute(f"SELECT DISTINCT j.id FROM jobs j JOIN uploads u ON u.job_id = j.id WHERE j.state IN {FINISHED}").fetchall():
            for key in st.list(jid, "videos/"):
                st.delete(jid, key)
            with db.tx(c):
                frames += c.execute("DELETE FROM uploads WHERE job_id = ?", (jid,)).rowcount
    with db.tx(c):
        limits = c.execute("DELETE FROM rate_limits WHERE updated_at < ? AND COALESCE(locked_until, 0) < ?", (t - 2 * 86400, t)).rowcount
        c.execute("DELETE FROM budget_ledger WHERE day < date(?, 'unixepoch', '-31 days')", (t,))
    return {"jobs_deleted": len(old), "frames_deleted": frames, "rate_limit_rows_deleted": limits}


if __name__ == "__main__":
    from .settings import get_settings
    from .storage import LocalStorage
    s = get_settings()
    with db.session(s.db_path) as c:
        db.migrate(c)
        print(sweep(c, LocalStorage(s.data_dir), s))
