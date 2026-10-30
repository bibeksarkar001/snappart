"""Database-backed build queue with leases. At-least-once: a worker claims a queued job (oldest first, at most one active build per client,
at most MAX_CONCURRENT_BUILDS across all workers), renews its lease with a heartbeat, and a job whose lease runs out goes back to the
queue and resumes from its last on-disk checkpoint. Everything here is one short write transaction, so API replicas and workers can share
one database."""
import json

from . import db, events
from .errors import ApiError


def queued_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'queued'").fetchone()[0]


def enqueue(conn, job_id: str, settings, ip_hash: str | None, warning: str | None = None) -> dict:
    """ready_to_build -> queued, with admission control: bounded queue (503 + Retry-After + how many are ahead) and a per-client cap."""
    with db.tx(conn):
        row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ApiError(404, "not_found", "No such job.")
        if row["state"] != "ready_to_build":
            raise ApiError(409, "illegal_transition", f"This part is {row['state'].replace('_', ' ')}; it can be started only after the capture check.")
        ahead = queued_count(conn)
        if ahead >= settings.max_queue_len:
            wait = max(30, int(eta_per_slot(conn, settings) or 60))
            raise ApiError(503, "queue_full", f"We are busy: {ahead} builds are waiting. Please try again in about {wait} s; your upload is kept.",
                           retry_after_s=wait)
        if ip_hash and conn.execute("SELECT COUNT(*) FROM jobs WHERE ip_hash = ? AND state IN ('queued', 'building')",
                                    (ip_hash,)).fetchone()[0] >= settings.max_queued_per_ip:
            raise ApiError(429, "too_many_builds", "You already have builds waiting. Start this one when one of them is finished.",
                           retry_after_s=60)
        t = db.now()
        conn.execute("UPDATE jobs SET state = 'queued', enqueued_at = ?, updated_at = ?, state_changed_at = ?, attempts = 0 WHERE id = ?",
                     (t, t, t, job_id))
        if warning:
            w = json.loads(conn.execute("SELECT warnings FROM jobs WHERE id = ?", (job_id,)).fetchone()[0])
            conn.execute("UPDATE jobs SET warnings = ? WHERE id = ?", (json.dumps(w + [warning]), job_id))
            events.append(conn, job_id, "warning", {"message": warning})
        events.append(conn, job_id, "state_changed", {"from": "ready_to_build", "to": "queued", "reason": "build requested"})
        events.append(conn, job_id, "queue", {"position": ahead + 1})
    return position(conn, job_id, settings)


def avg_build_s(conn, n=20) -> float | None:
    """Rolling average of the last n finished builds' durations (measured, not assumed)."""
    rows = conn.execute("SELECT build_finished_at - build_started_at FROM jobs WHERE build_finished_at IS NOT NULL AND build_started_at IS NOT NULL "
                        "ORDER BY build_finished_at DESC LIMIT ?", (n,)).fetchall()
    return sum(r[0] for r in rows) / len(rows) if rows else None


def eta_per_slot(conn, settings) -> float | None:
    a = avg_build_s(conn)
    return None if a is None else a / max(1, settings.max_concurrent_builds)


def position(conn, job_id: str, settings) -> dict:
    """{queue_position (1 = next), eta_s (rolling average build time x builds ahead per build slot; None until a build has finished)}."""
    row = conn.execute("SELECT state, enqueued_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["state"] != "queued":
        return {"queue_position": None, "eta_s": None}
    pos = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'queued' AND enqueued_at <= ?", (row["enqueued_at"],)).fetchone()[0]
    a = avg_build_s(conn)
    running = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'building'").fetchone()[0]
    slots = max(1, settings.max_concurrent_builds)
    eta = None if a is None else a * ((pos - 1 + running) // slots + 1)
    return {"queue_position": pos, "eta_s": None if eta is None else round(eta, 1)}


def claim(conn, owner: str, settings) -> str | None:
    """Take the oldest queued job whose client has no active build, if a build slot is free. Sets state building and the lease."""
    with db.tx(conn):
        if conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'building'").fetchone()[0] >= settings.max_concurrent_builds:
            return None
        row = conn.execute(
            "SELECT id FROM jobs WHERE state = 'queued' AND (ip_hash IS NULL OR ip_hash NOT IN "
            "(SELECT ip_hash FROM jobs WHERE state = 'building' AND ip_hash IS NOT NULL)) ORDER BY enqueued_at LIMIT 1").fetchone()
        if row is None:
            return None
        t = db.now()
        conn.execute("UPDATE jobs SET state = 'building', lease_owner = ?, lease_until = ?, attempts = attempts + 1, updated_at = ?, "
                     "state_changed_at = ?, build_started_at = COALESCE(build_started_at, ?) WHERE id = ?",
                     (owner, t + settings.lease_s, t, t, t, row["id"]))
        events.append(conn, row["id"], "state_changed", {"from": "queued", "to": "building", "reason": "a worker started the build"})
        return row["id"]


def heartbeat(conn, job_id: str, owner: str, settings) -> bool:
    """Renew the lease; False if the job is gone, cancelled or now owned by someone else (the worker must stop)."""
    with db.tx(conn):
        cur = conn.execute("UPDATE jobs SET lease_until = ? WHERE id = ? AND state = 'building' AND lease_owner = ?",
                           (db.now() + settings.lease_s, job_id, owner))
        return cur.rowcount == 1


def reap(conn, settings) -> list[str]:
    """Builds whose lease ran out (their worker died): back to queued (resume from checkpoint), or failed after max_attempts."""
    out = []
    with db.tx(conn):
        t = db.now()
        for r in conn.execute("SELECT id, attempts FROM jobs WHERE state = 'building' AND lease_until < ?", (t,)).fetchall():
            if r["attempts"] >= settings.max_attempts:
                conn.execute("UPDATE jobs SET state = 'failed', error_code = 'build_interrupted', error_message = ?, lease_owner = NULL, "
                             "updated_at = ?, state_changed_at = ? WHERE id = ?",
                             ("The build stopped several times. Please try again later.", t, t, r["id"]))
                events.append(conn, r["id"], "state_changed", {"from": "building", "to": "failed", "reason": "worker lost too often", "error": "build_interrupted"})
            else:
                conn.execute("UPDATE jobs SET state = 'queued', lease_owner = NULL, updated_at = ?, state_changed_at = ? WHERE id = ?", (t, t, r["id"]))
                events.append(conn, r["id"], "state_changed", {"from": "building", "to": "queued", "reason": "worker stopped; resuming from the last checkpoint"})
            out.append(r["id"])
    return out


def finish(conn, job_id: str, owner: str, to: str, reason: str, error: tuple[str, str] | None = None) -> bool:
    """building -> to, only while this worker still holds the lease; records the build's end time (for the ETA)."""
    with db.tx(conn):
        t = db.now()
        cur = conn.execute("UPDATE jobs SET state = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?, state_changed_at = ?, "
                           "build_finished_at = CASE WHEN ? IN ('needs_input', 'reviewing') THEN ? ELSE build_finished_at END, "
                           "error_code = ?, error_message = ? WHERE id = ? AND state = 'building' AND lease_owner = ?",
                           (to, t, t, to, t, error[0] if error else None, error[1] if error else None, job_id, owner))
        if cur.rowcount != 1:
            return False
        events.append(conn, job_id, "state_changed", {"from": "building", "to": to, "reason": reason, **({"error": error[0]} if error else {})})
    return True
