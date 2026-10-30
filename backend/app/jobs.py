"""Job records and the state machine. Every transition is checked against TRANSITIONS inside one write transaction and recorded as a
state_changed event with its timestamp; an illegal transition raises ApiError 409 and changes nothing."""
import json
import secrets

from . import db, events
from .errors import ApiError, not_found

STATES = ("created", "uploading", "checking", "ready_to_build", "queued", "building", "needs_input", "reviewing", "done",
          "failed", "cancelled", "expired")
_ANY_LIVE = {"cancelled", "expired", "failed"}
TRANSITIONS = {
    "created": {"uploading"} | {"cancelled", "expired"},
    "uploading": {"checking"} | _ANY_LIVE,
    "checking": {"ready_to_build", "uploading"} | _ANY_LIVE,           # back to uploading: the check asks for more footage
    "ready_to_build": {"queued", "uploading", "checking"} | _ANY_LIVE,
    "queued": {"building", "ready_to_build"} | _ANY_LIVE,               # back to ready_to_build: shed from the queue
    "building": {"needs_input", "reviewing", "queued"} | _ANY_LIVE,     # back to queued: lease lost, resumes from its checkpoint
    "needs_input": {"reviewing", "queued", "uploading"} | _ANY_LIVE,    # uploading: more footage requested
    "reviewing": {"done", "needs_input", "queued", "uploading"} | _ANY_LIVE,          # uploading: the user films more of an area
    "done": {"reviewing", "expired"},
    "failed": {"expired"},
    "cancelled": {"expired"},
    "expired": set(),
}
assert set(TRANSITIONS) == set(STATES) and all(t <= set(STATES) for t in TRANSITIONS.values())


def new_id() -> str:
    return secrets.token_urlsafe(24)                                    # 192 bits


def create(conn, mode: str, ip_hash: str | None = None) -> str:
    jid, t = new_id(), db.now()
    with db.tx(conn):
        conn.execute("INSERT INTO jobs (id, state, mode, created_at, updated_at, state_changed_at, ip_hash) VALUES (?, 'created', ?, ?, ?, ?, ?)",
                     (jid, mode, t, t, t, ip_hash))
        events.append(conn, jid, "state_changed", {"from": None, "to": "created", "reason": "job created"})
    return jid


def get(conn, job_id: str) -> dict:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise not_found()
    return dict(row)


def transition(conn, job_id: str, to: str, reason: str = "", error: tuple[str, str] | None = None, expect: str | None = None) -> dict:
    """Move a job to `to` if TRANSITIONS allows it from its current state (and, with expect, only from that state). Returns the new row."""
    with db.tx(conn):
        row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise not_found()
        cur = row["state"]
        if (expect is not None and cur != expect) or to not in TRANSITIONS[cur]:
            raise ApiError(409, "illegal_transition", f"This part is {cur.replace('_', ' ')}; it cannot go to {to.replace('_', ' ')} now.")
        t = db.now()
        conn.execute("UPDATE jobs SET state = ?, updated_at = ?, state_changed_at = ?, error_code = ?, error_message = ? WHERE id = ?",
                     (to, t, t, error[0] if error else None, error[1] if error else None, job_id))
        events.append(conn, job_id, "state_changed", {"from": cur, "to": to, "reason": reason, **({"error": error[0]} if error else {})})
    return get(conn, job_id)


def add_warning(conn, job_id: str, text: str):
    with db.tx(conn):
        w = json.loads(get(conn, job_id)["warnings"])
        conn.execute("UPDATE jobs SET warnings = ?, updated_at = ? WHERE id = ?", (json.dumps(w + [text]), db.now(), job_id))
        events.append(conn, job_id, "warning", {"message": text})


def public(row: dict, outputs: list[str] | None = None) -> dict:
    """What a client may see about a job (no hashes, no lease internals)."""
    return {"job_id": row["id"], "state": row["state"], "mode": row["mode"], "created_at": row["created_at"], "updated_at": row["updated_at"],
            "state_changed_at": row["state_changed_at"], "warnings": json.loads(row["warnings"]),
            "error": {"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None,
            "queue_position": None, "eta_s": None, "outputs": outputs or []}


def model_status(conn, models, mode: str, window_s: int, job_id: str | None = None) -> list[dict]:
    """Truthful model state from recorded calls only: "running" needs a successful recorded call (in this job, or globally within the
    window). In mock mode no model is running, whatever the table says."""
    out, since = [], db.now() - window_s
    for m in models:
        q = "SELECT SUM(ok), SUM(1 - ok), MAX(CASE WHEN ok THEN ts END) FROM model_calls WHERE model = ?"
        args = [m]
        if job_id:
            q += " AND job_id = ?"
            args.append(job_id)
        ok, bad, last_ok = conn.execute(q, args).fetchone()
        running = mode == "real" and last_ok is not None and (job_id is not None or last_ok >= since)
        reason = ("mock mode: no Meta model is called" if mode == "mock" else
                  "a call succeeded" + (" in this job" if job_id else " recently") if running else "no successful call recorded")
        out.append({"id": m, "running": running, "reason": reason, "calls_ok": ok or 0, "calls_failed": bad or 0, "last_ok_at": last_ok})
    return out
