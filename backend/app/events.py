"""Per-job event log: every state change and progress step is a row (job_id, seq, type, payload JSON, ts), so live progress can be replayed
after a refresh or reconnect (SSE Last-Event-ID or ?after=seq). seq is dense per job and assigned inside the caller's write transaction."""
import json

from . import db

TYPES = ("state_changed", "stage_started", "stage_finished", "progress", "warning", "build_finished", "queue")


def append(conn, job_id: str, type_: str, payload: dict) -> int:
    """Append one event; call inside db.tx(). Returns its seq."""
    assert type_ in TYPES, type_
    seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE job_id = ?", (job_id,)).fetchone()[0]
    conn.execute("INSERT INTO events (job_id, seq, type, payload, ts) VALUES (?, ?, ?, ?, ?)",
                 (job_id, seq, type_, json.dumps(payload, separators=(",", ":")), db.now()))
    return seq


def emit(conn, job_id: str, type_: str, payload: dict) -> int:
    """append() in its own transaction."""
    with db.tx(conn):
        return append(conn, job_id, type_, payload)


def after(conn, job_id: str, seq: int = 0, limit: int = 500) -> list[dict]:
    rows = conn.execute("SELECT seq, type, payload, ts FROM events WHERE job_id = ? AND seq > ? ORDER BY seq LIMIT ?", (job_id, seq, limit))
    return [{"seq": r["seq"], "type": r["type"], "data": json.loads(r["payload"]), "ts": r["ts"]} for r in rows]
