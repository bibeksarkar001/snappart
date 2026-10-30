"""SQLite in WAL mode, one short-lived connection per request or worker step, schema migrations recorded in a version table.

Writes that read-then-write run inside `tx()` (BEGIN IMMEDIATE), so concurrent API replicas and workers on one machine serialise safely.
Beyond one machine, swap this module for Postgres (see docs/ops.md); the SQL is plain."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def now() -> float:
    """Wall clock for every stored timestamp; tests replace it with a fake clock."""
    return time.time()


MIGRATIONS = [
    (1, """
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        mode TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        state_changed_at REAL NOT NULL,
        ip_hash TEXT,
        resume_hash TEXT,
        error_code TEXT,
        error_message TEXT,
        warnings TEXT NOT NULL DEFAULT '[]',
        enqueued_at REAL,
        lease_owner TEXT,
        lease_until REAL,
        attempts INTEGER NOT NULL DEFAULT 0,
        checkpoint TEXT
    );
    CREATE INDEX jobs_state ON jobs(state, enqueued_at);
    CREATE TABLE uploads (
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        video INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        size INTEGER NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (job_id, video, frame_index)
    );
    CREATE TABLE events (
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        ts REAL NOT NULL,
        PRIMARY KEY (job_id, seq)
    );
    CREATE TABLE answers (
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        question_id TEXT NOT NULL,
        value TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (job_id, question_id)
    );
    CREATE TABLE corrections (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        body TEXT NOT NULL,
        reverted INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE TABLE model_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
        model TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        ok INTEGER NOT NULL,
        latency_ms REAL,
        status_code INTEGER,
        tokens INTEGER,
        cost REAL,
        ts REAL NOT NULL
    );
    CREATE INDEX model_calls_model ON model_calls(model, ts);
    CREATE TABLE rate_limits (
        key TEXT PRIMARY KEY,
        tokens REAL NOT NULL,
        updated_at REAL NOT NULL,
        locked_until REAL
    );
    CREATE TABLE budget_ledger (
        day TEXT NOT NULL,
        model TEXT NOT NULL,
        calls INTEGER NOT NULL DEFAULT 0,
        cost REAL NOT NULL DEFAULT 0,
        PRIMARY KEY (day, model)
    );
    """),
    (2, """
    ALTER TABLE jobs ADD COLUMN part_name TEXT;
    ALTER TABLE jobs ADD COLUMN build_anyway INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE jobs ADD COLUMN build_started_at REAL;
    ALTER TABLE jobs ADD COLUMN build_finished_at REAL;
    CREATE INDEX jobs_resume ON jobs(resume_hash);
    CREATE TABLE tombstones (
        id_hash TEXT PRIMARY KEY,
        deleted_at REAL NOT NULL
    )
    """),
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def session(path: Path):
    """A connection that is closed on exit (sqlite3's own context manager only commits)."""
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def tx(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply missing migrations in order; returns the schema version. Safe to call on every start (idempotent)."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    with tx(conn):
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        have = row[0] or 0
        for version, sql in MIGRATIONS:
            if version > have:
                for stmt in filter(str.strip, sql.split(";")):
                    conn.execute(stmt)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                have = version
    return have
