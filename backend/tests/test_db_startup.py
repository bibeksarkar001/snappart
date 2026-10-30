"""Regression (stage 7 load test): several API processes starting at once on a fresh database. Switching to WAL could fail with
"database is locked" without waiting, which killed one uvicorn process at startup."""
import multiprocessing as mp
from pathlib import Path

from app import db


def _open(p):
    with db.session(Path(p)) as c:
        db.migrate(c)
        return c.execute("PRAGMA journal_mode").fetchone()[0]


def test_parallel_first_start(tmp_path):
    for k in range(5):
        p = tmp_path / f"{k}.db"
        with mp.get_context("spawn").Pool(6) as pool:
            assert pool.map(_open, [str(p)] * 6) == ["wal"] * 6
