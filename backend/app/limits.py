"""Limits shared by every worker on one machine for SAM 3.1 calls (used by the Meta client in stage 4):
  - a global concurrency semaphore (SAM_CONCURRENCY slots, file locks under DATA_DIR/locks, released if a process dies),
  - a token bucket for requests per second (in the database),
  - a circuit breaker that pauses calls for BREAKER_OPEN_S after BREAKER_FAILURES consecutive failures and says why,
  - retry with jittered exponential backoff for 429 / 5xx / timeouts."""
import fcntl
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from . import db, ratelimit


class Unavailable(Exception):
    """The breaker is open: callers wait or tell the job why."""

    def __init__(self, reason: str, retry_after_s: float):
        super().__init__(reason)
        self.reason, self.retry_after_s = reason, retry_after_s


def try_slot(lock_dir: Path, n: int) -> int | None:
    """Take one of n machine-wide slots without waiting (fcntl lock; the OS frees a dead process's lock). Returns the fd, or None."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        fd = os.open(lock_dir / f"slot{i}", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
    return None


def release_slot(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


@contextmanager
def slot(lock_dir: Path, n: int, poll_s: float = 0.05, timeout_s: float = 600.0):
    """Hold one of n machine-wide slots, waiting (blocking) for a free one."""
    t0 = time.monotonic()
    while (fd := try_slot(lock_dir, n)) is None:
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError("no free slot")
        time.sleep(poll_s)
    try:
        yield fd
    finally:
        release_slot(fd)


def rate(conn, key: str, rps: float, burst: float, sleep=time.sleep) -> float:
    """Block until the token bucket allows one request; returns the seconds waited."""
    waited = 0.0
    while (w := ratelimit.take(conn, key, burst, rps)) > 0:
        sleep(w)
        waited += w
    return waited


class Breaker:
    """Consecutive-failure circuit breaker kept in the rate_limits table (tokens = failure count, locked_until = open until)."""

    def __init__(self, conn, name: str, failures: int, open_s: float):
        self.conn, self.key, self.failures, self.open_s = conn, f"breaker:{name}", failures, open_s

    def check(self):
        left = ratelimit.locked_for(self.conn, self.key)
        if left > 0:
            raise Unavailable(f"The segmentation service failed several times in a row; calls are paused for {left:.0f} s.", left)

    def record(self, ok: bool):
        with db.tx(self.conn):
            t = db.now()
            row = self.conn.execute("SELECT tokens FROM rate_limits WHERE key = ?", (self.key,)).fetchone()
            n = 0 if ok else (row["tokens"] if row else 0) + 1
            until = t + self.open_s if n >= self.failures else 0
            self.conn.execute("INSERT INTO rate_limits (key, tokens, updated_at, locked_until) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                              "tokens = ?, updated_at = ?, locked_until = ?", (self.key, 0 if until else n, t, until, 0 if until else n, t, until))


def retry(fn, tries=5, base_s=0.5, cap_s=20.0, retry_on=lambda e: True, sleep=time.sleep, rng=random.random):
    """Call fn(); on an exception that retry_on accepts, wait base*2^k with full jitter (capped) and try again, up to `tries` calls."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1 or not retry_on(e):
                raise
            sleep(min(cap_s, base_s * 2 ** k) * rng())
