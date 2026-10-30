"""Token buckets and lockouts in the `rate_limits` table, so every API replica and worker on one database shares them. Client identity is a
salted hash of the IP (X-Forwarded-For trusted only from configured proxies); raw IPs are never stored."""
import hashlib
import math

from . import db
from .errors import ApiError


def take(conn, key: str, capacity: float, refill_per_s: float, cost: float = 1.0) -> float:
    """Spend `cost` tokens from bucket `key` (capacity, refill rate). Returns 0 if allowed, else the seconds until enough tokens exist."""
    with db.tx(conn):
        t = db.now()
        row = conn.execute("SELECT tokens, updated_at FROM rate_limits WHERE key = ?", (key,)).fetchone()
        tokens = capacity if row is None else min(capacity, row["tokens"] + (t - row["updated_at"]) * refill_per_s)
        if tokens >= cost:
            tokens -= cost
            wait = 0.0
        else:
            wait = (cost - tokens) / refill_per_s if refill_per_s > 0 else math.inf
        conn.execute("INSERT INTO rate_limits (key, tokens, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET tokens = ?, updated_at = ?",
                     (key, tokens, t, tokens, t))
    return wait


def limit(conn, key: str, per_hour: float, burst: float | None = None, what="requests"):
    """Raise 429 (with Retry-After) when the hourly bucket for key is empty."""
    wait = take(conn, key, burst or per_hour, per_hour / 3600.0)
    if wait > 0:
        s = max(1, math.ceil(wait))
        raise ApiError(429, "rate_limited", f"Too many {what} from your connection. Try again in {s} s.", retry_after_s=s)


def locked_for(conn, key: str) -> float:
    row = conn.execute("SELECT locked_until FROM rate_limits WHERE key = ?", (key,)).fetchone()
    return max(0.0, (row["locked_until"] or 0) - db.now()) if row else 0.0


def fail(conn, key: str, max_fails: int, base_s: float, max_s: float) -> float:
    """Count one failure on key; from the max_fails-th failure on, lock it for base_s doubling per further failure (capped at max_s).
    Returns the lock length (0 if not locked)."""
    with db.tx(conn):
        row = conn.execute("SELECT tokens FROM rate_limits WHERE key = ?", (key,)).fetchone()
        n = (row["tokens"] if row else 0) + 1
        lock = min(max_s, base_s * 2 ** (n - max_fails)) if n >= max_fails else 0.0
        t = db.now()
        conn.execute("INSERT INTO rate_limits (key, tokens, updated_at, locked_until) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                     "tokens = ?, updated_at = ?, locked_until = ?", (key, n, t, t + lock, n, t, t + lock))
    return lock


def client_ip(request, trusted_proxies) -> str:
    peer = request.client.host if request.client else "unknown"
    if peer in trusted_proxies:
        fwd = request.headers.get("x-forwarded-for", "")
        hops = [h.strip() for h in fwd.split(",") if h.strip()]
        for h in reversed(hops):                       # the right-most address not added by one of our proxies
            if h not in trusted_proxies:
                return h
    return peer


def ip_hash(ip: str, salt: bytes) -> str:
    return hashlib.sha256(salt + ip.encode()).hexdigest()[:24]
