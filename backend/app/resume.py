"""Resume codes: three words from the fixed 2048-word BIP-39 English list (wordlist.txt, MIT via @scure/bip39) plus two digits, e.g.
"river-taxi-orbit-42" (about 39.6 bits). Stored only as an scrypt hash keyed by a server pepper, so a leaked database does not reveal codes
and lookups need no plaintext. Guessing is stopped by per-IP and per-code lockouts in the API."""
import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

WORDS = Path(__file__).with_name("wordlist.txt").read_text().split()
assert len(WORDS) == 2048 and len(set(WORDS)) == 2048
_WORDSET = set(WORDS)


def new_code() -> str:
    return "-".join(secrets.choice(WORDS) for _ in range(3)) + f"-{secrets.randbelow(100):02d}"


def normalise(code: str) -> str | None:
    """Canonical form "w1-w2-w3-NN" of what a user typed ("River Taxi Orbit 42", "river-taxi-orbit-42"), or None if it cannot be a code."""
    parts = re.findall(r"[a-z]+|\d+", (code or "").lower()[:80])
    if len(parts) != 4 or not all(p in _WORDSET for p in parts[:3]) or not re.fullmatch(r"\d{2}", parts[3]):
        return None
    return "-".join(parts)


def hash_code(code: str, pepper: bytes) -> str:
    return hashlib.scrypt(code.encode(), salt=pepper, n=2 ** 14, r=8, p=1, dklen=32).hex()


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def pepper(settings) -> bytes:
    """The server pepper: from settings, else generated once into DATA_DIR/.resume_pepper (mode 0600)."""
    if settings.resume_pepper is not None:
        return settings.resume_pepper.get_secret_value().encode()
    p = settings.data_dir / ".resume_pepper"
    if not p.exists():
        tmp = p.with_name(f".resume_pepper.{secrets.token_hex(6)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
        try:
            os.link(tmp, p)                            # atomic and exclusive: a concurrent reader never sees an empty file
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    return p.read_text().strip().encode()
