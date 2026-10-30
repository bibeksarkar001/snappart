"""Job file storage. A small interface (put / get / exists / list / size / delete_job / path) with a local-disk implementation under
DATA_DIR/jobs/{id}/. Keys are server-generated, relative, and validated, so no user-controlled path ever reaches the filesystem. An
S3-compatible store can replace LocalStorage by implementing the same methods (`path` is only used by the local pipeline runner)."""
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

JOB_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
KEY = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,99}(/[A-Za-z0-9_-][A-Za-z0-9._-]{0,99}){0,5}$")


class StorageKeyError(ValueError):
    pass


def check(job_id: str, key: str | None = None):
    if not JOB_ID.match(job_id or ""):
        raise StorageKeyError("bad job id")
    if key is not None and (not KEY.match(key) or ".." in key.split("/")):
        raise StorageKeyError("bad storage key")


class Storage(Protocol):
    def put(self, job_id: str, key: str, data: bytes) -> None: ...
    def get(self, job_id: str, key: str) -> bytes: ...
    def exists(self, job_id: str, key: str) -> bool: ...
    def list(self, job_id: str, prefix: str = "") -> list[str]: ...
    def size(self, job_id: str) -> int: ...
    def delete(self, job_id: str, key: str) -> None: ...
    def delete_job(self, job_id: str) -> None: ...
    def path(self, job_id: str, key: str = "") -> Path: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str, key: str = "") -> Path:
        check(job_id, key or None)
        return self.root / job_id / key if key else self.root / job_id

    def put(self, job_id: str, key: str, data: bytes) -> None:
        p = self.path(job_id, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)                       # atomic: readers never see a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, job_id: str, key: str) -> bytes:
        return self.path(job_id, key).read_bytes()

    def exists(self, job_id: str, key: str) -> bool:
        return self.path(job_id, key).is_file()

    def list(self, job_id: str, prefix: str = "") -> list[str]:
        base = self.path(job_id)
        if not base.is_dir():
            return []
        out = [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and not p.name.startswith(".tmp-")]
        return sorted(k for k in out if k.startswith(prefix))

    def size(self, job_id: str) -> int:
        base = self.path(job_id)
        return sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) if base.is_dir() else 0

    def delete(self, job_id: str, key: str) -> None:
        self.path(job_id, key).unlink(missing_ok=True)

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self.path(job_id), ignore_errors=True)
