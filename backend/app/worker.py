"""Build worker: `python -m backend.app.worker` (repo root) or `python -m app.worker` (backend/). Run as many as the machine allows; they
coordinate through the database (leases) and never share memory with the API.

Loop: reap lost leases -> claim the next fair job -> run app.buildproc for it in a subprocess with memory / CPU rlimits and a wall-clock
watchdog -> heartbeat the lease every HEARTBEAT_S (stop the build if the job was deleted or taken over) -> apply the outcome. A crashed
build subprocess is requeued (resuming from its checkpoints) until MAX_ATTEMPTS, then failed with a plain reason."""
import json
import os
import resource
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import db, jobqueue, retention
from .settings import Settings, get_settings
from .storage import LocalStorage

BACKEND = Path(__file__).resolve().parents[1]
FRIENDLY = {"build_timeout": "This build took too long and was stopped. Please try again, or film a little less footage.",
            "build_crashed": "The build stopped unexpectedly. Please try again."}


def settings_env(s: Settings) -> dict:
    """Environment for the build subprocess: the parent's, plus this worker's settings (so tests and CLI overrides reach it)."""
    env = dict(os.environ)
    env.update({"SNAPPART_DATA_DIR": str(s.data_dir), "SNAPPART_MOCK": "1" if s.mock else "0", "SNAPPART_MOCK_BUILD_S": str(s.mock_build_s),
                "SNAPPART_ENVIRONMENT": s.environment, "SNAPPART_ALLOW_MOCK_IN_PRODUCTION": "1" if s.allow_mock_in_production else "0"})
    return env


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        os.setsid()
        for res, val in ((getattr(resource, "RLIMIT_AS", None), mem_mb * 1024 * 1024), (resource.RLIMIT_CPU, cpu_s)):
            if res is None:
                continue
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):                       # macOS does not enforce RLIMIT_AS; the watchdog still applies
                pass
    return apply


class Worker:
    def __init__(self, settings: Settings | None = None, name: str | None = None):
        self.s = settings or get_settings()
        self.owner = name or f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"
        self.storage = LocalStorage(self.s.data_dir)
        self.stopping = False
        self.child: subprocess.Popen | None = None

    def step(self) -> str | None:
        """Reap, claim, run one build. Returns the job id it ran, or None if there was nothing to do."""
        with db.session(self.s.db_path) as c:
            db.migrate(c)
            jobqueue.reap(c, self.s)
            jid = None if self.stopping else jobqueue.claim(c, self.owner, self.s)
        if jid:
            self.run_job(jid)
        return jid

    def run_job(self, jid: str):
        work = self.storage.path(jid, "work")
        work.mkdir(parents=True, exist_ok=True)
        (work / "result.json").unlink(missing_ok=True)
        cpu = max(60, int(self.s.build_timeout_s * 2 * max(1, os.cpu_count() or 1)))
        with open(work / "build.log", "ab") as logf:                    # a file, not a pipe: a chatty build can never block on it
            self.child = subprocess.Popen([sys.executable, "-m", "app.buildproc", jid], cwd=BACKEND, env=settings_env(self.s),
                                          preexec_fn=_limits(self.s.build_mem_mb, cpu), stdout=logf, stderr=logf)
        t0, outcome = time.monotonic(), None
        while True:
            try:
                rc = self.child.wait(timeout=self.s.heartbeat_s)
                break
            except subprocess.TimeoutExpired:
                pass
            with db.session(self.s.db_path) as c:
                alive = jobqueue.heartbeat(c, jid, self.owner, self.s)
            if not alive:                                           # deleted, cancelled, or another worker took it over
                self._kill()
                return
            if self.stopping:
                self._kill()
                outcome = ("queued", "worker shutting down; the build resumes on another worker", None)
                break
            if time.monotonic() - t0 > self.s.build_timeout_s:
                self._kill()
                outcome = ("failed", "watchdog: build time limit reached", ("build_timeout", FRIENDLY["build_timeout"]))
                break
        self.child = None
        with db.session(self.s.db_path) as c:
            if outcome is None and rc == 0 and self.storage.exists(jid, "work/result.json"):
                r = json.loads(self.storage.get(jid, "work/result.json"))
                outcome = (r["state"], r["reason"], tuple(r["error"]) if r["error"] else None)
            elif outcome is None:
                attempts = c.execute("SELECT attempts FROM jobs WHERE id = ?", (jid,)).fetchone()
                again = attempts is not None and attempts[0] < self.s.max_attempts
                print(f"[worker {self.owner}] a build exited with code {rc}; see its work/build.log", file=sys.stderr)
                outcome = ("queued", "the build stopped; resuming from the last checkpoint", None) if again else \
                          ("failed", "the build stopped unexpectedly", ("build_crashed", FRIENDLY["build_crashed"]))
            jobqueue.finish(c, jid, self.owner, *outcome)

    def _kill(self):
        if self.child and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def run_forever(self):
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "stopping", True))
        signal.signal(signal.SIGINT, lambda *a: setattr(self, "stopping", True))
        swept = 0.0
        while not self.stopping:
            if time.monotonic() - swept > self.s.sweep_every_s:
                with db.session(self.s.db_path) as c:
                    retention.sweep(c, self.storage, self.s)
                swept = time.monotonic()
            if self.step() is None:
                time.sleep(self.s.worker_poll_s)


def main():
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"SnapPart worker starting: mode {s.mode}, data {s.data_dir}, max concurrent builds {s.max_concurrent_builds}", flush=True)
    Worker(s).run_forever()


if __name__ == "__main__":
    main()
