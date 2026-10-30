"""Load and failure test: real API processes (uvicorn), real worker processes, the MOCK pipeline and a FAKE Meta server. No real Meta call
and no real key: every child gets MODEL_API_KEY=loadtest-fake-key, and the fake server checks that this is the only key it ever sees.

Phase 1 (mock builds): --users simulated users arrive evenly over --duration seconds, each from its own client IP (X-Forwarded-For through
a trusted proxy address): create a part, upload 3 frames, capture check, start, read the job, sometimes long-poll events. At the same time
a hostile mix runs: oversized bodies, malformed JPEGs, resume-code guessing, an SSE storm, slow headers (slowloris) and stalled bodies.
Mid-run one worker is killed (with its build) and the job must resume on another worker. Then the queue drains and every acknowledged
job and frame is checked on disk and in the database.
Phase 2 (fake Meta): an API in real mode whose capture check asks the fake Muse Spark for a second opinion, under --meta-users users.

Run from the repo root with the project environment: `python scripts/loadtest.py` (defaults: 10,000 users in 60 s, 6 workers). Results
go to --out (JSON); docs/load-test.md is written by hand from them."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tests"))
FAKE_KEY = "loadtest-fake-key"


# ---------------------------------------------------------------- fake Meta server (its own process)

def serve_fake_meta(port: int, latency: float):
    from aiohttp import web
    stats = {"requests": 0, "bad_key": 0}

    async def chat(req):
        stats["requests"] += 1
        if req.headers.get("authorization") != f"Bearer {FAKE_KEY}":
            stats["bad_key"] += 1
            return web.json_response({"error": {"message": "bad key"}}, status=401)
        body = await req.json()
        n = sum(1 for m in body["messages"] if isinstance(m.get("content"), list) for c in m["content"] if c.get("type") != "text")
        await asyncio.sleep(latency)
        out = {"frames": [{"index": i, "mat_visible": True, "part_fully_visible": True, "blurry": False, "glare": False, "hand_in_frame": False}
                          for i in range(n)], "overall": "looks_fine", "tip": ""}
        return web.json_response({"choices": [{"message": {"content": json.dumps(out)}, "finish_reason": "stop"}], "usage": {"total_tokens": 100}})

    async def get_stats(req):
        return web.json_response(stats)
    app = web.Application(client_max_size=64 * 2 ** 20)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/stats", get_stats)
    web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)


# ---------------------------------------------------------------- processes

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def child_env(data_dir: Path, extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SNAPPART_") and k != "MODEL_API_KEY"}
    env.update({"SNAPPART_DATA_DIR": str(data_dir), "SNAPPART_ENVIRONMENT": "test", "MODEL_API_KEY": FAKE_KEY,
                "SNAPPART_TRUSTED_PROXIES": '["127.0.0.1"]', "SNAPPART_MIN_USABLE_FRAMES": "3", "PYTHONUNBUFFERED": "1"})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def start_api(env: dict, port: int, procs: int, log: Path) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", str(procs),
                          "--limit-concurrency", "4000", "--timeout-keep-alive", "5", "--h11-max-incomplete-event-size", "16384",
                          "--no-server-header", "--no-access-log", "--backlog", "4096"], cwd=BACKEND, env=env,
                         stdout=open(log, "ab"), stderr=subprocess.STDOUT)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.5):
                return p
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("API did not start; see " + str(log))


def start_worker(env: dict, log: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "app.worker"], cwd=BACKEND, env=env, stdout=open(log, "ab"), stderr=subprocess.STDOUT)


def rss_mb(p: subprocess.Popen) -> float:
    try:
        pp = psutil.Process(p.pid)
        return sum(x.memory_info().rss for x in [pp, *pp.children(recursive=True)]) / 2 ** 20
    except psutil.Error:
        return 0.0


def stop(p: subprocess.Popen):
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------------------------------------------------------------- client side

class Rec:
    def __init__(self):
        self.rows = []                                                   # (route, status or error, seconds, t)

    def add(self, route, status, dt):
        self.rows.append((route, status, dt, time.monotonic()))

    def summary(self, routes=None):
        out = {}
        for route in sorted({r[0] for r in self.rows} if routes is None else routes):
            rs = [r for r in self.rows if r[0] == route]
            lat = np.array([r[2] for r in rs]) * 1000
            st = {}
            for r in rs:
                st[str(r[1])] = st.get(str(r[1]), 0) + 1
            out[route] = {"n": len(rs), "status": st, **({"p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
                                                          "p99_ms": round(float(np.percentile(lat, 99)), 1), "max_ms": round(float(lat.max()), 1)} if len(rs) else {})}
        return out


async def call(sess, rec, route, method, url, **kw):
    t = time.monotonic()
    try:
        async with sess.request(method, url, **kw) as r:
            body = await r.read()
            rec.add(route, r.status, time.monotonic() - t)
            return r.status, body, r.headers
    except Exception as e:                                               # timeouts, resets: recorded, never hidden
        rec.add(route, type(e).__name__, time.monotonic() - t)
        return None, b"", {}


def ip_of(i: int, base=10) -> str:
    return f"{base}.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}"


async def user(sess, base, rec, i, frames, ledger, poll_share):
    h = {"x-forwarded-for": ip_of(i)}
    s, b, _ = await call(sess, rec, "POST /jobs", "POST", f"{base}/v1/jobs", headers=h)
    if s != 201:
        return
    jid = json.loads(b)["job_id"]
    ledger["jobs"][jid] = {"frames": {}, "start": None}
    for k in range(3):
        s, b, _ = await call(sess, rec, "PUT frame", "PUT", f"{base}/v1/jobs/{jid}/videos/1/frames/{k}", data=frames[(i + k) % len(frames)],
                             headers={**h, "content-type": "image/jpeg"})
        if s == 200:
            ledger["jobs"][jid]["frames"][k] = json.loads(b)["sha256"]
    s, b, _ = await call(sess, rec, "POST capture-check", "POST", f"{base}/v1/jobs/{jid}/capture-check", headers=h)
    if s != 200:
        return
    s, b, _ = await call(sess, rec, "POST start", "POST", f"{base}/v1/jobs/{jid}/start", json={"build_anyway": True}, headers=h)
    ledger["jobs"][jid]["start"] = s
    await call(sess, rec, "GET job", "GET", f"{base}/v1/jobs/{jid}", headers=h)
    if random.random() < poll_share:
        await call(sess, rec, "GET events (long-poll)", "GET", f"{base}/v1/jobs/{jid}/events?wait=2", headers=h)


async def hostile(sess, base, port, rec, ledger, duration, jpeg):
    big = b"\xff\xd8\xff" + os.urandom(5 * 2 ** 20)
    junk = [b"\xff\xd8\xff\xe0" + os.urandom(2000), b"PK\x03\x04" + os.urandom(2000), b"\xff\xd8\xff" + b"\0" * 5000]

    async def victim(ip):
        s, b, _ = await call(sess, rec, "hostile: create", "POST", f"{base}/v1/jobs", headers={"x-forwarded-for": ip})
        return json.loads(b)["job_id"] if s == 201 else None

    async def oversized(n):
        jid = await victim("172.16.0.1")
        for k in range(n):
            await call(sess, rec, "hostile: oversized frame", "PUT", f"{base}/v1/jobs/{jid}/videos/1/frames/{k % 60}", data=big,
                       headers={"x-forwarded-for": f"172.16.1.{k % 250}", "content-type": "image/jpeg"})

    async def malformed(n):
        jid = await victim("172.16.0.2")
        for k in range(n):
            await call(sess, rec, "hostile: malformed jpeg", "PUT", f"{base}/v1/jobs/{jid}/videos/1/frames/{k % 60}", data=junk[k % 3],
                       headers={"x-forwarded-for": f"172.16.2.{k % 250}", "content-type": "image/jpeg"})

    async def guess(n):
        words = (BACKEND / "app" / "wordlist.txt").read_text().split()
        for k in range(n):
            code = "-".join(random.sample(words, 3)) + f"-{random.randint(0, 99):02d}"
            await call(sess, rec, "hostile: resume guess", "GET", f"{base}/v1/resume/{code}", headers={"x-forwarded-for": f"172.16.3.{k % 20}"})

    async def sse_storm(n, hold):
        jids = [await victim(f"172.16.4.{k}") for k in range(5)]

        async def one(k):
            t = time.monotonic()
            try:
                async with sess.get(f"{base}/v1/jobs/{jids[k % 5]}/events", headers={"accept": "text/event-stream",
                                                                                      "x-forwarded-for": f"172.16.5.{k % 10}"}) as r:
                    if r.status == 200:
                        try:
                            await asyncio.wait_for(r.content.read(), hold)
                        except asyncio.TimeoutError:
                            pass
                    rec.add("hostile: SSE storm", r.status, time.monotonic() - t)
            except Exception as e:
                rec.add("hostile: SSE storm", type(e).__name__, time.monotonic() - t)
        await asyncio.gather(*[one(k) for k in range(n)])

    async def slowloris(n, until):
        socks, closed_by_server = [], 0
        for _ in range(n):
            try:
                r, w = await asyncio.open_connection("127.0.0.1", port)
                w.write(b"GET /healthz HTTP/1.1\r\nHost: x\r\n")
                socks.append((r, w))
            except OSError:
                ledger["slowloris_connect_failed"] = ledger.get("slowloris_connect_failed", 0) + 1
        while time.monotonic() < until:
            await asyncio.sleep(5)
            for r, w in socks:
                if w.is_closing() or r.at_eof():
                    continue
                try:
                    w.write(b"X-a: b\r\n")
                    await w.drain()
                except (ConnectionError, OSError):
                    closed_by_server += 1
        open_at_end = sum(1 for r, w in socks if not w.is_closing() and not r.at_eof())
        for r, w in socks:
            w.close()
        ledger["slowloris"] = {"opened": len(socks), "closed_by_server": closed_by_server, "still_open_at_end": open_at_end}

    async def stalled_bodies(n):
        jid = await victim("172.16.0.3")

        async def one(k):
            t = time.monotonic()
            try:
                r, w = await asyncio.open_connection("127.0.0.1", port)
                w.write(f"PUT /v1/jobs/{jid}/videos/1/frames/{k} HTTP/1.1\r\nHost: x\r\nContent-Type: image/jpeg\r\nContent-Length: 100000\r\n"
                        f"X-Forwarded-For: 172.16.6.{k}\r\n\r\n".encode() + jpeg[:1000])
                await w.drain()
                line = await asyncio.wait_for(r.readline(), 60)
                rec.add("hostile: stalled body", int(line.split()[1]) if line else "closed", time.monotonic() - t)
                w.close()
            except Exception as e:
                rec.add("hostile: stalled body", type(e).__name__, time.monotonic() - t)
        await asyncio.gather(*[one(k) for k in range(n)])

    until = time.monotonic() + duration
    await asyncio.gather(oversized(200), malformed(600), guess(2000), sse_storm(300, 10), slowloris(300, until), stalled_bodies(50))


async def monitor(base, db_path, api, workers, samples, stop_evt):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        t0 = time.monotonic()
        while not stop_evt.is_set():
            t = time.monotonic()
            try:
                async with s.get(f"{base}/healthz") as r:
                    health = r.status
            except Exception as e:
                health = type(e).__name__
            hl = time.monotonic() - t
            try:
                q = dict(con.execute("SELECT state, COUNT(*) FROM jobs WHERE state IN ('queued', 'building') GROUP BY state").fetchall())
            except sqlite3.Error:
                q = {}
            samples.append({"t": round(t - t0, 1), "health": health, "health_ms": round(hl * 1000, 1), "queued": q.get("queued", 0),
                            "building": q.get("building", 0), "api_rss_mb": round(rss_mb(api), 1),
                            "workers_rss_mb": round(sum(rss_mb(w) for w in workers if w.poll() is None), 1)})
            await asyncio.sleep(1.0)


async def kill_one_worker(db_path, workers, ledger, at, restart):
    await asyncio.sleep(at)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    for _ in range(60):
        row = con.execute("SELECT id, lease_owner FROM jobs WHERE state = 'building' ORDER BY lease_until LIMIT 1").fetchone()
        if row:
            for w in workers:
                if w.poll() is None and f":{w.pid}:" in row[1]:
                    for c in psutil.Process(w.pid).children(recursive=True):        # the build dies with its worker, as in a crash
                        c.kill()
                    w.kill()
                    w.wait()
                    workers[workers.index(w)] = restart()                  # as the container's restart policy would
                    ledger["killed"] = {"job_id": row[0], "worker_pid": w.pid, "at_s": at, "replacement_started": True}
                    return
        await asyncio.sleep(0.5)
    ledger["killed"] = None


async def phase1(a, base, port, data, api, workers, restart):
    from test_app_stage2 import frame_jpeg
    frames = [frame_jpeg(az, 30) for az in range(0, 360, 30)]
    rec, ledger, samples = Rec(), {"jobs": {}}, []
    stop_evt = asyncio.Event()
    conn = aiohttp.TCPConnector(limit=a.connections, limit_per_host=a.connections)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=60)) as sess:
        mon = asyncio.create_task(monitor(base, data / "snappart.db", api, workers, samples, stop_evt))
        killer = asyncio.create_task(kill_one_worker(data / "snappart.db", workers, ledger, a.duration / 2, restart))
        host = asyncio.create_task(hostile(sess, base, port, rec, ledger, a.duration, frames[0]))
        t0 = time.monotonic()
        tasks = []
        for i in range(a.users):
            delay = t0 + i * a.duration / a.users - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(user(sess, base, rec, i, frames, ledger, 0.1)))
        await asyncio.gather(*tasks)
        load_s = time.monotonic() - t0
        await host
        await killer
        drain_t = time.monotonic()
        con = sqlite3.connect(f"file:{data / 'snappart.db'}?mode=ro", uri=True, timeout=30)
        while time.monotonic() - drain_t < a.drain_s:
            if con.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'building')").fetchone()[0] == 0:
                break
            await asyncio.sleep(1)
        drained_s = time.monotonic() - drain_t
        stop_evt.set()
        await mon
    return rec, ledger, samples, load_s, drained_s


def integrity(data: Path, ledger: dict) -> dict:
    con = sqlite3.connect(data / "snappart.db")
    out = {"integrity_check": con.execute("PRAGMA integrity_check").fetchone()[0], "jobs_acknowledged": len(ledger["jobs"]),
           "jobs_missing": 0, "frames_acknowledged": 0, "frames_missing_or_corrupt": 0, "start_queued": 0, "start_shed_503": 0,
           "start_other": {}, "final_states_of_queued": {}, "final_states_of_shed": {}}
    for jid, j in ledger["jobs"].items():
        row = con.execute("SELECT state, attempts FROM jobs WHERE id = ?", (jid,)).fetchone()
        if row is None:
            out["jobs_missing"] += 1
            continue
        for k, sha in j["frames"].items():
            out["frames_acknowledged"] += 1
            p = data / "jobs" / jid / "videos" / "1" / "frames" / f"{k:04d}.jpg"
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                out["frames_missing_or_corrupt"] += 1
        key = "final_states_of_queued" if j["start"] == 200 else "final_states_of_shed" if j["start"] == 503 else None
        if j["start"] == 200:
            out["start_queued"] += 1
        elif j["start"] == 503:
            out["start_shed_503"] += 1
        elif j["start"] is not None:
            out["start_other"][str(j["start"])] = out["start_other"].get(str(j["start"]), 0) + 1
        if key:
            out[key][row[0]] = out[key].get(row[0], 0) + 1
    k = ledger.get("killed")
    if k:
        row = con.execute("SELECT state, attempts FROM jobs WHERE id = ?", (k["job_id"],)).fetchone()
        ev = [json.loads(r[1]) | {"type": r[0]} for r in con.execute("SELECT type, payload FROM events WHERE job_id = ? ORDER BY seq", (k["job_id"],))]
        started = [e["stage"] for e in ev if e["type"] == "stage_started"]
        out["killed_job"] = {"final_state": row[0], "attempts": row[1], "stage_starts": started,
                             "stages_run_twice": sorted({s for s in started if started.count(s) > 1}),
                             "resumed_from_checkpoint": len(set(started)) < len(started) or row[1] > 1}
    return out


async def phase2(a, base):
    from test_app_stage2 import frame_jpeg
    frames = [frame_jpeg(az, 30) for az in range(0, 360, 90)]
    rec = Rec()

    async def u(sess, i):
        h = {"x-forwarded-for": ip_of(i, 11)}
        s, b, _ = await call(sess, rec, "POST /jobs", "POST", f"{base}/v1/jobs", headers=h)
        if s != 201:
            return None
        jid = json.loads(b)["job_id"]
        for k, f in enumerate(frames):
            await call(sess, rec, "PUT frame", "PUT", f"{base}/v1/jobs/{jid}/videos/1/frames/{k}", data=f, headers={**h, "content-type": "image/jpeg"})
        s, b, _ = await call(sess, rec, "POST capture-check (+ fake Spark)", "POST", f"{base}/v1/jobs/{jid}/capture-check", headers=h)
        return json.loads(b).get("second_opinion", {}).get("status") if s == 200 else None
    conn = aiohttp.TCPConnector(limit=a.connections)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=120)) as sess:
        t0, tasks = time.monotonic(), []
        for i in range(a.meta_users):
            delay = t0 + i * a.meta_duration / a.meta_users - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(u(sess, i)))
        ops = await asyncio.gather(*tasks)
    st = {}
    for o in ops:
        st[str(o)] = st.get(str(o), 0) + 1
    return rec, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10_000)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--api-procs", type=int, default=4)
    ap.add_argument("--connections", type=int, default=1000)
    ap.add_argument("--mock-build-s", type=float, default=3.0)
    ap.add_argument("--drain-s", type=float, default=600.0)
    ap.add_argument("--meta-users", type=int, default=1000)
    ap.add_argument("--meta-duration", type=float, default=30.0)
    ap.add_argument("--meta-latency", type=float, default=0.4)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "backend" / "storage" / "loadtest.json")
    ap.add_argument("--fake-meta", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.fake_meta:
        return serve_fake_meta(a.fake_meta, a.meta_latency)

    root = a.data_dir or Path(tempfile.mkdtemp(prefix="snappart-load-"))
    data1, data2 = root / "phase1", root / "phase2"
    for d in (data1, data2):
        d.mkdir(parents=True, exist_ok=True)
    from app import mock
    t = time.monotonic()
    mock.bundle(data1)                                                   # made once before the clock starts (about 30 s)
    print(f"mock bundle ready in {time.monotonic() - t:.0f} s; data in {root}", flush=True)

    meta_port, port1, port2 = free_port(), free_port(), free_port()
    fake = subprocess.Popen([sys.executable, __file__, "--fake-meta", str(meta_port), "--meta-latency", str(a.meta_latency)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    common = {"SNAPPART_META_BASE_URL": f"http://127.0.0.1:{meta_port}/v1", "SNAPPART_LEASE_S": 15, "SNAPPART_HEARTBEAT_S": 3,
              "SNAPPART_WORKER_POLL_S": 0.5}
    env1 = child_env(data1, common | {"SNAPPART_MOCK": 1, "SNAPPART_MOCK_BUILD_S": a.mock_build_s, "SNAPPART_MAX_CONCURRENT_BUILDS": a.workers})
    results = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "fake_meta"},
               "machine": {"cpus": os.cpu_count(), "memory_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "platform": sys.platform}}
    api = start_api(env1, port1, a.api_procs, root / "api1.log")
    workers = [start_worker(env1, root / f"worker{k}.log") for k in range(a.workers)]
    try:
        rec, ledger, samples, load_s, drained_s = asyncio.run(phase1(a, f"http://127.0.0.1:{port1}", port1, data1, api, workers,
                                                                          lambda: start_worker(env1, root / "worker-replacement.log")))
        results["phase1"] = {"load_seconds": round(load_s, 1), "drain_seconds": round(drained_s, 1), "api_alive": api.poll() is None,
                             "requests": rec.summary(), "samples": samples, "slowloris": ledger.get("slowloris"), "killed": ledger.get("killed"),
                             "integrity": integrity(data1, ledger)}
        done = [r for r in rec.rows if not r[0].startswith("hostile")]
        results["phase1"]["user_requests"] = len(done)
        results["phase1"]["user_requests_per_s"] = round(len(done) / load_s, 1)
    finally:
        for w in workers:
            stop(w)
        stop(api)
    builds = sqlite3.connect(data1 / "snappart.db").execute(
        "SELECT COUNT(*), MIN(build_started_at), MAX(build_finished_at) FROM jobs WHERE build_finished_at IS NOT NULL").fetchone()
    results["phase1"]["mock_builds"] = {"finished": builds[0], "span_s": round((builds[2] or 0) - (builds[1] or 0), 1),
                                        "per_worker_per_min": round(builds[0] / max(1e-9, (builds[2] - builds[1])) * 60 / a.workers, 2) if builds[0] else 0}

    env2 = child_env(data2, common | {"SNAPPART_MOCK": 0})
    api2 = start_api(env2, port2, a.api_procs, root / "api2.log")
    try:
        rec2, opinions = asyncio.run(phase2(a, f"http://127.0.0.1:{port2}"))
    finally:
        stop(api2)
    calls = sqlite3.connect(data2 / "snappart.db").execute("SELECT model, ok, COUNT(*) FROM model_calls GROUP BY model, ok").fetchall()
    import urllib.request
    fstats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{meta_port}/stats").read())
    fake.terminate()
    results["phase2"] = {"requests": rec2.summary(), "second_opinion_status": opinions, "model_calls": [list(c) for c in calls], "fake_server": fstats}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    p1 = results["phase1"]
    print(json.dumps({"phase1": {k: v for k, v in p1.items() if k != "samples"}, "phase2": results["phase2"]}, indent=1))
    print(f"peak queue {max(s['queued'] for s in p1['samples'])}, peak API RSS {max(s['api_rss_mb'] for s in p1['samples'])} MB, "
          f"health failures {sum(s['health'] != 200 for s in p1['samples'])} / {len(p1['samples'])}; written to {a.out}")


if __name__ == "__main__":
    main()
