"""Backend stage 2: resume codes and lockout, idempotent frame uploads, deterministic capture check on synthetic mat frames, queue
admission and fairness, leases and requeue, the worker running isolated build subprocesses (mock), SSE replay, delete, shared limits.
Synthetic only: frames are the printed mat rendered by capture/synth.py."""
import json
import math
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import capture_check, db, events, jobqueue, jobs, limits, resume
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage
from app.worker import Worker
from capture import synth

F, SIZE = 1400.0, (1920, 1080)
CANVAS = None


def frame_jpeg(az_deg, el_deg, dist=330.0, blur=0.0):
    """The printed mat seen from a camera at (azimuth, elevation) around the page centre, as JPEG bytes."""
    global CANVAS
    if CANVAS is None:
        from capture import mat
        CANVAS = mat.render_mat(8.0)
    a, e = math.radians(az_deg), math.radians(el_deg)
    c = np.array([*synth.CENTRE, 3.0])
    R, t = synth.look_at(c + dist * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]), c)
    K = np.array([[F, 0, SIZE[0] / 2], [0, F, SIZE[1] / 2], [0, 0, 1.0]])
    img = synth.render_mat_view(CANVAS, 8.0, K, R, t, SIZE).astype(np.float32) * 0.82 + 12          # paper ~221, not clipped white
    img = np.clip(img + np.random.default_rng(int(az_deg * 10 + el_deg)).normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data2")


def make(data_dir, **kw):
    base = dict(data_dir=data_dir, environment="test", mock=True, mock_build_s=0.4, heartbeat_s=0.2, worker_poll_s=0.05, sse_poll_s=0.05,
                rate_create_per_hour=1000, rate_start_per_hour=1000, rate_resume_per_hour=1000)
    s = Settings(**{**base, **kw})
    return s, TestClient(create_app(s), raise_server_exceptions=False)


@pytest.fixture
def clock(monkeypatch):
    t = [1_000_000.0]
    monkeypatch.setattr(db, "now", lambda: t[0])
    return t


def test_resume_codes_and_lockout(data_dir, clock):
    s, cl = make(data_dir, resume_max_fails=3, resume_lock_base_s=60)
    r = cl.post("/v1/jobs").json()
    code, jid = r["resume_code"], r["job_id"]
    w = code.split("-")
    assert len(w) == 4 and all(x in resume.WORDS for x in w[:3]) and len(w[3]) == 2 and w[3].isdigit()
    with db.session(s.db_path) as c:
        stored = c.execute("SELECT resume_hash FROM jobs WHERE id = ?", (jid,)).fetchone()[0]
    assert code not in stored and len(stored) == 64                           # only a hash is stored
    assert cl.get(f"/v1/resume/{code}").json() == {"job_id": jid, "state": "created"}
    assert cl.get(f"/v1/resume/{' '.join(w).upper()}").json()["job_id"] == jid  # spaces and capitals are fine
    wrong = "-".join(w[:3]) + f"-{(int(w[3]) + 1) % 100:02d}"
    a, b = cl.get(f"/v1/resume/{wrong}"), cl.get("/v1/resume/not-a-code-at-all")
    assert a.status_code == b.status_code == 404 and a.json()["error"]["message"] == b.json()["error"]["message"]
    r = cl.get(f"/v1/resume/{wrong}")                                         # third miss from this client: locked
    r = cl.get(f"/v1/resume/{code}")
    assert r.status_code == 429 and r.json()["error"]["code"] == "locked" and int(r.headers["Retry-After"]) >= 60
    clock[0] += 61
    assert cl.get(f"/v1/resume/{code}").status_code == 200                   # lock over
    cl.get(f"/v1/resume/{wrong}")
    assert int(cl.get(f"/v1/resume/{code}").headers["Retry-After"]) > 100      # doubled


def test_uploads_idempotent(data_dir):
    a, b = frame_jpeg(0, 50), frame_jpeg(30, 50)
    big = max(len(a), len(b))
    s, cl = make(data_dir, max_frame_bytes=big + 1000, max_job_bytes=len(b) + 2 * len(a) + 10, max_frames_per_video=5)
    jid = cl.post("/v1/jobs").json()["job_id"]
    put = lambda v, i, body, ct="image/jpeg": cl.put(f"/v1/jobs/{jid}/videos/{v}/frames/{i}", content=body, headers={"content-type": ct})
    r = put(1, 0, a)
    assert r.status_code == 200 and r.json()["stored"] and cl.get(f"/v1/jobs/{jid}").json()["state"] == "uploading"
    assert put(1, 0, a).json()["stored"] is False                              # same body: no-op
    assert put(1, 0, b).json()["stored"] is True                               # different body before start: replaces
    assert put(1, 1, b"\x89PNG....", "image/jpeg").status_code == 415
    assert put(1, 1, a, "image/png").status_code == 415
    assert put(1, 5, a).status_code == 422                                     # index beyond the per-video limit
    assert put(4, 0, a).status_code == 422
    assert put(1, 1, a + b"\0" * (big + 1000)).status_code == 413              # frame too large
    assert put(1, 1, b"\xff\xd8\xff" + b"\0" * 100).status_code == 422        # JPEG magic but not an image
    for i in (1, 2):
        put(2, i, a)
    r = put(3, 0, a)
    assert r.status_code == 413 and r.json()["error"]["code"] == "job_too_large"
    up = cl.get(f"/v1/jobs/{jid}/uploads").json()
    assert up["videos"] == {"1": [0], "2": [1, 2], "3": []} and up["frames"] == 3


def upload_video(cl, jid, v, azs, el, blur=0.0):
    for i, az in enumerate(azs):
        r = cl.put(f"/v1/jobs/{jid}/videos/{v}/frames/{i}", content=frame_jpeg(az, el, blur=blur), headers={"content-type": "image/jpeg"})
        assert r.status_code == 200, r.text


def test_capture_check_and_start(data_dir):
    s, cl = make(data_dir, min_usable_frames=8)
    jid = cl.post("/v1/jobs").json()["job_id"]
    assert cl.post(f"/v1/jobs/{jid}/capture-check").status_code == 409        # nothing uploaded yet
    upload_video(cl, jid, 1, range(0, 360, 20), 60)                           # high orbit, all round
    upload_video(cl, jid, 2, range(0, 200, 20), 25)                           # low orbit, misses 200..360 deg
    t0 = time.perf_counter()
    r = cl.post(f"/v1/jobs/{jid}/capture-check").json()
    dt = time.perf_counter() - t0
    assert dt < 5.0, dt
    v = r["videos"]
    assert v["1"]["verdict"] == "ok" and v["1"]["usable_frames"] == 18
    assert v["2"]["verdict"] == "weak" and any("more seconds" in t for t in v["2"]["tips"])
    assert v["3"]["verdict"] == "missing" and r["usable"] is False
    assert sum(map(sum, r["coverage_map"]["counts"])) == 28
    j = cl.get(f"/v1/jobs/{jid}").json()
    assert j["state"] == "ready_to_build" and j["capture_check"]["verdict"] == "unusable"
    r = cl.post(f"/v1/jobs/{jid}/start")
    assert r.status_code == 409 and r.json()["error"]["code"] == "capture_unusable" and "Video 3" in r.json()["error"]["message"]
    r = cl.post(f"/v1/jobs/{jid}/start", json={"build_anyway": True})
    assert r.json()["state"] == "queued" and "although the capture check" in r.json()["warnings"][0]
    blurry = capture_check.frame_stats(frame_jpeg(0, 50, blur=6), s)
    assert blurry["blurry"] and not blurry["usable"] and capture_check.frame_stats(frame_jpeg(0, 50), s)["usable"]


def ready_job(c, s, st, ip=None):
    jid = jobs.create(c, s.mode, ip)
    for to in ("uploading", "checking", "ready_to_build"):
        jobs.transition(c, jid, to)
    st.put(jid, "capture_check.json", json.dumps({"usable": True, "verdict": "usable", "videos": {}}).encode())
    return jid


def test_queue_admission_fairness_and_leases(data_dir, clock):
    s, cl = make(data_dir, max_queue_len=3, max_queued_per_ip=2, max_concurrent_builds=2, max_attempts=2, lease_s=30)
    st = LocalStorage(data_dir)
    with db.session(s.db_path) as c, db.tx(c):
        c.execute("UPDATE jobs SET state = 'cancelled' WHERE state IN ('queued', 'building')")      # earlier tests' jobs
    with db.session(s.db_path) as c:
        a1, a2, a3 = (ready_job(c, s, st, "ipA") for _ in range(3))
        b1, c1, d1 = ready_job(c, s, st, "ipB"), ready_job(c, s, st, "ipC"), ready_job(c, s, st, "ipD")
        jobqueue.enqueue(c, a1, s, "ipA")
        clock[0] += 1
        jobqueue.enqueue(c, a2, s, "ipA")
        with pytest.raises(Exception) as e:
            jobqueue.enqueue(c, a3, s, "ipA")                                   # per-client cap
        assert e.value.status == 429
        clock[0] += 1
        jobqueue.enqueue(c, b1, s, "ipB")
        with pytest.raises(Exception) as e:
            jobqueue.enqueue(c, c1, s, "ipC")                                   # queue full: 503 with Retry-After and how many are ahead
        assert e.value.status == 503 and e.value.retry_after_s >= 30 and "3 builds are waiting" in e.value.message
        assert jobqueue.position(c, b1, s)["queue_position"] == 3
        first = jobqueue.claim(c, "w1", s)
        second = jobqueue.claim(c, "w2", s)
        assert (first, second) == (a1, b1)                                      # a2 waits: one active build per client
        assert jobqueue.claim(c, "w3", s) is None                               # and at most 2 builds at once
        assert jobqueue.heartbeat(c, a1, "w1", s) and not jobqueue.heartbeat(c, a1, "w2", s)
        clock[0] += 31                                                          # w1 died: lease runs out
        assert jobqueue.reap(c, s) == [a1, b1] and jobs.get(c, a1)["state"] == "queued"
        assert jobqueue.claim(c, "w4", s) == a1
        clock[0] += 31
        jobqueue.reap(c, s)
        row = jobs.get(c, a1)
        assert row["state"] == "failed" and row["error_code"] == "build_interrupted"            # max_attempts reached
        assert not jobqueue.finish(c, a1, "w4", "needs_input", "late")          # a lost lease cannot write the outcome


def test_worker_mock_build_resume_and_watchdog(data_dir):
    s, cl = make(data_dir, max_concurrent_builds=4)
    st = LocalStorage(data_dir)
    with db.session(s.db_path) as c, db.tx(c):
        c.execute("UPDATE jobs SET state = 'cancelled' WHERE state IN ('queued', 'building')")
    with db.session(s.db_path) as c:
        jid = ready_job(c, s, st)
        jobqueue.enqueue(c, jid, s, None)
    w = Worker(s, "test-worker")
    assert w.step() == jid                                                      # build subprocess (mock), first run makes the bundle
    j = cl.get(f"/v1/jobs/{jid}").json()
    assert j["state"] == "needs_input" and "outputs/model.stl" in j["outputs"]
    with db.session(s.db_path) as c:
        assert jobqueue.avg_build_s(c) is not None                              # measured build time now feeds the ETA
        # resume from checkpoint: a requeued build skips the stages it already finished
        jid2 = ready_job(c, s, st)
        st.put(jid2, "work/checkpoint.json", json.dumps({"done": ["marker detection", "mat self-calibration", "SAM"]}).encode())
        jobqueue.enqueue(c, jid2, s, None)
    w.step()
    with db.session(s.db_path) as c:
        started = [e["data"]["stage"] for e in events.after(c, jid2) if e["type"] == "stage_started"]
        assert "SAM" not in started and "joint carve" in started and jobs.get(c, jid2)["state"] == "needs_input"
    # watchdog: a build longer than the limit is killed and the job fails with a plain reason
    s2, _ = make(data_dir, mock_build_s=60.0, build_timeout_s=1)
    with db.session(s.db_path) as c:
        jid3 = ready_job(c, s2, st)
        jobqueue.enqueue(c, jid3, s2, None)
    t0 = time.time()
    Worker(s2, "w-timeout").step()
    with db.session(s.db_path) as c:
        row = jobs.get(c, jid3)
    assert row["state"] == "failed" and row["error_code"] == "build_timeout" and time.time() - t0 < 10
    # real mode: the build is not wired to the Meta client yet, and says so
    s3, _ = make(data_dir, mock=False)
    with db.session(s.db_path) as c:
        jid4 = ready_job(c, s3, st)
        jobqueue.enqueue(c, jid4, s3, None)
    Worker(s3, "w-real").step()
    with db.session(s.db_path) as c:
        row = jobs.get(c, jid4)
    assert row["state"] == "failed" and row["error_code"] == "real_build_not_available"


def test_sse_replay_longpoll_and_delete(data_dir):
    s, cl = make(data_dir)
    r = cl.post("/v1/jobs").json()
    jid = r["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "failed"):
            jobs.transition(c, jid, to, "test", error=("x", "y") if to == "failed" else None)
    with cl.stream("GET", f"/v1/jobs/{jid}/events", headers={"accept": "text/event-stream", "last-event-id": "2"}) as resp:
        body = "".join(resp.iter_text())
    ids = [int(line[4:]) for line in body.splitlines() if line.startswith("id: ")]
    assert resp.headers["content-type"].startswith("text/event-stream") and ids == [3, 4]          # replay after Last-Event-ID
    t0 = time.perf_counter()
    lp = cl.get(f"/v1/jobs/{jid}/events?after=4&wait=1").json()
    assert lp["events"] == [] and time.perf_counter() - t0 < 0.5                # terminal job: no waiting
    jid2 = cl.post("/v1/jobs").json()["job_id"]

    def later():
        time.sleep(0.3)
        with db.session(s.db_path) as c:
            jobs.transition(c, jid2, "uploading", "late event")
    threading.Thread(target=later).start()
    lp = cl.get(f"/v1/jobs/{jid2}/events?after=1&wait=3").json()                # long-poll returns when the event arrives
    assert lp["events"][0]["data"]["to"] == "uploading"
    st = LocalStorage(data_dir)
    st.put(jid2, "videos/1/frames/0000.jpg", b"x")
    code = r["resume_code"]
    assert cl.delete(f"/v1/jobs/{jid2}").json() == {"deleted": True}
    assert cl.get(f"/v1/jobs/{jid2}").status_code == 404 and st.list(jid2) == []
    with db.session(s.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM events WHERE job_id = ?", (jid2,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] >= 1
    assert cl.delete(f"/v1/jobs/{jid}").status_code == 200 and cl.get(f"/v1/resume/{code}").status_code == 404


def test_shared_limits(data_dir, clock):
    s, _ = make(data_dir)
    held, order = threading.Event(), []

    def hold(i):
        with limits.slot(data_dir / "locks", 2):
            order.append(i)
            held.wait(2)
    ts = [threading.Thread(target=hold, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    time.sleep(0.3)
    assert len(order) == 2                                                      # the third caller waits for a free slot
    held.set()
    for t in ts:
        t.join()
    assert len(order) == 3
    with db.session(s.db_path) as c:
        slept = []
        assert limits.rate(c, "sam:test", rps=2.0, burst=2, sleep=lambda w: (slept.append(w), clock.__setitem__(0, clock[0] + w))) == 0
        limits.rate(c, "sam:test", 2.0, 2, sleep=lambda w: (slept.append(w), clock.__setitem__(0, clock[0] + w)))
        limits.rate(c, "sam:test", 2.0, 2, sleep=lambda w: (slept.append(w), clock.__setitem__(0, clock[0] + w)))
        assert abs(sum(slept) - 0.5) < 1e-6                                     # third call within the burst waited 1 / rps
        br = limits.Breaker(c, "sam", failures=3, open_s=30)
        for _ in range(3):
            br.check()
            br.record(False)
        with pytest.raises(limits.Unavailable) as e:
            br.check()
        assert "paused" in e.value.reason and e.value.retry_after_s == 30
        clock[0] += 31
        br.check()
        br.record(True)
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError
        return "ok"
    assert limits.retry(flaky, sleep=waits.append, rng=lambda: 1.0) == "ok" and waits == [0.5, 1.0]


def test_api_responsive_with_long_queue(data_dir):
    s, cl = make(data_dir, max_queue_len=10_000)
    with db.session(s.db_path) as c, db.tx(c):
        t = db.now()
        c.executemany("INSERT INTO jobs (id, state, mode, created_at, updated_at, state_changed_at, enqueued_at) VALUES (?, 'queued', 'mock', ?, ?, ?, ?)",
                      [(jobs.new_id(), t, t, t, t + i) for i in range(3000)])
        last = c.execute("SELECT id FROM jobs WHERE state = 'queued' ORDER BY enqueued_at DESC LIMIT 1").fetchone()[0]
    t0 = time.perf_counter()
    for _ in range(20):
        assert cl.get("/healthz").status_code == 200
    j = cl.get(f"/v1/jobs/{last}").json()
    assert (time.perf_counter() - t0) < 2.0 and j["queue_position"] >= 3000
    with db.session(s.db_path) as c, db.tx(c):
        c.execute("UPDATE jobs SET state = 'cancelled' WHERE state = 'queued'")
