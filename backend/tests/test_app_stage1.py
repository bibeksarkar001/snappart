"""Backend stage 1: state machine, error envelope, mock mode exposure, truthful model state, mock build, storage keys, migrations.
Synthetic only (no private data); the mock bundle is generated once for the module."""
import json

import pytest
from fastapi.testclient import TestClient

from app import db, events, jobs, mock
from app.errors import ApiError
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage, StorageKeyError


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


def make(data_dir, **kw):
    s = Settings(data_dir=data_dir, environment="test", mock=True, mock_build_s=0.0, **kw)
    return s, TestClient(create_app(s), raise_server_exceptions=False)


def test_migrations_idempotent(data_dir):
    with db.session(data_dir / "m.db") as c:
        assert db.migrate(c) == len(db.MIGRATIONS) and db.migrate(c) == len(db.MIGRATIONS)
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"jobs", "uploads", "events", "answers", "corrections", "model_calls", "rate_limits", "budget_ledger"} <= names


def test_create_and_headers(data_dir):
    s, cl = make(data_dir)
    r = cl.post("/v1/jobs")
    assert r.status_code == 201 and len(r.json()["job_id"]) == 32 and r.json()["mode"] == "mock"
    assert r.headers["X-SnapPart-Mode"] == "mock" and r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Request-ID"]
    assert cl.get("/readyz").json()["mock"] is True
    assert cl.get("/healthz").headers["X-SnapPart-Mode"] == "mock"
    j = cl.get(f"/v1/jobs/{r.json()['job_id']}").json()
    assert j["state"] == "created" and j["queue_position"] is None


def test_state_machine(data_dir):
    s, cl = make(data_dir)
    jid = cl.post("/v1/jobs").json()["job_id"]
    r = cl.post(f"/v1/jobs/{jid}/start")                                     # created -> queued is illegal
    assert r.status_code == 409 and r.json()["error"]["code"] == "illegal_transition" and r.json()["error"]["request_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build"):
            jobs.transition(c, jid, to, "test")
        with pytest.raises(ApiError) as e:
            jobs.transition(c, jid, "done")
        assert e.value.status == 409 and jobs.get(c, jid)["state"] == "ready_to_build"
    assert cl.post(f"/v1/jobs/{jid}/start").json()["error"]["code"] == "check_needed"
    LocalStorage(data_dir).put(jid, "capture_check.json", json.dumps({"usable": True, "verdict": "usable", "videos": {}}).encode())
    assert cl.post(f"/v1/jobs/{jid}/start").json()["state"] == "queued"
    assert cl.post(f"/v1/jobs/{jid}/start").status_code == 409                # not twice
    ev = cl.get(f"/v1/jobs/{jid}/events").json()["events"]
    assert [e["data"]["to"] for e in ev if e["type"] == "state_changed"] == ["created", "uploading", "checking", "ready_to_build", "queued"]
    assert all(e["ts"] > 0 for e in ev) and [e["seq"] for e in ev] == list(range(1, len(ev) + 1))
    assert cl.get(f"/v1/jobs/{jid}/events?after=3").json()["events"][0]["seq"] == 4
    assert cl.get(f"/v1/jobs/{jid}").json()["queue_position"] >= 1
    assert all(t <= set(jobs.STATES) for t in jobs.TRANSITIONS.values()) and not jobs.TRANSITIONS["expired"]


def test_error_envelope_never_leaks(data_dir):
    s, cl = make(data_dir)
    app = cl.app

    @app.get("/boom")
    def boom():
        raise RuntimeError("/Users/secret/path SELECT * FROM jobs")
    r = cl.get("/boom")
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal"
    assert "secret" not in r.text and "SELECT" not in r.text and r.headers["X-SnapPart-Mode"] == "mock"
    for path in ("/v1/jobs/" + "a" * 32, "/v1/jobs/..%2F..%2Fetc", "/nope"):
        r = cl.get(path)
        assert r.status_code == 404 and set(r.json()["error"]) == {"code", "message", "retry_after_s", "request_id"}
    r = cl.post("/v1/jobs", json={"part_name": "x", "evil": 1})                # extra fields are rejected
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_mock_never_silent_in_production(data_dir):
    with pytest.raises(ValueError):
        Settings(data_dir=data_dir, environment="production", mock=True)
    s = Settings(data_dir=data_dir, environment="production", mock=True, allow_mock_in_production=True)
    r = TestClient(create_app(s)).get("/readyz")
    assert r.json()["mode"] == "mock" and r.headers["X-SnapPart-Mode"] == "mock"
    assert repr(Settings(data_dir=data_dir, MODEL_API_KEY="sk-test-not-a-real-key")).count("sk-test") == 0   # the key never prints


def test_meta_models_truthful(data_dir):
    s, cl = make(data_dir)
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c, db.tx(c):
        c.execute("INSERT INTO model_calls (job_id, model, endpoint, ok, ts) VALUES (?, 'sam-3.1', '/responses', 1, ?)", (jid, db.now()))
        c.execute("INSERT INTO model_calls (job_id, model, endpoint, ok, ts) VALUES (?, 'muse-spark-1.3', '/responses', 0, ?)", (jid, db.now()))
    m = {x["id"]: x for x in cl.get("/v1/meta/models").json()["models"]}
    assert not any(x["running"] for x in m.values()) and m["sam-3.1"]["calls_ok"] == 1          # mock: never "running"
    real = Settings(data_dir=data_dir, environment="test")
    rc = TestClient(create_app(real))
    m = {x["id"]: x for x in rc.get(f"/v1/meta/models?job_id={jid}").json()["models"]}
    assert m["sam-3.1"]["running"] and not m["muse-spark-1.3"]["running"] and not m["muse-voice-transcribe-1.0"]["running"]
    assert m["muse-spark-1.3"]["calls_failed"] == 1


def test_mock_build(data_dir):
    s, cl = make(data_dir)
    jid = cl.post("/v1/jobs").json()["job_id"]
    st = LocalStorage(data_dir)
    slept = []
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        nxt = mock.run_build(c, st, s.model_copy(update={"mock_build_s": 5.0}), jid, sleep=slept.append)
        ev = events.after(c, jid)
    assert nxt[0] == "needs_input" and abs(sum(slept) - 5.0) < 1e-6
    started = [e["data"]["stage"] for e in ev if e["type"] == "stage_started"]
    assert started == [n for n, _ in mock.STAGES] and [e for e in ev if e["type"] == "progress"][-1]["data"]["fraction"] == 1.0
    out = cl.get(f"/v1/jobs/{jid}").json()["outputs"]
    assert {"outputs/model.stl", "outputs/report.json", "outputs/model.npz", "outputs/model.json"} <= set(out)
    rep = json.loads(st.get(jid, "outputs/report.json"))
    assert rep["mode"] == "mock" and rep["synthetic"] and rep["meta_models_ran"] == [] and "Mock mode" in rep["warnings"][0]
    for q in rep["questions"]:
        assert f"outputs/cards/{q['id']}/card.png" in out and json.loads(st.get(jid, f"outputs/cards/{q['id']}/card.json"))["mock"]
    stl = st.get(jid, "outputs/model.stl")
    assert len(stl) == 84 + 50 * int.from_bytes(stl[80:84], "little")         # binary STL


def test_storage_keys(tmp_path):
    st = LocalStorage(tmp_path)
    jid = jobs.new_id()
    st.put(jid, "videos/1/frames/000.jpg", b"x")
    assert st.get(jid, "videos/1/frames/000.jpg") == b"x" and st.list(jid) == ["videos/1/frames/000.jpg"] and st.size(jid) == 1
    for bad in ("../x", "a/../../x", "/etc/passwd", "a//b", ".hidden", "a/" * 10 + "b", "x\\..\\y"):
        with pytest.raises(StorageKeyError):
            st.put(jid, bad, b"x")
    with pytest.raises(StorageKeyError):
        st.put("../../etc", "x", b"x")
    st.delete_job(jid)
    assert st.list(jid) == []
