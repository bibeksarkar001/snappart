"""Replay mode: the recorded part B run (examples/recorded/part_b/) served through the same API. Every response says "recorded run"
with its date, no model is ever shown running and no Meta call is made, edits are refused plainly, and replay is the default without a key."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import db, jobs, replay
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage

BUNDLE = Path(__file__).resolve().parents[2] / "examples" / "recorded" / "part_b"


def test_replay_is_default_without_a_key(tmp_path):
    assert Settings(data_dir=tmp_path, model_api_key=None).mode == "replay"
    assert Settings(data_dir=tmp_path, model_api_key=None, mock=True).mode == "mock"
    assert Settings(data_dir=tmp_path, model_api_key="k" * 20).mode == "real"
    assert Settings(data_dir=tmp_path, model_api_key=None, replay=False).mode == "real"
    assert Settings(data_dir=tmp_path, model_api_key="  ").mode == "replay"                       # a blank key line is no key
    assert Settings(data_dir=tmp_path, turnstile_secret="").turnstile_secret is None                # blank example lines switch nothing on


def test_replay_serves_the_recorded_run(tmp_path, capsys):
    s = Settings(data_dir=tmp_path, environment="test", model_api_key=None, rate_create_per_hour=1000)
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(tmp_path)
    rec = json.loads((BUNDLE / "timings.json").read_text())["recorded_at"][:10]
    assert f"recorded run, {rec}" in capsys.readouterr().out                                       # the startup message says so
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        nxt, why = replay.run_build(c, st, s, jid)
        jobs.transition(c, jid, nxt, why)
    label = f"recorded run, {rec}"
    for path in ("/v1/meta/models", f"/v1/jobs/{jid}", f"/v1/jobs/{jid}/report", f"/v1/jobs/{jid}/questions", f"/v1/jobs/{jid}/summary",
                 f"/v1/jobs/{jid}/files/model.stl", "/nope"):
        r = cl.get(path)
        assert r.headers["x-snappart-mode"] == "replay" and r.headers["x-snappart-recorded"] == label, path
    job = cl.get(f"/v1/jobs/{jid}").json()
    assert job["mode"] == "replay" and job["recorded"]["label"] == label and job["recorded"]["live_model_calls"] is False
    rep = cl.get(f"/v1/jobs/{jid}/report").json()
    assert rep["mode"] == "replay" and rep["recorded"]["label"] == label and rep["warnings"][0].startswith(f"Recorded run ({label})")
    assert rep["held_out_iou"] == json.loads((BUNDLE / "report.json").read_text())["held_out_iou"]
    q = cl.get(f"/v1/jobs/{jid}/questions").json()
    if replay.answers(s):                                                                           # the owner answered and confirmed all
        assert q["open_questions"] == 0 and all(x["answered"] for x in q["questions"]), [x["id"] for x in q["questions"] if not x["answered"]]
        plate = next((x for x in q["questions"] if x["id"] == "plate_thickness_mm"), None)
        assert plate is None or plate["confirmed"] and plate["status"] == "user_supplied"
    assert q["recorded"]["label"] == label and all(x["friendly"] is None or x["friendly"]["recorded"] == label for x in q["questions"])
    assert cl.get(f"/v1/jobs/{jid}/summary").json()["recorded"]["label"] == label
    m = cl.get("/v1/meta/models").json()
    assert not any(x["running"] for x in m["models"]) and all(x["reason"].startswith("replay mode") for x in m["models"])
    assert m["recorded"]["recorded_meta_calls"]["sam-3.1"]["ok"] > 0                                 # the recorded counts, labelled as such
    r = cl.post(f"/v1/jobs/{jid}/more-footage", json={"request_id": "film_1", "action": "film"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "replay_read_only"
    if replay.editable(s):                                                                          # the bundle carries the editable model
        qid = q["questions"][0]["id"]
        assert cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {qid: "skip"}}).status_code == 200
        rb = cl.post(f"/v1/jobs/{jid}/rebuild").json()
        assert rb["label"] == replay.OFFLINE and cl.get(f"/v1/jobs/{jid}/report").json()["rebuild"]["label"] == replay.OFFLINE
        p = cl.post(f"/v1/jobs/{jid}/proposals", json={"text": "make this flat"}).json()
        assert p["label"] == replay.OFFLINE_PROPOSAL and p["source"].startswith("fallback")
    with db.session(s.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0                     # no Meta call was made


def test_replay_without_the_model_refuses_edits(tmp_path):
    import shutil
    b = tmp_path / "bundle"
    shutil.copytree(BUNDLE, b, ignore=shutil.ignore_patterns("model.npz", "model.json", "*.stl", "*.glb"))
    s = Settings(data_dir=tmp_path / "d", environment="test", model_api_key=None, replay_dir=b)
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(tmp_path / "d")
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        jobs.transition(c, jid, *replay.run_build(c, st, s, jid))
    for path, body in (("answers", {"answers": {"groove_depth_mm": 3.0}}), ("rebuild", None), ("proposals", {"text": "make it flat"})):
        r = cl.post(f"/v1/jobs/{jid}/{path}", json=body)
        assert r.status_code == 409 and r.json()["error"]["code"] == "replay_read_only", path


def built_replay(tmp_path, **kw):
    import shutil
    b = tmp_path / "bundle"
    shutil.copytree(BUNDLE, b, ignore=shutil.ignore_patterns("*.stl"))                              # the bundle carries no STLs
    s = Settings(data_dir=tmp_path / "d", environment="test", model_api_key=None, replay_dir=b, **kw)
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(tmp_path / "d")
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        jobs.transition(c, jid, *replay.run_build(c, st, s, jid))
    return s, cl, st, jid


def test_replay_makes_stls_on_request_and_keeps_the_before_model(tmp_path):
    s, cl, st, jid = built_replay(tmp_path)
    before = cl.get(f"/v1/jobs/{jid}/files/preview_before.glb")
    src = BUNDLE / ("preview_before.glb" if (BUNDLE / "preview_before.glb").is_file() else "preview.glb")       # the raw build
    assert before.status_code == 200 and before.content == src.read_bytes()
    if not replay.editable(s):
        return
    assert not st.exists(jid, "outputs/model.stl")
    r = cl.get(f"/v1/jobs/{jid}/files/model.stl")
    assert r.status_code == 200 and r.content[:5] != b"solid" and r.headers["x-snappart-label"] == replay.OFFLINE
    assert cl.get(f"/v1/jobs/{jid}/report").json()["rebuild"]["label"] == replay.OFFLINE
    assert cl.get(f"/v1/jobs/{jid}/files/preview.stl").status_code == 200
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": 3.7}})
    cl.post(f"/v1/jobs/{jid}/rebuild")
    assert cl.get(f"/v1/jobs/{jid}/files/preview_before.glb").content == before.content                    # rebuilds keep it


def test_mock_build_keeps_the_before_model(tmp_path):
    s = Settings(data_dir=tmp_path, environment="test", mock=True, mock_build_s=0.0)
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(tmp_path)
    from app import mock
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        jobs.transition(c, jid, *mock.run_build(c, st, s, jid, sleep=lambda t: None))
    before = cl.get(f"/v1/jobs/{jid}/files/preview_before.glb").content
    qid = cl.get(f"/v1/jobs/{jid}/questions").json()["questions"][0]["id"]
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {qid: "skip"}})
    assert cl.post(f"/v1/jobs/{jid}/rebuild").status_code == 200
    assert before and cl.get(f"/v1/jobs/{jid}/files/preview_before.glb").content == before


def test_conflict_can_be_resolved_with_the_video_value(tmp_path):
    s, cl, st, jid = built_replay(tmp_path)
    plate = next((x for x in cl.get(f"/v1/jobs/{jid}/questions").json()["questions"] if x["id"] == "plate_thickness_mm"), None)
    if plate is None or not replay.editable(s):
        return
    from app import results
    sep = results._feature_value(results.base_report(st, jid), "plate.faces_separation")["value"]
    assert cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"plate_thickness_mm": "use video"}}).status_code == 200
    q = next(x for x in cl.get(f"/v1/jobs/{jid}/questions").json()["questions"] if x["id"] == "plate_thickness_mm")
    assert q["answer"] == round(sep, 2) and q["answered"] and q["confirmed"]
    assert cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": "keep my answer"}}).status_code == 422


def test_caliper_table_and_report_errors(tmp_path):
    cal = json.loads((BUNDLE / "calipers.json").read_text())["calipers_mm"] if (BUNDLE / "calipers.json").is_file() else None
    s = Settings(data_dir=tmp_path / "m", environment="test", mock=True)
    r = TestClient(create_app(s)).get("/v1/recorded/calipers")
    if cal is None:
        assert r.status_code == 404
        return
    rep = json.loads((BUNDLE / "report.json").read_text())
    rows = {x["dimension"]: x for x in r.json()["rows"]}
    for k, c in cal.items():
        assert rows[k]["caliper_mm"] == c and rows[k]["error_mm"] == round(rep["dims_mm"][k]["measured"] - c, 2)
    s2, cl, st, jid = built_replay(tmp_path)
    dims = cl.get(f"/v1/jobs/{jid}/report").json()["dims_mm"]
    assert all(dims[k]["caliper"] == c and dims[k]["caliper_status"] == "user_supplied" for k, c in cal.items())
