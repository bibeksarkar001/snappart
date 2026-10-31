"""GET /v1/jobs/{id}/inside: this job's frames with outlines, poses, timings and Meta call log; nothing for mock builds; gone with DELETE."""
import json

from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage
from test_app_stage2 import frame_jpeg


def test_inside_endpoint_and_delete(tmp_path):
    s = Settings(data_dir=tmp_path, environment="test", mock=True)
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(tmp_path)
    jid = cl.post("/v1/jobs").json()["job_id"]
    r = cl.get(f"/v1/jobs/{jid}/inside").json()
    assert r["available"] is False and r["frames"] == [] and "mock" in r["reason"]
    st.put(jid, "outputs/inside/1_0007.jpg", frame_jpeg(0, 30))                    # as realbuild.write_inside writes them
    st.put(jid, "outputs/inside/frames.json", json.dumps([{"video": 1, "index": 7, "pose": True, "outline_accepted": True,
                                                           "file": "inside/1_0007.jpg"}]).encode())
    st.put(jid, "outputs/inside/poses.json", json.dumps({"poses": [{"video": 1, "index": 7, "camera_centre_mm": [1, 2, 3]}]}).encode())
    st.put(jid, "outputs/report.json", json.dumps({"timings": {"per_stage_s": {"SAM": 49.1}}}).encode())
    with db.session(s.db_path) as c, db.tx(c):
        c.execute("INSERT INTO model_calls (job_id, model, endpoint, ok, status_code, latency_ms, ts) VALUES (?, 'sam-3.1', '/responses', 1, 200, 812.3, ?)",
                  (jid, db.now()))
    r = cl.get(f"/v1/jobs/{jid}/inside").json()
    assert r["available"] and r["frames"][0]["url"] == f"/v1/jobs/{jid}/files/inside/1_0007.jpg" and r["timings"]["per_stage_s"]["SAM"] == 49.1
    assert r["meta_calls"] == [{"timestamp": r["meta_calls"][0]["timestamp"], "model": "sam-3.1", "endpoint": "/responses", "status": 200,
                                "ok": True, "latency_ms": 812.3}]
    img = cl.get(r["frames"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert cl.get(f"/v1/jobs/{jid}/files/inside/../../x.jpg").status_code == 404
    cl.delete(f"/v1/jobs/{jid}")
    assert not st.path(jid).exists() and cl.get(f"/v1/jobs/{jid}/inside").status_code == 404
