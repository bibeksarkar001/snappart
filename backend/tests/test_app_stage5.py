"""Backend stage 5: quality gates on synthetic fields, the film-more-footage loop (blocking request, film or skip, new footage must face the
area), draft labelling of downloads, and job answers that come from the job (no part-B constants). Synthetic only."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, jobs, mock
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage
from capture import carve, gates as G, model as Mo
from test_app_stage2 import frame_jpeg


def slab(extra=None):
    g = carve.Grid([-12, -12, -4], [12, 12, 10], 0.25)
    X = g.points()
    f = np.minimum.reduce([10 - np.abs(X[:, 0]), 8 - np.abs(X[:, 1]), X[:, 2], 6 - X[:, 2]])
    if extra is not None:
        f = np.maximum(f, extra(X))
    return np.clip(f, -3, 3).astype(np.float32).reshape(g.shape), g


def test_gates():
    f, g = slab()
    v, fc = Mo.mesh(f, g)
    q = G.check(v, fc, f, 0.95, g)
    assert q["status"] == "ready" and not q["reasons"] and q["roughness"][0]["rms_mm"] < 0.05
    assert G.check(v, fc, f, None, g)["not_run"] == ["held_out_iou"]
    assert G.check(v, fc, f, 0.85, g)["status"] == "draft"                      # held-out IoU under the threshold
    # a one-voxel-thick fin standing on the top face
    occ = (f > 0)
    fin = np.zeros_like(occ)
    ix, iz = int((5 - g.lo[0]) / g.vs), int((6 - g.lo[2]) / g.vs)
    fin[ix, 20:80, iz:iz + 12] = True
    ff = np.where(fin & ~occ, 0.1, f).astype(np.float32)
    q2 = G.check(v, fc, ff, 0.95, g)
    assert q2["status"] == "draft" and any(x["gate"] == "no_zero_thickness" and x["ok"] is False for x in q2["gates"])
    # two bodies
    f2, g2 = slab(lambda X: np.minimum.reduce([1.5 - np.abs(X[:, 0] - 0), 1.5 - np.abs(X[:, 1]), X[:, 2] - 8, 9.5 - X[:, 2]]))
    v2, fc2 = carve.to_mesh(f2, g2, keep_largest=False)
    assert G.check(v2, fc2, None, None)["gates"][1]["value"]["bodies"] == 2
    eb = G.error_bars([{"length": 20.0, "width": 16.0, "height": 6.0}, {"length": 20.2, "width": 16.0, "height": 6.0}], 0.2, 0.15)
    assert eb["plus_minus_mm"]["length"] > eb["plus_minus_mm"]["width"] > 0 and eb["status"] == "estimated"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("data5")
    s = Settings(data_dir=d, environment="test", mock=True, mock_build_s=0.0, min_usable_frames=3)
    return s, TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(d)


def built(env):
    s, cl, st = env
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        nxt, why = mock.run_build(c, st, s, jid, sleep=lambda t: None)
        jobs.transition(c, jid, nxt, why)
    return jid


def put_frames(cl, jid, video, start, azs, el):
    for i, az in enumerate(azs):
        r = cl.put(f"/v1/jobs/{jid}/videos/{video}/frames/{start + i}", content=frame_jpeg(az, el), headers={"content-type": "image/jpeg"})
        assert r.status_code == 200, r.text


def test_more_footage_loop(env):
    s, cl, st = env
    jid = built(env)
    q = cl.get(f"/v1/jobs/{jid}/questions").json()
    req = q["coverage_requests"][0]
    assert req["id"] == "film_1" and req["blocking"] and req["decision"] is None and "Film about" in req["message"]
    r = cl.post(f"/v1/jobs/{jid}/more-footage", json={"request_id": "film_1", "action": "film"}).json()
    assert r["state"] == "uploading" and r["request"]["decision"] == "filming"
    put_frames(cl, jid, 1, 0, [0, 20, 340, 10], 25)                           # new frames from the right side: not the requested left side
    cl.post(f"/v1/jobs/{jid}/capture-check")
    e = cl.post(f"/v1/jobs/{jid}/start", json={"build_anyway": True}).json()
    assert e["error"]["code"] == "footage_missing_area" and "barely saw the left side" in e["error"]["message"]
    put_frames(cl, jid, 1, 4, [170, 180, 190, 200], 25)                       # now the left side
    cl.post(f"/v1/jobs/{jid}/capture-check")
    j = cl.post(f"/v1/jobs/{jid}/start", json={"build_anyway": True}).json()
    assert j["state"] == "queued" and not any("could not be checked" in w for w in j["warnings"])
    assert st.exists(jid, "more_footage_used.json") and not st.exists(jid, "more_footage.json")
    bad = cl.post(f"/v1/jobs/{jid}/more-footage", json={"request_id": "film_1", "action": "explode"})
    assert bad.status_code == 422


def test_skip_and_draft_labelling(env):
    s, cl, st = env
    jid = built(env)
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {q["id"]: "skip" for q in cl.get(f"/v1/jobs/{jid}/questions").json()["questions"]}})
    r = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    assert r["quality"]["status"] == "ready" and r["state"] == "reviewing"
    assert cl.post(f"/v1/jobs/{jid}/accept").json()["error"]["code"] == "footage_requested"
    cl.post(f"/v1/jobs/{jid}/more-footage", json={"request_id": "film_1", "action": "skip"})
    rep = json.loads(st.get(jid, "outputs/report.json"))
    rep["quality"] = {"status": "draft", "reasons": ["Parts of the model are paper-thin."], "gates": []}
    st.put(jid, "outputs/report.json", json.dumps(rep).encode())
    d = cl.get(f"/v1/jobs/{jid}/files/model.stl")
    assert d.headers["x-snappart-draft"] == "true" and d.headers["content-disposition"].endswith('-model-DRAFT.stl"')
    e = cl.post(f"/v1/jobs/{jid}/accept").json()
    assert e["error"]["code"] == "draft_model" and "paper-thin" in e["error"]["message"]
    rep["quality"]["status"] = "ready"
    st.put(jid, "outputs/report.json", json.dumps(rep).encode())
    assert cl.get(f"/v1/jobs/{jid}/files/model.stl").headers["x-snappart-draft"] == "false"
    assert cl.post(f"/v1/jobs/{jid}/accept").json()["state"] == "done"


def test_mock_report_has_details_and_quality(env):
    s, cl, st = env
    jid = built(env)
    rep = cl.get(f"/v1/jobs/{jid}/report").json()
    kinds = {d["kind"] for d in rep["details"]}
    assert {"step"} <= kinds and all(d["status"] in ("measured", "question") for d in rep["details"])   # slots / ridge give steps on the synthetic part
    assert rep["quality"]["status"] == "ready" and rep["quality"]["not_run"] == ["held_out_iou"]
    q = {x["id"]: x for x in cl.get(f"/v1/jobs/{jid}/questions").json()["questions"]}
    assert q["groove_depth_mm"]["default_source"] == "estimated" and q["bump_height_mm"]["default_source"] == "measured"
