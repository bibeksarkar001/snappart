"""Backend stage 3: questions with bounds and cards, validated answers, corrections with undo, rebuild with a report diff, review choices
and user marks, accept, and downloads (whitelisted names, correct headers, binary STL, GLB, preview size). Synthetic mock part only."""
import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, jobs, mock, results
from app.main import create_app
from app.settings import Settings
from app.storage import LocalStorage
from capture import carve


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("data3")
    s = Settings(data_dir=d, environment="test", mock=True, mock_build_s=0.0)
    return s, TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(d)


def built(env):
    """A job with the mock build's outputs, in needs_input."""
    s, cl, st = env
    jid = cl.post("/v1/jobs").json()["job_id"]
    with db.session(s.db_path) as c:
        for to in ("uploading", "checking", "ready_to_build", "queued", "building"):
            jobs.transition(c, jid, to)
        nxt, why = mock.run_build(c, st, s, jid, sleep=lambda t: None)
        jobs.transition(c, jid, nxt, why)
    return jid


def test_questions_answers_and_bounds(env):
    s, cl, st = env
    jid = cl.post("/v1/jobs").json()["job_id"]
    assert cl.get(f"/v1/jobs/{jid}/questions").json()["error"]["code"] == "not_ready"
    jid = built(env)
    r = cl.get(f"/v1/jobs/{jid}/questions").json()
    qs = {q["id"]: q for q in r["questions"]}
    assert set(qs) == {"groove_depth_mm", "bump_height_mm", "bump_width_mm", "bump_length_mm", "bump_top_shape"} and r["open_questions"] == 5
    d = qs["groove_depth_mm"]
    h = cl.get(f"/v1/jobs/{jid}/report").json()["dims_mm"]["height"]["measured"]                   # measured on the meshed synthetic part
    assert abs(h - 6.0) < 0.2 and d["max"] == round(h, 2)
    assert d["kind"] == "number" and d["min"] == 0.2 and d["applies_to"] == 2 and d["unit"] == "mm" and d["feature"] == "grooves"
    assert "2 openings measured 10.0 x 3.0 mm" in d["detail"] and d["impact"] is None
    assert qs["bump_top_shape"]["kind"] == "choice" and qs["bump_top_shape"]["choices"] == ["sharp", "slightly rounded", "quite rounded"]
    img = cl.get(d["card"]["image"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png" and img.headers["content-disposition"].startswith("inline")
    assert r["review_items"][0]["id"] == "M01" and cl.get(r["review_items"][0]["image"]).status_code == 200
    bad = [({"groove_depth_mm": 7.5}, "answer_out_of_range"), ({"groove_depth_mm": 0.1}, "answer_out_of_range"), ({"nope": 1}, "unknown_question"),
           ({"bump_top_shape": "pointy"}, "answer_out_of_range"), ({"groove_depth_mm": "abc"}, "answer_out_of_range")]
    for body, code in bad:
        e = cl.post(f"/v1/jobs/{jid}/answers", json={"answers": body})
        assert e.status_code == 422 and e.json()["error"]["code"] == code, body
    assert "deeper than the part is tall" in cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": 6.6}}).json()["error"]["message"]
    ok = cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": "2.5", "bump_top_shape": "skip"}}).json()
    assert ok["saved"] == ["bump_top_shape", "groove_depth_mm"] and ok["open_questions"] == 3
    q = {x["id"]: x for x in cl.get(f"/v1/jobs/{jid}/questions").json()["questions"]}
    assert q["groove_depth_mm"]["answer"] == 2.5 and q["groove_depth_mm"]["status"] == "user_supplied" and q["bump_top_shape"]["status"] == "estimated"


def test_rebuild_corrections_undo_review_and_accept(env):
    s, cl, st = env
    jid = built(env)
    r0 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    assert r0["state"] == "needs_input" and r0["open_questions"] == 5 and r0["watertight"] and r0["seconds"] < 10
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": 2.0}})
    r1 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    ch = {c["id"]: c for c in r1["changes"]}
    assert ch["grooves.depth"]["to"] == 2.0 and ch["grooves.depth"]["status_to"] == "user_supplied"
    assert r1["volume_mm3"] > r0["volume_mm3"]                                  # shallower slots: more material
    rep = cl.get(f"/v1/jobs/{jid}/report").json()
    assert rep["answers"] == {"groove_depth_mm": 2.0} and rep["mode"] == "mock" and rep["rebuild"]["volume_mm3"] == r1["volume_mm3"]
    # corrections: only the engine's ops, typed; each can be undone
    assert cl.post(f"/v1/jobs/{jid}/corrections", json={"correction": {"op": "rm -rf /"}}).json()["error"]["code"] == "invalid_correction"
    assert cl.post(f"/v1/jobs/{jid}/corrections", json={"correction": {"op": "set_feature_shape", "feature_id": "nope", "shape": "box"}}).status_code == 422
    cid = cl.post(f"/v1/jobs/{jid}/corrections", json={"correction": {"op": "set_feature_shape", "feature_id": "bump", "shape": "box"}}).json()["correction_id"]
    r2 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    assert r2["volume_mm3"] > r1["volume_mm3"]                                  # a box ridge holds more than the hill
    assert cl.delete(f"/v1/jobs/{jid}/corrections/{cid}").json()["undone"] == cid
    assert cl.delete(f"/v1/jobs/{jid}/corrections/{cid}").status_code == 404
    r3 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    assert r3["volume_mm3"] == r1["volume_mm3"]                                 # undo restores the model exactly
    assert [c["reverted"] for c in cl.get(f"/v1/jobs/{jid}/corrections").json()["corrections"]] == [True]
    # review: remove the flagged patch, change your mind, keep it
    rv = cl.post(f"/v1/jobs/{jid}/review/M01", json={"choice": "Remove"}).json()
    assert rv["correction_id"] and cl.post(f"/v1/jobs/{jid}/review/M01", json={"choice": "Explode"}).status_code == 422
    r4 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    assert r4["volume_mm3"] < r3["volume_mm3"] - 10                            # 6 x 12 mm patch cut 0.3 mm down (and the ridge base above it)
    assert cl.post(f"/v1/jobs/{jid}/review/M01", json={"choice": "Keep"}).json()["correction_id"] is None
    assert cl.post(f"/v1/jobs/{jid}/rebuild").json()["volume_mm3"] == r3["volume_mm3"]
    m = cl.post(f"/v1/jobs/{jid}/marks", json={"point": [-15.0, 10.0, 6.0], "radius_mm": 2.0, "note": "scratch"}).json()
    assert m["id"] == "U01" and m["choices"] == ["Keep", "Remove", "Make flat"]
    assert cl.post(f"/v1/jobs/{jid}/review/U01", json={"choice": "Make flat"}).json()["correction_id"]
    assert cl.post(f"/v1/jobs/{jid}/marks", json={"point": [0, 0], "radius_mm": 2}).status_code == 422
    # accept: not while questions are open; a later change reopens review
    assert cl.post(f"/v1/jobs/{jid}/accept").json()["error"]["code"] == "questions_open"
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {k: "skip" for k in ("bump_height_mm", "bump_width_mm", "bump_length_mm", "bump_top_shape")}})
    assert cl.post(f"/v1/jobs/{jid}/rebuild").json()["state"] == "reviewing"
    assert cl.post(f"/v1/jobs/{jid}/accept").json()["error"]["code"] == "footage_requested"            # a blocking coverage request is open
    assert cl.post(f"/v1/jobs/{jid}/more-footage", json={"request_id": "film_1", "action": "skip"}).json()["request"]["decision"] == "skipped"
    assert cl.post(f"/v1/jobs/{jid}/accept").json()["state"] == "done"
    cl.post(f"/v1/jobs/{jid}/answers", json={"answers": {"groove_depth_mm": 3.0}})
    assert cl.get(f"/v1/jobs/{jid}").json()["state"] == "reviewing"


def test_review_choice_ops():
    one = {"id": "R02", "kind": "one_video", "feature_id": "lock_bump"}
    assert results.review_correction(one, "Opposite side") == {"op": "move_feature_to_opposite_face", "feature_id": "bump", "source": "review", "review_item": "R02"}
    assert results.review_correction(one, "Current side") is None and results.review_correction(one, "Remove")["feature"] == {}
    side = {"id": "R03", "kind": "lump", "polygon_xy": [[0, 0], [1, 0], [1, 1]], "z_mm": [1, 2, 3], "normal": [1, 0, 0]}
    with pytest.raises(Exception) as e:
        results.review_correction(side, "Make flat")
    assert e.value.code == "choice_not_available"
    top = {**side, "normal": [0, 0, 1.0]}
    assert results.review_correction(top, "Make flat")["region"]["z_mm"] == [2, 6.0]


def test_downloads(env):
    s, cl, st = env
    jid = built(env)
    r = cl.get(f"/v1/jobs/{jid}/files/model.stl")
    assert r.status_code == 200 and r.headers["content-type"] == "model/stl" and r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store" and r.headers["content-disposition"] == f'attachment; filename="snappart-{jid[:8]}-model.stl"'
    assert len(r.content) == 84 + 50 * struct.unpack("<I", r.content[80:84])[0]
    for bad in ("../report.json", "model.npz", "work/result.json", "cards/../../x/card.png", "report.json/../model.npz", "review/M01.PNG"):
        assert cl.get(f"/v1/jobs/{jid}/files/{bad}").status_code == 404, bad
    assert cl.get(f"/v1/jobs/{jid}/files/report.json").headers["content-type"] == "application/json"
    svg = cl.get(f"/v1/jobs/{jid}/files/cards/groove_depth_mm/diagram.svg")
    assert svg.headers["content-type"].startswith("image/svg+xml") and "sandbox" in svg.headers["content-security-policy"]
    g = cl.get(f"/v1/jobs/{jid}/files/preview.glb")
    magic, ver, total = struct.unpack("<4sII", g.content[:12])
    jl = struct.unpack("<I", g.content[12:16])[0]
    gl = json.loads(g.content[20:20 + jl])
    assert g.headers["content-type"] == "model/gltf-binary" and magic == b"glTF" and ver == 2 and total == len(g.content)
    assert gl["accessors"][2]["count"] % 3 == 0 and gl["nodes"][0]["scale"] == [0.001] * 3
    assert cl.get(f"/v1/jobs/{jid}/files/preview.stl").status_code == 200


def test_preview_is_decimated(monkeypatch):
    g = carve.Grid([-12, -12, -4], [12, 12, 8], 0.2)
    X = g.points()
    field = np.clip(np.minimum.reduce([10 - np.abs(X[:, 0]), 8 - np.abs(X[:, 1]), X[:, 2], 6 - X[:, 2]]), -3, 3).astype(np.float32).reshape(g.shape)
    from capture import model as Mo
    v, f = Mo.mesh(field, g)
    monkeypatch.setattr(results, "PREVIEW_MAX_TRIANGLES", len(f) // 3)
    pv, pf = results.preview_mesh(field, g, v, f)
    assert len(pf) <= len(f) // 3 and carve.edge_census(pv, pf)["watertight"]
    assert np.allclose(pv.min(0), v.min(0), atol=0.6) and np.allclose(pv.max(0), v.max(0), atol=0.6)     # same part, coarser
