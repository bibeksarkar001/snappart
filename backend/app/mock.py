"""Mock mode (SNAPPART_MOCK=1): the same API and job states, but the build step replays canned progress over settings.mock_build_s seconds
and delivers a canned result instead of running the capture pipeline or calling any Meta model.

The canned result is a SYNTHETIC part made here (no private data): a 40 x 30 x 6 mm plate with two slots and a hill-shaped ridge, turned
into an editable model, STL, report.json and question cards by the real capture code (model.py, report.py, question_cards.py), so the
answer / correction / rebuild endpoints work on it too. Every report it produces says "mode": "mock" and "synthetic": true."""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import db, events

STAGES = [("marker detection", 0.02), ("mat self-calibration", 0.06), ("poses + undistorted crops", 0.02), ("SAM", 0.30),
          ("registration", 0.17), ("joint carve", 0.21), ("features", 0.18), ("apply answers + mesh + STL", 0.04)]   # real run's time shares


def synthetic_model():
    """Carve-style field (inside positive, mm) and features of the synthetic part, in the same form the pipeline produces."""
    from capture import carve
    g = carve.Grid([-25, -20, -4], [25, 20, 10], 0.25)
    X = g.points()
    field = np.clip(np.minimum.reduce([20 - np.abs(X[:, 0]), 15 - np.abs(X[:, 1]), X[:, 2], 6 - X[:, 2]]), -3, 3).astype(np.float32).reshape(g.shape)
    ax = [[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]                         # long along y, short along x, opening normal +z (top face)
    grooves = [{"centre": [x, 0.0, 6.0], "axes": ax, "long_mm": 10.0, "short_mm": 3.0, "views": 0} for x in (-12.0, 12.0)]
    ridge = {"kind": "ridge", "shape": "hill", "base_mid": [0.0, 0.0, 6.0], "axis": [0, 1.0, 0], "across": [1.0, 0, 0], "normal": [0, 0, 1.0],
             "detected_length_mm": 16.0, "ends": [[0, -8.0, 6.0], [0, 8.0, 6.0]], "extent_mm": 2.0, "extent_views": 0, "agreeing_views": 0,
             "face": "top face (synthetic mock part)"}
    return field, g, {"grooves": grooves, "bump": ridge}


def make_bundle(out: Path) -> None:
    """Write model.npz/.json, model.stl, report.json and cards/<id>/ for the synthetic part into out."""
    import cv2
    from capture import model as Mo, question_cards as Q, render as Rn, report as Rp
    field, g, feats = synthetic_model()
    ans = Mo.default_answers()
    v0, f0 = Mo.mesh(field, g)
    v, fc = Mo.rebuild({"field": field, "grid": g, "features": feats}, ans)
    from . import results
    a_full = Mo.resolve_answers(ans)
    f_full = Mo.apply(field, g, feats, a_full)
    pv, pf = results.preview_mesh(f_full, g, v, fc)
    (out / "model.stl").write_bytes(results.write_stl_bytes(v, fc))
    (out / "preview.stl").write_bytes(results.write_stl_bytes(pv, pf))
    (out / "preview.glb").write_bytes(results.glb_bytes(pv, pf))
    Mo.save(out / "model", field, g, feats)
    rep = Rp.build((v0, f0), feats, ans, user_given=())
    rep.update(mode="mock", synthetic=True, meta_models_ran=[],
               timings={"per_stage_s": {name: None for name, _ in STAGES}, "note": "mock: no stage ran"})
    rep["warnings"].insert(0, "Mock mode: a canned result of a synthetic test part. Nothing you uploaded was processed and no Meta model ran.")
    # card "frames": renders of the synthetic model from an orbit (there are no photos in mock mode), labelled MOCK
    target = (v.max(0) + v.min(0)) / 2
    views, imgs = [], []
    for az in range(0, 360, 30):
        for el in (35, 60):
            vw = Rn.orbit(target, az, el, 220.0, f=1400.0, size=(960, 720))
            im = Rn.render(v, fc, vw)
            cv2.putText(im, "MOCK", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60, 60, 255), 3, cv2.LINE_AA)
            views.append(("mock", vw))
            imgs.append(im)
    # one canned review item so the review screen can be wired: a patch of the top face (the synthetic part has no real defects)
    from capture import review as Rv
    poly = [[3.0, -6.0], [9.0, -6.0], [9.0, 6.0], [3.0, 6.0]]
    ctr = (v[fc].mean(1))
    tri = np.nonzero((ctr[:, 0] > 3) & (ctr[:, 0] < 9) & (np.abs(ctr[:, 1]) < 6) & (ctr[:, 2] > 5.5))[0]
    q_, ch = Rv.QUESTIONS["lump"]
    rep["review_items"] = [{"id": "M01", "kind": "lump", "reason": "MOCK: a canned example of a flagged area, not a real finding",
                            "question": q_, "choices": ch, "centre": [6.0, 0.0, 6.0], "normal": [0, 0, 1.0], "polygon_xy": poly,
                            "z_mm": [5.9, 6.0, 6.1], "area_mm2": 72.0, "triangles": int(len(tri))}]
    img, _ = Q.region_card(tri, v, fc, [0, 0, 1.0], views, imgs, (v, fc), "M01: MOCK review item")
    (out / "review").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "review" / "M01.png"), img)
    for q in rep["questions"]:
        img, rec = Q.frame_card(Q.anchors(q["id"], feats, ans), views, imgs, (v, fc))
        cv2.putText(img, "MOCK", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 60, 255), 3, cv2.LINE_AA)   # the crop drops the frame's label
        Q.write_card(out / "cards" / q["id"], q, img, {**(rec or {}), "mock": True}, Q.diagram(q["id"]))
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))


def bundle(data_dir: Path) -> Path:
    """The canned bundle, generated once per data dir (atomic: built in a temp dir, then renamed into place)."""
    final = Path(data_dir) / "mock_bundle_v2"                           # bump the version when the bundle's contents change
    if (final / "report.json").exists():
        return final
    tmp = Path(tempfile.mkdtemp(dir=data_dir, prefix=".mock-"))
    try:
        make_bundle(tmp)
        try:
            os.replace(tmp, final)
        except OSError:                                                  # another process finished first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def run_build(conn, storage, settings, job_id: str, sleep=time.sleep) -> tuple[str, str]:
    """Mock build step for a job a worker has claimed (state building): progress events over mock_build_s, then the canned outputs.
    Stage checkpoints in work/checkpoint.json: a resumed build skips the stages it already finished. Returns (next state, reason)."""
    src = bundle(settings.data_dir)
    ck = "work/checkpoint.json"
    done = set(json.loads(storage.get(job_id, ck))["done"]) if storage.exists(job_id, ck) else set()
    n, frac = len(STAGES), 0.0
    for i, (name, share) in enumerate(STAGES):
        frac += share
        if name in done:
            continue
        events.emit(conn, job_id, "stage_started", {"stage": name, "index": i + 1, "of": n})
        dt = settings.mock_build_s * share
        sleep(dt)
        with db.tx(conn):
            events.append(conn, job_id, "stage_finished", {"stage": name, "index": i + 1, "of": n, "seconds": round(dt, 2), "mock": True})
            events.append(conn, job_id, "progress", {"fraction": round(frac, 3)})
        done.add(name)
        storage.put(job_id, ck, json.dumps({"done": sorted(done)}).encode())
    keys = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            key = "outputs/" + p.relative_to(src).as_posix()
            storage.put(job_id, key, p.read_bytes())
            keys.append(key)
    rep = json.loads((src / "report.json").read_text())
    events.emit(conn, job_id, "build_finished", {"outputs": keys, "questions": len(rep["questions"]), "mock": True})
    return ("needs_input", "questions the camera cannot answer") if rep["questions"] else ("reviewing", "ready for review")
