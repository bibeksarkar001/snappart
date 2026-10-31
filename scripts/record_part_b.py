"""Record a real run of part B through the API and save the derived results to examples/recorded/part_b/ (tracked).

Input: the owner's private videos assets/private/lock_set/Video/A_orbit_0{1,2,3}.mov (read only; nothing from them is copied into a
tracked path as is). Each video is cut into 60 equal windows and the sharpest frame of each window (Laplacian variance at 1280 px, every
3rd frame checked) is uploaded at 1920 px through the real API (in-process TestClient), as a capture client that drops blurry frames would.
The capture check, build (worker -> build subprocess -> SAM 3.1 through the Meta client) and Muse Spark steps run for real with the key
from .env (loaded by the app's settings, never printed). If assets/private/voice/answer.wav exists it goes through POST .../voice, the
proposal is applied and the model rebuilt. The data directory is a temporary folder outside the repo.

Saved: report.json, preview.glb, model.npz + model.json (if the npz is 40 MB or less; replay makes the STLs from it), cards/<id>/card.json +
diagram.svg, inside/frames.json + poses.json, questions, summary, capture check, voice.json (transcript, proposal, rebuild diff; never
audio), timings.json and meta_calls.jsonl. Images (cards, review, inside) are only staged under --stage, as part-only crops, for a check
by eye; the checked ones are copied into examples/recorded/part_b/ afterwards. Run from the repo root: `python scripts/record_part_b.py --stage DIR [--long-side 0] [--voice] [--out DIR]`."""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
VIDEOS = ROOT / "assets" / "private" / "lock_set" / "Video"
VOICE = ROOT / "assets" / "private" / "voice" / "answer.wav"
OUT = ROOT / "examples" / "recorded" / "part_b"
PER_VIDEO, MAX_MODEL_MB, MAX_FRAME_BYTES = 60, 40, 3 * 1024 * 1024


def sharpness(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, 1280 / max(g.shape))
    return float(cv2.Laplacian(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.CV_64F).var())


def frames(path: Path, long_side: int):
    """Sharpest frame (every 3rd checked) in each of PER_VIDEO equal windows, as 1920 px JPEGs."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    edges = np.linspace(0, n, PER_VIDEO + 1).astype(int)
    best = {}                                                            # window -> (sharpness, index, frame)
    for i in range(n):
        if i % 3:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        w = int(np.searchsorted(edges, i, side="right") - 1)
        sh = sharpness(fr)
        if w not in best or sh > best[w][0]:
            best[w] = (sh, i, fr)
    out, picked = [], []
    for w in sorted(best):
        sh, i, fr = best[w]
        s = long_side / max(fr.shape[:2])
        fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else fr
        for q in (92, 88, 84, 80, 75):                                   # stay under the per-frame upload cap
            b = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
            if len(b) <= MAX_FRAME_BYTES:
                break
        out.append(b)
        picked.append({"window": w, "frame": i, "sharpness": round(sh, 1), "jpeg_quality": q, "size": [fr.shape[1], fr.shape[0]]})
    return out, n, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True, help="folder outside the repo for the part-only images to check before copying")
    ap.add_argument("--out", type=Path, default=OUT, help="bundle folder (default examples/recorded/part_b)")
    ap.add_argument("--long-side", type=int, default=1920, help="uploaded frame size; 0 = the video's full resolution")
    ap.add_argument("--voice", action="store_true", help="send assets/private/voice/answer.wav through the voice endpoint")
    a = ap.parse_args()
    out_dir = a.out
    data = Path(tempfile.mkdtemp(prefix="snappart-partb-"))
    os.environ.update({"SNAPPART_DATA_DIR": str(data), "SNAPPART_MOCK": "0", "SNAPPART_REPLAY": "0", "SNAPPART_ENVIRONMENT": "development",
                       "SNAPPART_BUILD_TIMEOUT_S": "2400", "SNAPPART_MIN_FREE_BYTES": "0", "SNAPPART_PART_ONLY_PICTURES": "1",
                       "SNAPPART_MAX_FRAME_LONG_EDGE_PX": "4096", "SNAPPART_MAX_IMAGE_PIXELS": str(4096 * 2304)})   # full-resolution frames
    from fastapi.testclient import TestClient
    from app import db
    from app.main import create_app
    from app.settings import Settings
    from app.storage import LocalStorage
    from app.worker import Worker
    s = Settings()
    if s.model_api_key is None:
        sys.exit("No MODEL_API_KEY configured: a recorded real run needs it (it is read from .env by the app and never printed).")
    cl, st = TestClient(create_app(s), raise_server_exceptions=False), LocalStorage(data)
    t = {}
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    jid = cl.post("/v1/jobs", json={"part_name": "brown plastic buckle"}).json()["job_id"]
    print("job", jid, "data", data, flush=True)
    src = {}
    tx = time.perf_counter()
    for v in (1, 2, 3):
        fr, n, picked = frames(VIDEOS / f"A_orbit_0{v}.mov", a.long_side or 10 ** 6)
        src[v] = {"video_frames": n, "uploaded": len(fr), "selection": "sharpest of every 3rd frame in each of 60 equal windows",
                  "median_sharpness": round(float(np.median([p["sharpness"] for p in picked])), 1), "frame_size_px": picked[0]["size"],
                  "jpeg_quality": sorted({p["jpeg_quality"] for p in picked}), "max_frame_bytes": max(len(b) for b in fr)}
        for i, b in enumerate(fr):
            r = cl.put(f"/v1/jobs/{jid}/videos/{v}/frames/{i}", content=b, headers={"content-type": "image/jpeg"})
            assert r.status_code == 200, r.text
    t["select_and_upload_s"] = time.perf_counter() - tx
    tx = time.perf_counter()
    cc = cl.post(f"/v1/jobs/{jid}/capture-check").json()
    t["capture_check_s"] = time.perf_counter() - tx
    print("capture check", cc.get("verdict"), "second opinion", (cc.get("second_opinion") or {}).get("status"), flush=True)
    st_ = cl.post(f"/v1/jobs/{jid}/start", json={"build_anyway": not cc.get("usable", False)}).json()
    assert st_.get("state") == "queued", st_
    tx = time.perf_counter()
    Worker(s, "record-part-b").step()
    t["build_s"] = time.perf_counter() - tx
    job = cl.get(f"/v1/jobs/{jid}").json()
    print("build", job["state"], job.get("error"), f"{t['build_s']:.0f} s", flush=True)
    if job["state"] not in ("needs_input", "reviewing"):
        log = data / "jobs" / jid / "work" / "build.log"
        print(log.read_text()[-3000:] if log.exists() else "no build log")
        sys.exit(1)
    tx = time.perf_counter()
    q = cl.get(f"/v1/jobs/{jid}/questions").json()
    t["questions_with_spark_wording_s"] = time.perf_counter() - tx
    tx = time.perf_counter()
    summ = cl.get(f"/v1/jobs/{jid}/summary").json()
    t["summary_s"] = time.perf_counter() - tx
    raw = cl.get(f"/v1/jobs/{jid}/report").json()                         # the build's own report (raw state, no answers)
    raw_preview = cl.get(f"/v1/jobs/{jid}/files/preview_before.glb").content
    # the owner's caliper answers (tests/fixtures/part_b.json; the top radius there is an estimate, so it is not sent) and the confirmed
    # ridge shape: the "after answers" state, labelled user_supplied by the report
    fx = json.loads((ROOT / "backend" / "tests" / "fixtures" / "part_b.json").read_text())["answers"]
    owner = {k: v for k, v in fx.items() if k != "bump_top_radius_mm"} | {"bump_shape": "hill"}
    asked = {qq["id"] for qq in raw["questions"]}
    recorded_answers = {k: v for k, v in owner.items() if k in asked}
    tx = time.perf_counter()
    ar = cl.post(f"/v1/jobs/{jid}/answers", json={"answers": recorded_answers})
    assert ar.status_code == 200, ar.text
    rb0 = cl.post(f"/v1/jobs/{jid}/rebuild").json()
    t["owner_answers_rebuild_s"] = time.perf_counter() - tx
    rep = cl.get(f"/v1/jobs/{jid}/report").json()                         # after the owner's answers
    print("owner answers", recorded_answers, "not asked:", sorted(set(owner) - asked), "rebuild", rb0.get("seconds"), flush=True)
    voice = None
    if a.voice and VOICE.is_file():
        tx = time.perf_counter()
        vr = cl.post(f"/v1/jobs/{jid}/voice", files={"audio": ("answer.wav", VOICE.read_bytes(), "audio/wav")}).json()
        voice = {"transcript": vr.get("transcript"), "audio_seconds": vr.get("audio_seconds"), "proposal": vr.get("proposal"), "audio_kept": False}
        corr = (vr.get("proposal") or {}).get("correction")
        if corr:
            voice["applied"] = cl.post(f"/v1/jobs/{jid}/corrections", json={"correction": corr}).json()
            rb = cl.post(f"/v1/jobs/{jid}/rebuild").json()
            after = cl.get(f"/v1/jobs/{jid}/report").json()
            rep = after
            voice["rebuild"] = {"seconds": (after.get("rebuild") or {}).get("seconds"), "response": rb,
                                "dims_before_mm": {k: rep["dims_mm"][k]["measured"] for k in rep["dims_mm"]},
                                "dims_after_mm": {k: after["dims_mm"][k]["measured"] for k in after["dims_mm"]},
                                "quality_before": rep["quality"]["status"], "quality_after": after["quality"]["status"]}
        t["voice_s"] = time.perf_counter() - tx
    t["total_s"] = time.perf_counter() - t0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "cards").mkdir(parents=True)
    (out_dir / "inside").mkdir()
    for name in ("preview.glb",):                                        # model.stl / preview.stl are made from model.npz in replay
        r = cl.get(f"/v1/jobs/{jid}/files/{name}")
        assert r.status_code == 200, (name, r.status_code)
        (out_dir / name).write_bytes(r.content)
    (out_dir / "preview_before.glb").write_bytes(raw_preview)             # the raw build, before any answer
    (out_dir / "answers.json").write_text(json.dumps({"answers": recorded_answers, "status": "user_supplied",
                                                      "source": "owner's calipers (backend/tests/fixtures/part_b.json) and confirmed ridge shape"}, indent=1))
    (out_dir / "report_before.json").write_text(json.dumps(raw, indent=1))
    npz = st.path(jid, "outputs/model.npz")
    model_mb = round(npz.stat().st_size / 2 ** 20, 1)
    if model_mb <= MAX_MODEL_MB:
        shutil.copy(npz, out_dir / "model.npz")
        shutil.copy(st.path(jid, "outputs/model.json"), out_dir / "model.json")
    for qq in rep["questions"]:
        d = out_dir / "cards" / qq["id"]
        d.mkdir()
        for name in ("card.json", "diagram.svg"):
            r = cl.get(f"/v1/jobs/{jid}/files/cards/{qq['id']}/{name}")
            if r.status_code == 200:
                (d / name).write_bytes(r.content)
    for name in ("frames.json", "poses.json"):
        shutil.copy(st.path(jid, f"outputs/inside/{name}"), out_dir / "inside" / name)
    # part-only images: staged outside the repo for a check by eye (owner option A); nothing image-like is written to examples/ here
    stage = a.stage
    if stage.exists():
        shutil.rmtree(stage)
    for part in ("cards", "review", "inside"):                          # drawn part-only by the build (SNAPPART_PART_ONLY_PICTURES)
        (stage / part).mkdir(parents=True)
        for k in st.list(jid, f"outputs/{part}/"):
            if k.endswith((".png", ".jpg")):
                shutil.copy(st.path(jid, k), stage / part / (k.split("/")[-2] + ".png" if part == "cards" else Path(k).name))
    recorded = {"recorded_at": started.isoformat(timespec="seconds"), "kind": "recorded run", "part": "part B (briefcase lock plate)",
                "input": {"videos": "owner's private videos, not in this repository", "frames": src, "long_side_px": a.long_side or "full",
                          "part_name": "brown plastic buckle"}}
    (out_dir / "report.json").write_text(json.dumps(rep | {"recorded": recorded}, indent=1))
    (out_dir / "questions.json").write_text(json.dumps(q, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    (out_dir / "capture_check.json").write_text(json.dumps(cc, indent=1))
    if voice:
        (out_dir / "voice.json").write_text(json.dumps(voice, indent=1))
    with db.session(s.db_path) as c:
        calls = [dict(r) for r in c.execute("SELECT ts, model, endpoint, ok, status_code, latency_ms FROM model_calls WHERE job_id = ? ORDER BY id", (jid,))]
    with open(out_dir / "meta_calls.jsonl", "w") as f:
        for r in calls:
            f.write(json.dumps({"timestamp": datetime.fromtimestamp(r["ts"], timezone.utc).isoformat(timespec="milliseconds"), "model": r["model"],
                                "endpoint": r["endpoint"], "status": r["status_code"], "ok": bool(r["ok"]),
                                "latency_ms": None if r["latency_ms"] is None else round(r["latency_ms"], 1)}) + "\n")
    by = {}
    for r in calls:
        k = by.setdefault(r["model"], {"calls": 0, "ok": 0})
        k["calls"] += 1
        k["ok"] += r["ok"]
    (out_dir / "timings.json").write_text(json.dumps(recorded | {"api_wall_s": {k: round(x, 1) for k, x in t.items()},
                                                            "pipeline_per_stage_s": rep.get("timings"), "meta_calls": by,
                                                            "model_npz_mb": model_mb, "model_in_bundle": model_mb <= MAX_MODEL_MB,
                                                            "voice": "sent" if voice else "no voice file"}, indent=1))
    print(json.dumps({"timings": t, "held_out_iou": rep.get("held_out_iou"), "dims": {k: rep["dims_mm"][k]["measured"] for k in rep["dims_mm"]},
                      "meta_calls": by, "quality": rep["quality"], "cards_missing": rep.get("cards_missing"), "model_npz_mb": model_mb,
                      "staged": {p: len(list((stage / p).iterdir())) for p in ("cards", "review", "inside")}, "frames": src}, indent=1, default=str))
    print("job data kept for inspection in", data)


if __name__ == "__main__":
    main()
