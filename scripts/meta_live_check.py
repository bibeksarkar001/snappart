"""Live check of every Meta model path through backend/app/meta_client.py (REAL calls, small cost). Prints measured numbers only; never
prints the key. Reads the private part-B cache in place (nothing is copied into the repo); writes a throwaway database under the given dir.

Usage (repo root): python scripts/meta_live_check.py OUT_DIR [--sam-calls 40] [--parallel 16]"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import db, meta_client as M, spark, voice  # noqa: E402
from app.settings import Settings  # noqa: E402
from capture import masks as MK, sam as S  # noqa: E402

CACHE = ROOT / "backend" / "spikes" / "out" / "carve" / "cache"
V4 = ROOT / "backend" / "spikes" / "out" / "v4" / "SEND_TO_CLAUDE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sam-calls", type=int, default=40)
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Settings(data_dir=out, environment="test", sam_concurrency=a.parallel, meta_concurrency=a.parallel)
    with db.session(s.db_path) as c:
        db.migrate(c)
    if s.model_api_key is None:
        sys.exit("MODEL_API_KEY not configured")
    cl = M.MetaClient(s, None)
    rep = {}

    # 1. SAM 3.1: n crops in parallel through capture/sam.adapter; masks compared with the cached v1 SAM masks of the same crops
    crops = sorted((CACHE / "crops").glob("*.jpg"))
    crops = [p for p in crops if (CACHE / "sam" / f"{p.stem}.json").exists()]
    pick = [crops[i] for i in range(0, len(crops), max(len(crops) // a.sam_calls, 1))][:a.sam_calls]
    jobs = [(p.stem, cv2.imread(str(p)), "brown plastic buckle", None) for p in pick]
    t0 = time.perf_counter()
    got = S.adapter(cl.segment, log=print)(jobs)
    wall = time.perf_counter() - t0
    ious = []
    for key, crop, _, _ in jobs:
        r = got.get(key)
        meta = json.loads((CACHE / "sam" / f"{key}.json").read_text())
        old = [cv2.imread(str(CACHE / "sam" / f"{key}_{j}.png"), 0) for j in range(len(meta["confs"]))]
        if r and old:
            new = max(r, key=lambda cm: cm[0])[1]
            ious.append(max(MK.iou(new, o) for o in old if o is not None and o.shape == new.shape) if any(o is not None and o.shape == new.shape for o in old) else float("nan"))
    with db.session(s.db_path) as c:
        rows = c.execute("SELECT ok, status_code, latency_ms FROM model_calls WHERE model = 'sam-3.1'").fetchall()
    lat = [r["latency_ms"] / 1000 for r in rows if r["ok"]]
    rep["sam"] = {"calls": len(jobs), "parallel": a.parallel, "wall_s": round(wall, 1), "with_masks": sum(1 for k, *_ in jobs if got.get(k)),
                  "http_tries": len(rows), "failed_tries": sum(1 for r in rows if not r["ok"]), "n429": sum(1 for r in rows if r["status_code"] == 429),
                  "latency_median_s": round(statistics.median(lat), 2) if lat else None,
                  "iou_vs_cached_v1_masks": {"median": round(float(np.nanmedian(ious)), 4) if ious else None, "min": round(float(np.nanmin(ious)), 4) if ious else None,
                                             "n": len(ious)}}
    print("SAM", json.dumps(rep["sam"]))

    # 2. Muse Spark, one call per job
    report = json.loads((V4 / "report.json").read_text())
    from app import results
    q0 = report["questions"][0]
    q = {"id": q0["id"], "title": q0["question"], "detail": "", "hint": q0["hint"], "unit": q0["unit"], "default": q0["default"], "min": 0.2,
         "max": 11.69, "choices": q0.get("choices"), "label": "Slots"}
    item = {"id": "R01", "kind": "lump", "reason": "surface deviates from its local plane", "question": "Is it flat?", "choices": ["Keep", "Remove", "Make flat"],
            "centre": [151.0, 101.0, 9.0], "normal": [0, 0, -1.0], "polygon_xy": [[149.5, 86.0], [153.0, 86.0], [153.0, 117.0], [149.5, 117.0]],
            "z_mm": [7.5, 8.5, 9.5], "feature_id": None}
    frames = [p for p in sorted((CACHE / "frames" / "A_orbit_01").glob("*.jpg"))][:2]
    small = [cv2.imencode(".jpg", cv2.resize(cv2.imread(str(p)), None, fx=0.4, fy=0.4))[1].tobytes() for p in frames]
    overlays = []
    for key, crop, _, _ in jobs[:2]:
        o = crop.copy()
        if got.get(key):
            cs, _ = cv2.findContours(max(got[key], key=lambda cm: cm[0])[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, (255, 0, 255), 3)
        overlays.append(cv2.imencode(".jpg", o)[1].tobytes())
    t = time.perf_counter()
    rep["spark"] = {
        "word_question": spark.word_question(cl, q),
        "summarize": spark.summarize(cl, report),
        "propose": spark.propose(cl, "this part is flat like the bottom", item, ["grooves", "bump", "plate"]),
        "capture_opinion": spark.capture_opinion(cl, [(1, b) for b in small]),
        "outline_check": spark.outline_check(cl, overlays),
        "wall_s": None}
    rep["spark"]["wall_s"] = round(time.perf_counter() - t, 1)
    for k, v in rep["spark"].items():
        print("SPARK", k, json.dumps(v)[:400])

    # 3. Muse Voice Transcribe: a spoken sentence made with macOS `say`, converted by voice.to_wav, then a proposal from the transcript
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
        sentence = "Please make this area flat, like the bottom of the part."
        subprocess.run(["say", "-o", str(aiff), sentence], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "2", str(aiff), str(wav)], check=True)   # 44.1 kHz stereo: tests conversion
        raw = wav.read_bytes()
    w, dur = voice.to_wav(raw, s.max_audio_bytes)
    t = time.perf_counter()
    tr = M.run(cl.transcribe(w, voice.KEYWORDS))
    rep["voice"] = {"spoken": sentence, "transcript": tr["transcript"], "audio_s": round(dur, 2), "latency_s": round(time.perf_counter() - t, 2),
                    "proposal": spark.propose(cl, tr["transcript"], item, ["grooves", "bump", "plate"])}
    print("VOICE", json.dumps(rep["voice"])[:600])

    with db.session(s.db_path) as c:
        rep["model_calls"] = [dict(r) for r in c.execute("SELECT model, SUM(ok) ok, SUM(1 - ok) failed, ROUND(AVG(latency_ms)) avg_ms FROM model_calls GROUP BY model")]
        rep["budget_today"] = [dict(r) for r in c.execute("SELECT day, model, calls FROM budget_ledger")]
    print("CALLS", json.dumps(rep["model_calls"]))
    (out / "live_check.json").write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
