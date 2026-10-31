"""Replay mode: the same API and job states, but the build step delivers the recorded real run of part B (examples/recorded/part_b/)
instead of building, and no Meta model is ever called. Everything served is labelled "recorded run" with the recording date: the
X-SnapPart-Recorded header on every response, a "recorded" block in job, report, questions, summary and model-state responses, and
report warnings. Replay is the default when no MODEL_API_KEY is set (settings.py). When the bundle carries the editable model
(model.npz / model.json), answers, corrections, rebuilds, review choices, marks and typed proposals run offline on this computer and are
labelled so; filming more footage and transcribing new speech need a live key. model.stl and preview.stl are not in the bundle: they are
made from model.npz on this computer when first downloaded, labelled like any replay rebuild."""
import json
import shutil
from functools import lru_cache
from pathlib import Path

from . import db, events

EDITS_OFF = "Replay mode shows a recorded run; answering questions and changing the model need a live build with a Meta API key."
FILM_OFF = "Replay mode shows a recorded run; filming more footage needs a live build with a Meta API key."
OFFLINE = "rebuilt on your computer from a recorded run; no model call"
OFFLINE_PROPOSAL = "proposed on your computer by keyword rules from a recorded run; no model call"


def editable(s) -> bool:
    """The recorded bundle carries the editable model (model.npz / model.json), so answers and rebuilds can run offline."""
    return (Path(s.replay_dir) / "model.npz").is_file() and (Path(s.replay_dir) / "model.json").is_file()


@lru_cache
def recorded(bundle: str) -> dict:
    t = json.loads((Path(bundle) / "timings.json").read_text())
    calls = {}
    for line in (Path(bundle) / "meta_calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = calls.setdefault(r["model"], {"calls": 0, "ok": 0})
        k["calls"] += 1
        k["ok"] += bool(r["ok"])
    day = t["recorded_at"][:10]
    voice = json.loads((Path(bundle) / "voice.json").read_text()) if (Path(bundle) / "voice.json").is_file() else None
    return {"recorded_voice": voice, "label": f"recorded run, {day}", "recorded_at": t["recorded_at"], "part": t["part"], "live_model_calls": False,
            "recorded_meta_calls": calls, "pipeline_per_stage_s": (t.get("pipeline_per_stage_s") or {}).get("per_stage_s"),
            "note": f"Replay of a real run recorded on {day}. Nothing you upload is processed and no Meta model is called now."}


def note(s) -> dict:
    return recorded(str(s.replay_dir))


def run_build(conn, storage, settings, job_id: str) -> tuple[str, str]:
    """Deliver the recorded bundle as this job's outputs, with the recorded per-stage times as events (labelled recorded, not re-run)."""
    src, rec = Path(settings.replay_dir), note(settings)
    for name, sec in (rec["pipeline_per_stage_s"] or {}).items():
        events.emit(conn, job_id, "stage_finished", {"stage": name, "seconds": sec, "recorded": rec["label"]})
    keys = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and (rel in ("preview.glb", "model.npz", "model.json") or rel.startswith(("cards/", "review/", "inside/"))):
            storage.put(job_id, "outputs/" + rel, p.read_bytes())
            keys.append("outputs/" + rel)
    before = src / ("preview_before.glb" if (src / "preview_before.glb").is_file() else "preview.glb")
    if before.is_file():                                                # the recorded build's raw preview is the "before" model
        storage.put(job_id, "outputs/preview_before.glb", before.read_bytes())
    rep = json.loads((src / "report.json").read_text())
    rep.update(mode="replay", recorded=rec)
    rep["warnings"].insert(0, f"Recorded run ({rec['label']}): {rec['note']}")
    storage.put(job_id, "outputs/report.json", json.dumps(rep, indent=1).encode())
    keys.append("outputs/report.json")
    ans = answers(settings)                                             # the owner's recorded answers: the replay starts after them
    if ans:
        from . import results
        results.save_answers(conn, storage, job_id, ans["answers"])
    events.emit(conn, job_id, "build_finished", {"outputs": keys, "questions": len(rep["questions"]), "mock": False, "recorded": rec["label"]})
    return ("needs_input", "recorded run: questions the camera could not answer") if rep["questions"] else ("reviewing", "recorded run")


def answers(s) -> dict | None:
    """The owner's answers saved with the recording ({"answers": {question: value}, "status": "user_supplied", "source"}), or None."""
    p = Path(s.replay_dir) / "answers.json"
    return json.loads(p.read_text()) if p.is_file() else None


def calls(s) -> list[dict]:
    """The recorded run's Meta call log (labelled recorded by the caller)."""
    return [json.loads(x) for x in (Path(s.replay_dir) / "meta_calls.jsonl").read_text().splitlines()]


def questions(s) -> dict:
    return json.loads((Path(s.replay_dir) / "questions.json").read_text())


def summary(s) -> dict:
    return json.loads((Path(s.replay_dir) / "summary.json").read_text())
