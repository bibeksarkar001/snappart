"""After a build: questions (with bounds and cards), answers, corrections with an undo stack, rebuild (no AI call), review items and the
choices that turn them into corrections, and the deliverables (model.stl, preview.stl <= 250k triangles, preview.glb, report.json).

Files under jobs/{id}/outputs/: model.npz/.json (editable model), report_build.json (the build's report, never changed), report.json
(the current one, rewritten by each rebuild), model.stl, preview.stl, preview.glb, cards/<question>/, review/<item>.png."""
import copy
import json
import secrets
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from capture import carve, corrections as Cr, gates as G, learnings as L, model as Mo, question_cards as Q, report as Rp, review as Rv

from . import db, events
from .errors import ApiError

PREVIEW_MAX_TRIANGLES = 250_000
LABEL = {"grooves": "Slots", "lock_bump": "Lock ridge", "plate": "Flat plate"}
MODEL_KEY = {"lock_bump": "bump"}                      # review.py names the ridge "lock_bump"; the editable model stores it as "bump"


# ---------------------------------------------------------------- reading

def file_url(jid, rel) -> str:
    return f"/v1/jobs/{jid}/files/{rel}"


def base_report(storage, jid) -> dict:
    """The build's own report (report_build.json; created from report.json on first use)."""
    if not storage.exists(jid, "outputs/report.json"):
        raise ApiError(409, "no_result", "This part has no result yet.")
    if not storage.exists(jid, "outputs/report_build.json"):
        storage.put(jid, "outputs/report_build.json", storage.get(jid, "outputs/report.json"))
    return json.loads(storage.get(jid, "outputs/report_build.json"))


def current_report(storage, jid) -> dict:
    base_report(storage, jid)
    return json.loads(storage.get(jid, "outputs/report.json"))


def load_model(storage, jid) -> dict:
    if not storage.exists(jid, "outputs/model.npz"):
        raise ApiError(409, "no_model", "This part has no editable model yet.")
    return Mo.load(storage.path(jid, "outputs/model"))


def part_dims(rep) -> dict:
    d = rep["dims_mm"]
    return {"length_mm": d["length"]["measured"], "width_mm": d["width"]["measured"], "height_mm": d["height"]["measured"]}


def stored_answers(conn, jid) -> dict:
    return {r["question_id"]: json.loads(r["value"]) for r in conn.execute("SELECT question_id, value FROM answers WHERE job_id = ?", (jid,))}


# ---------------------------------------------------------------- questions and answers

def _feature_value(rep, fid):
    return next((f for f in rep["features"] if f["id"] == fid), None)


def _detail(q, rep) -> str:
    """One line of context built only from measured report values (no invented numbers)."""
    if q["id"] == "groove_depth_mm":
        lo, sh, n = (_feature_value(rep, k) for k in ("grooves.long", "grooves.short", "grooves.count"))
        if lo and sh and n:
            return f"The {n['value']} openings measured {lo['value']} x {sh['value']} mm. A camera cannot see how far down they go."
    if q["id"] == "bump_height_mm":
        e = _feature_value(rep, "lock_ridge.extent_out_of_wall")
        if e and e["views"]:
            return f"{e['views']} head-on frames measured about {e['value']} mm."
    if q["id"] == "bump_top_shape":
        return "No frame shows the ridge end-on, so its top is an estimate until you say."
    if q["id"] == "plate_thickness_mm":
        sep = _feature_value(rep, "plate.faces_separation")
        if sep and sep["value"] is not None:
            return f"The video measured {sep['value']} mm between the plate's faces."
    return ""


def questions_view(storage, jid, answers) -> list[dict]:
    """The build's questions with everything the Review screen needs: kind, bounds, default, card, answer state."""
    rep = base_report(storage, jid)
    part = part_dims(rep)
    out = []
    for q in rep["questions"]:
        qid, b = q["id"], Q.bounds(q["id"], part)
        sep = _feature_value(rep, "plate.faces_separation") if qid == "plate_thickness_mm" else None
        a = answers.get(qid)
        conflict = L.plate_conflict(sep["value"] if sep else None, a if isinstance(a, (int, float)) else None)   # learnings rule 4
        card = f"outputs/cards/{qid}/"
        out.append({
            "id": qid, "feature": q["feature_id"], "label": LABEL.get(q["feature_id"], q["feature_id"]),
            "kind": "choice" if q.get("choices") else "conflict" if conflict else "number", "default_source": q.get("default_source"),
            "title": q["question"], "detail": _detail(q, rep), "hint": q["hint"], "unit": q["unit"], "default": q["default"],
            "min": b[0] if b else None, "max": b[1] if b else None, "choices": q.get("choices"),
            "candidates": conflict["candidates"] if conflict else None, "conflict_message": conflict["message"] if conflict else None,
            "applies_to": (_feature_value(rep, "grooves.count") or {}).get("value") if q["feature_id"] == "grooves" else None,
            "impact": None,
            "answered": a is not None and not conflict, "answer": a,
            "status": None if a is None else "estimated" if a == "skip" else "conflict" if conflict else "user_supplied",
            "card": {"image": file_url(jid, f"cards/{qid}/card.png"), "diagram": file_url(jid, f"cards/{qid}/diagram.svg"),
                     "meta": file_url(jid, f"cards/{qid}/card.json")} if storage.exists(jid, card + "card.png") else None,
        })
    return out


def save_answers(conn, storage, jid, given: dict) -> dict:
    """Validate (bounds, units, choices; "skip" = I don't know -> labelled estimate) and store; returns {saved, open_questions}."""
    rep = base_report(storage, jid)
    ids = {q["id"] for q in rep["questions"]}
    part = part_dims(rep)
    clean = {}
    for qid, v in given.items():
        if qid not in ids:
            raise ApiError(422, "unknown_question", f"There is no question {qid!r} for this part.")
        if v != "skip":
            why = Q.check_answer(qid, v, part)
            if why is None and Q.bounds(qid, part) and not (Q.bounds(qid, part)[0] <= float(v) < Q.bounds(qid, part)[1]):
                why = f"it must be between {Q.bounds(qid, part)[0]} and {Q.bounds(qid, part)[1]} mm"
            if why:
                raise ApiError(422, "answer_out_of_range", f"{v}{' mm' if Q.bounds(qid, part) else ''} does not fit this part: {why}.")
            v = v if qid in Q.CHOICES else float(v)
        clean[qid] = v
    with db.tx(conn):
        for qid, v in clean.items():
            conn.execute("INSERT INTO answers (job_id, question_id, value, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(job_id, question_id) "
                         "DO UPDATE SET value = excluded.value, created_at = excluded.created_at", (jid, qid, json.dumps(v), db.now()))
    ans = stored_answers(conn, jid)
    return {"saved": sorted(clean), "open_questions": sum(1 for q in ids if q not in ans), "rebuild_needed": True}


# ---------------------------------------------------------------- corrections (undo stack)

def correction_rows(conn, jid, include_reverted=False) -> list[dict]:
    q = "SELECT id, seq, body, reverted, created_at FROM corrections WHERE job_id = ?" + ("" if include_reverted else " AND reverted = 0")
    return [{"id": r["id"], "seq": r["seq"], "correction": json.loads(r["body"]), "reverted": bool(r["reverted"]), "created_at": r["created_at"]}
            for r in conn.execute(q + " ORDER BY seq", (jid,))]


def add_correction(conn, storage, jid, body: dict) -> str:
    """Validate against the corrections-engine schema (the five ops only, typed parameters) and the model's feature / plane names."""
    m = load_model(storage, jid)
    feats = m["features"]
    try:
        Cr.validate(body, list(feats) + ["bump"], list(feats.get("planes", {})))
    except ValueError as e:
        raise ApiError(422, "invalid_correction", str(e)) from None
    cid = "c_" + secrets.token_hex(6)
    with db.tx(conn):
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM corrections WHERE job_id = ?", (jid,)).fetchone()[0]
        conn.execute("INSERT INTO corrections (id, job_id, seq, body, created_at) VALUES (?, ?, ?, ?, ?)", (cid, jid, seq, json.dumps(body), db.now()))
    return cid


def revert_correction(conn, jid, cid) -> None:
    with db.tx(conn):
        if conn.execute("UPDATE corrections SET reverted = 1 WHERE job_id = ? AND id = ? AND reverted = 0", (jid, cid)).rowcount != 1:
            raise ApiError(404, "not_found", "No such correction (or it was already undone).")


# ---------------------------------------------------------------- deliverables

def write_stl_bytes(v, f) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".stl") as t:
        carve.write_stl(t.name, v, f)
        return Path(t.name).read_bytes()


def glb_bytes(v, f) -> bytes:
    """Binary glTF 2.0 of a triangle mesh: positions in mm with a node scale of 0.001 (glTF is in metres) and a rotation turning the part's
    z-up frame into glTF's y-up, smooth vertex normals, uint32 indices."""
    v = np.asarray(v, np.float32)
    f = np.asarray(f, np.uint32)
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    pos, nrm, idx = v.tobytes(), n.tobytes(), f.ravel().tobytes()
    bin_ = pos + nrm + idx
    bin_ += b"\0" * (-len(bin_) % 4)
    gltf = {"asset": {"version": "2.0", "generator": "SnapPart", "extras": {"unit_of_positions": "mm"}},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "scale": [0.001] * 3, "rotation": [-0.7071068, 0, 0, 0.7071068]}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
            "buffers": [{"byteLength": len(bin_)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": 34962},
                            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(nrm), "target": 34962},
                            {"buffer": 0, "byteOffset": len(pos) + len(nrm), "byteLength": len(idx), "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3", "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                          {"bufferView": 1, "componentType": 5126, "count": len(v), "type": "VEC3"},
                          {"bufferView": 2, "componentType": 5125, "count": f.size, "type": "SCALAR"}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(bin_), b"BIN\0") + bin_)


def preview_mesh(field, g, v, f):
    """The mesh itself if it has <= 250k triangles, else the same field meshed on a coarser grid (every k-th voxel) until it does."""
    if len(f) <= PREVIEW_MAX_TRIANGLES:
        return v, f
    for k in range(2, 9):
        gc = carve.Grid(g.lo, g.lo + g.vs * k, g.vs * k)
        gc.shape = field[::k, ::k, ::k].shape
        vc, fc = Mo.mesh(field[::k, ::k, ::k], gc)
        if len(fc) <= PREVIEW_MAX_TRIANGLES:
            return vc, fc
    return vc, fc


def write_deliverables(storage, jid, field, g, v, f) -> dict:
    pv, pf = preview_mesh(field, g, v, f)
    storage.put(jid, "outputs/model.stl", write_stl_bytes(v, f))
    storage.put(jid, "outputs/preview.stl", write_stl_bytes(pv, pf))
    storage.put(jid, "outputs/preview.glb", glb_bytes(pv, pf))
    return {"model_triangles": int(len(f)), "preview_triangles": int(len(pf))}


# ---------------------------------------------------------------- rebuild

def model_answers(stored: dict, question_ids) -> tuple[dict, list[str]]:
    """Stored answers -> rebuild answers (skips leave the frozen default, labelled estimated) and the ids the user supplied."""
    a, given = {}, []
    for qid, v in stored.items():
        if qid in question_ids and v != "skip":
            a[qid] = v
            given.append(qid)
    return a, given


def _diff(old: dict, new: dict) -> list[dict]:
    o = {f["id"]: f for f in old.get("features", [])}
    out = []
    for f in new["features"]:
        p = o.get(f["id"])
        if p is None or p["value"] != f["value"] or p["status"] != f["status"]:
            out.append({"id": f["id"], "from": None if p is None else p["value"], "to": f["value"],
                        "status_from": None if p is None else p["status"], "status_to": f["status"]})
    ob, nb = old.get("rebuild") or {}, new["rebuild"]
    if ob.get("volume_mm3") is not None:
        out.append({"id": "model.volume_mm3", "from": ob["volume_mm3"], "to": nb["volume_mm3"], "status_from": None, "status_to": None})
    return out


def rebuild(conn, storage, jid) -> dict:
    """Answers + corrections -> new model.stl, previews and report.json, in about 2 s on part B, with no model call."""
    t0 = time.perf_counter()
    base = base_report(storage, jid)
    old = current_report(storage, jid)
    m = load_model(storage, jid)
    qids = {q["id"] for q in base["questions"]}
    stored = stored_answers(conn, jid)
    a, given = model_answers(stored, qids)
    user_corr = correction_rows(conn, jid)
    corr = list(base.get("corrections") or []) + [r["correction"] for r in user_corr]
    height = Mo.part_height(m["field"], m["grid"])
    full = Mo.resolve_answers(a, m["features"], height)
    try:
        field = Mo.apply(m["field"], m["grid"], m["features"], full, corr)
        feats_eff = Mo.correct_features(m["field"], m["grid"], m["features"], corr)
    except ValueError as e:
        raise ApiError(422, "rebuild_rejected", f"This change cannot be applied to the model: {e}") from None
    v, f = Mo.mesh(field, m["grid"])
    out = write_deliverables(storage, jid, field, m["grid"], v, f)
    quality = G.check(v, f, field, base.get("held_out_iou"), m["grid"])
    cen, pl = carve.edge_census(v, f), carve.obb_plan(v)
    rep = copy.deepcopy(base)
    rep["features"] = Rp.features_list(feats_eff, full, set(given), Mo.estimate_answers(feats_eff, height)[1])
    rep["questions"] = Rp.questions(feats_eff, {k: stored[k] for k in given}, height)
    rep["answers"] = stored
    rep["corrections"] = corr
    rep["user_corrections"] = [{"id": r["id"], "correction": r["correction"]} for r in user_corr]
    rep["rebuild"] = {"seconds": round(time.perf_counter() - t0, 2), "at": db.now(), "watertight": bool(cen["watertight"]),
                      "volume_mm3": round(float(cen["volume_mm3"]), 1), "model_size_mm": [round(pl["length_mm"], 2), round(pl["width_mm"], 2),
                                                                                      round(pl["z_max"] - pl["z_min"], 2)], **out}
    rep["quality"] = quality
    storage.put(jid, "outputs/report.json", json.dumps(rep, indent=1, default=float).encode())
    changes = _diff(old, rep)
    open_q = sum(1 for q in qids if q not in stored)
    events.emit(conn, jid, "build_finished", {"rebuild": True, "seconds": rep["rebuild"]["seconds"], "changes": len(changes)})
    return {"seconds": rep["rebuild"]["seconds"], "changes": changes, "open_questions": open_q, "watertight": rep["rebuild"]["watertight"],
            "volume_mm3": rep["rebuild"]["volume_mm3"], "quality": {"status": quality["status"], "reasons": quality["reasons"]}, **out}


# ---------------------------------------------------------------- review items

def user_marks(storage, jid) -> list[dict]:
    return json.loads(storage.get(jid, "review/user_marks.json")) if storage.exists(jid, "review/user_marks.json") else []


def review_items(conn, storage, jid) -> list[dict]:
    rep = base_report(storage, jid)
    ans = stored_answers(conn, jid)
    out = []
    details = [{"id": f"D{i + 1:02d}", "kind": "detail", "reason": f"possible {d['kind'].replace('_', ' ')}, {d['size_mm']}, "
                f"{d['evidence']}", "question": "Is this text or a mark?" if d["kind"] == "text_or_mark" else f"Is this a {d['kind']} or a mark?",
                "choices": ["Text", "Just a mark"] if d["kind"] == "text_or_mark" else [f"It is a {d['kind']}", "Just a mark"],
                "centre": d.get("centre"), "normal": d.get("normal") or d.get("axis"), "polygon_xy": None, "z_mm": None, "feature_id": None}
               for i, d in enumerate(d for d in rep.get("details") or [] if d["status"] == "question")]
    for it in list(rep.get("review_items") or []) + details + user_marks(storage, jid):
        pic = f"review/{it['id']}.png"
        out.append({**{k: it.get(k) for k in ("id", "kind", "reason", "question", "choices", "centre", "normal", "polygon_xy", "z_mm", "area_mm2",
                                              "feature_id", "note")},
                    "image": file_url(jid, pic) if storage.exists(jid, "outputs/" + pic) else None, "decision": ans.get(f"review:{it['id']}")})
    return out


def add_mark(storage, jid, point, radius_mm, normal, note) -> dict:
    """A spot the user marks on the model (part frame, mm): a 12-sided region around the point, offered with the same three choices."""
    marks = user_marks(storage, jid)
    p, n = np.asarray(point, float), np.asarray(normal or [0, 0, 1.0], float)
    n /= np.linalg.norm(n)
    poly = [[round(p[0] + radius_mm * np.cos(t), 3), round(p[1] + radius_mm * np.sin(t), 3)] for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    q, ch = Rv.QUESTIONS["lump"]
    it = {"id": f"U{len(marks) + 1:02d}", "kind": "user_mark", "reason": "marked by you", "question": "What should happen to the spot you marked?",
          "choices": ch, "centre": p.round(3).tolist(), "normal": n.round(3).tolist(), "polygon_xy": poly,
          "z_mm": [round(p[2] - 0.5, 3), round(p[2], 3), round(p[2] + 0.5, 3)],              # a thin band at the marked surface
          "area_mm2": round(float(np.pi * radius_mm ** 2), 1),
          "note": note}
    storage.put(jid, "review/user_marks.json", json.dumps(marks + [it]).encode())
    return it


def review_correction(it: dict, choice: str) -> dict | None:
    """The corrections-engine op for a review choice (None = keep as is). Make flat trims what stands above (or below) the region's middle
    height; it never adds material."""
    if choice in ("Keep", "Current side") or it["kind"] == "detail":       # detail answers are recorded; they do not change the geometry
        return None
    fid = MODEL_KEY.get(it.get("feature_id"), it.get("feature_id"))
    if choice == "Opposite side":
        return {"op": "move_feature_to_opposite_face", "feature_id": fid, "source": "review", "review_item": it["id"]}
    if choice == "Remove" and it["kind"] == "one_video":
        return {"op": "add_feature", "feature_id": fid, "feature": {}, "source": "review", "review_item": it["id"],
                "reason": "user: this feature is not on the part"}
    if not it.get("z_mm") or not it.get("polygon_xy"):
        raise ApiError(422, "choice_not_available", "This area has no height range recorded, so it can only be kept for now.")
    zlo, zmid, zhi = it["z_mm"]
    up = (it.get("normal") or [0, 0, 1])[2]
    if abs(up) < 0.5:
        raise ApiError(422, "choice_not_available", "Remove and Make flat work on areas facing up or down for now; this one faces sideways.")
    if choice == "Remove":
        z = [zlo - 0.2, zhi + 3.0] if up > 0 else [zlo - 3.0, zhi + 0.2]
    else:                                                                   # Make flat
        z = [zmid, zhi + 3.0] if up > 0 else [zlo - 3.0, zmid]
    return {"op": "remove_region", "region": {"polygon_xy": it["polygon_xy"], "z_mm": z}, "source": "review", "review_item": it["id"],
            "reason": f"user chose {choice.lower()}"}


def choose(conn, storage, jid, item_id: str, choice: str) -> dict:
    it = next((x for x in review_items(conn, storage, jid) if x["id"] == item_id), None)
    if it is None:
        raise ApiError(404, "not_found", "No such review item.")
    if choice not in (it["choices"] or []):
        raise ApiError(422, "invalid_choice", f"Choose one of: {', '.join(it['choices'])}.")
    corr = review_correction(it, choice)
    for r in correction_rows(conn, jid):                                    # changing your mind undoes the earlier choice's edit
        if r["correction"].get("review_item") == item_id:
            revert_correction(conn, jid, r["id"])
    cid = add_correction(conn, storage, jid, corr) if corr else None
    with db.tx(conn):
        conn.execute("INSERT INTO answers (job_id, question_id, value, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(job_id, question_id) "
                     "DO UPDATE SET value = excluded.value, created_at = excluded.created_at", (jid, f"review:{item_id}", json.dumps(choice), db.now()))
    return {"item": item_id, "decision": choice, "correction_id": cid, "rebuild_needed": cid is not None}


# ---------------------------------------------------------------- coverage: "film N more seconds of this area"

def coverage_items(conn, storage, jid) -> list[dict]:
    """The build's requests for more footage, each with the user's decision (filming / skipped / None)."""
    rep = base_report(storage, jid)
    ans = stored_answers(conn, jid)
    return [{**r, "decision": ans.get(f"coverage:{r['id']}")} for r in rep.get("coverage_requests") or []]


def open_blocking(conn, storage, jid) -> list[dict]:
    return [r for r in coverage_items(conn, storage, jid) if r.get("blocking") and r["decision"] is None]


def decide_coverage(conn, storage, jid, request_id: str, action: str) -> dict:
    item = next((r for r in coverage_items(conn, storage, jid) if r["id"] == request_id), None)
    if item is None:
        raise ApiError(404, "not_found", "No such request.")
    value = {"film": "filming", "skip": "skipped"}[action]
    with db.tx(conn):
        conn.execute("INSERT INTO answers (job_id, question_id, value, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(job_id, question_id) "
                     "DO UPDATE SET value = excluded.value, created_at = excluded.created_at", (jid, f"coverage:{request_id}", json.dumps(value), db.now()))
    if action == "film":
        base = {str(r["video"]): r["m"] for r in conn.execute("SELECT video, MAX(frame_index) m FROM uploads WHERE job_id = ? GROUP BY video", (jid,))}
        storage.put(jid, "more_footage.json", json.dumps({"request": item, "baseline": base, "at": db.now()}).encode())
    return {**item, "decision": value}


def footage_covers(storage, jid, conn) -> dict:
    """After 'film more': do the new frames face the requested area? Only video 1's frames can be checked (its mat frame is the part frame);
    new usable frames of other videos are accepted but marked unverified. Returns {ok, verified, new_frames, facing_frames, message}."""
    mf = json.loads(storage.get(jid, "more_footage.json"))
    cc = json.loads(storage.get(jid, "capture_check.json")) if storage.exists(jid, "capture_check.json") else {"videos": {}}
    n = np.asarray(mf["request"]["normal"], float)
    new, facing, other = 0, 0, 0
    for v, vd in cc["videos"].items():
        for fr in vd.get("frame_stats", []):
            if fr["index"] <= mf["baseline"].get(str(v), -1) or not fr.get("usable"):
                continue
            new += 1
            if str(v) == "1" and fr["az"] is not None:
                a, e = np.radians(fr["az"]), np.radians(fr["el"])
                d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                facing += int(d @ n > 0.5)
            else:
                other += 1
    ok = facing > 0 or other > 0
    return {"ok": ok, "verified": facing > 0, "new_frames": new, "facing_frames": facing,
            "message": None if ok else "The new frames do not show the area that was asked for. " + mf["request"]["message"]}


def quality_status(storage, jid) -> dict:
    """{"status": ready | draft | unknown, "reasons"} of the current model."""
    try:
        q = current_report(storage, jid).get("quality") or {}
    except ApiError:
        return {"status": "unknown", "reasons": []}
    return {"status": q.get("status", "unknown"), "reasons": q.get("reasons", [])}

