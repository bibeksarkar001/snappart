"""Real build step (worker subprocess, real mode): the uploaded frames go through the capture pipeline (capture/pipeline.run_frames) with
SAM 3.1 through the Meta client, then held-out IoU, quality gates, details, review items, question cards and previews, written to the
job's outputs/ in the same layout as the mock bundle. Nothing here changes what the pipeline computes; it only wires it to the job.

Video roles are fixed by upload order, as the capture screen asks: video 1 is the reference orbit (part resting on its base), video 2 the
second orbit (part on its edge for part B), video 3 the part flipped over. A requeued real build starts again from the beginning (no
stage checkpoints yet)."""
import json
import re
import time

import cv2
import numpy as np

from . import events, meta_client, results
from .errors import BuildFailure
from capture import details as Dt, gates as G, holdout as H, learnings as Ln, model as Mo, pipeline as P, question_cards as Q, \
    report as Rp, review as Rv, sam as S

ROLES = {"ref": "1", "stand": "2", "flat": "3"}


def frames_of(conn, st, job_id):
    out = {}
    for r in conn.execute("SELECT video, frame_index FROM uploads WHERE job_id = ? ORDER BY video, frame_index", (job_id,)).fetchall():
        img = cv2.imdecode(np.frombuffer(st.get(job_id, f"videos/{r['video']}/frames/{r['frame_index']:04d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            out.setdefault(str(r["video"]), {})[r["frame_index"]] = img
    return out


def run(conn, st, s, job_id: str) -> tuple[str, str]:
    t_all = time.perf_counter()
    if s.model_api_key is None:
        raise BuildFailure("no_model_key", "This server has no Meta API key, so it cannot outline the part (SAM 3.1).")
    fr = frames_of(conn, st, job_id)
    if sorted(fr) != ["1", "2", "3"]:
        raise BuildFailure("videos_missing", "A real build needs all three videos (high orbit, second orbit, flipped).")
    name = conn.execute("SELECT part_name FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    concepts = ((name,) if name else ()) + tuple(c for c in P.C.SAM_CONCEPTS if c != name)   # a job's part name goes first (config.py)
    cl = meta_client.MetaClient(s, job_id)
    sam = S.adapter(cl.segment)

    def log(m):
        a = re.search(r"\[(.+)\] start$", m)
        b = re.search(r"\[(.+)\] ([0-9.]+) s$", m)
        if a:
            events.emit(conn, job_id, "stage_started", {"stage": a.group(1)})
        elif b:
            events.emit(conn, job_id, "stage_finished", {"stage": b.group(1), "seconds": float(b.group(2))})
    work = st.path(job_id, "work/pipeline")
    try:
        rec = P.run_frames(fr, ROLES, work, sam, answers=None, log=log, concepts=concepts)
    except meta_client.BudgetExceeded as e:
        raise BuildFailure("daily_capacity_used", e.message) from None
    except Exception as e:                                               # the pipeline could not make a model from this footage
        raise BuildFailure("build_failed", f"The model could not be built from this footage ({type(e).__name__}).") from e
    write_inside(st, job_id, fr, P.run_frames.poses, P.run_frames.masks, P.run_frames.intr, s.part_only_pictures)
    return finish(conn, st, job_id, P.run_frames.views, P.run_frames.T, *P.run_frames.field, rec["features"], P.run_frames.crops,
                  dict(rec["seconds"]), {k: rec[k] for k in ("sam_calls", "views", "calib_rms_px")}, t_all, P.run_frames.masks,
                  s.part_only_pictures)


THUMB_PX = 320


def part_only_crop(crop, mask, outline=True, pad=6):
    """The part alone, for sharing: the crop cut to the outline's box, everything outside the outline turned white."""
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    out = np.where(m[..., None] > 0, crop, 255).astype(np.uint8)
    if outline:
        cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (60, 200, 60), 2, cv2.LINE_AA)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1
    return out[y0:y1, x0:x1]


def write_inside(st, job_id, fr, poses, masks, intr, part_only=False):
    """How it works, for this job only (outputs/inside/, deleted with the job): a small JPEG of every uploaded frame with SAM's accepted
    outline drawn on it, and the camera pose per frame. With part_only (recordings), each image is instead the part alone with its outline,
    cut tight, and frames without an accepted outline get no image."""
    frames, pose_rows = [], []
    for v, imgs in sorted(fr.items()):
        K, dist = intr[v]
        for i, img in sorted(imgs.items()):
            key = f"{v}_{i}"
            und = cv2.undistort(img, K, dist, None, K)
            name = f"inside/{v}_{i:04d}.jpg"
            frames.append({"video": int(v), "index": i, "pose": key in poses, "outline_accepted": key in masks, "file": name})
            if part_only:
                po = part_only_crop(und[poses[key][0][1]:poses[key][0][3], poses[key][0][0]:poses[key][0][2]].copy(), masks[key]) \
                    if key in masks else None
                if po is not None:
                    st.put(job_id, "outputs/" + name, cv2.imencode(".jpg", po, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
            if key in masks and not part_only:
                box = poses[key][0]
                full = np.zeros(und.shape[:2], np.uint8)
                full[box[1]:box[3], box[0]:box[2]] = masks[key]
                cs, _ = cv2.findContours((full > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(und, cs, -1, (60, 200, 60), max(2, und.shape[1] // 400), cv2.LINE_AA)
            if not part_only:
                s = THUMB_PX / max(und.shape[:2])
                th = cv2.resize(und, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                st.put(job_id, "outputs/" + name, cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            if key in poses:
                _, rv, tv = poses[key]
                R = cv2.Rodrigues(np.asarray(rv, float))[0]
                pose_rows.append({"video": int(v), "index": i, "rvec": np.round(rv, 6).tolist(), "tvec_mm": np.round(tv, 3).tolist(),
                                  "camera_centre_mm": np.round(-R.T @ np.asarray(tv, float), 2).tolist()})
    st.put(job_id, "outputs/inside/frames.json", json.dumps(frames).encode())
    st.put(job_id, "outputs/inside/poses.json", json.dumps({"frame": "mat (reference video's world, mm)", "poses": pose_rows,
                                                             "intrinsics": {v: np.round(K, 3).tolist() for v, (K, _) in intr.items()}}).encode())



def finish(conn, st, job_id, views, T, field, g, feats, crops, timings, extra, t_all, masks=None, part_only=False):
    """Everything after the pipeline's carve: held-out IoU, model files, previews, report, review items, cards. Split out so it can be
    checked on cached views without new Meta calls."""
    n_views = sum(len(v) for v in views.values())

    t = time.perf_counter()
    events.emit(conn, job_id, "stage_started", {"stage": "held-out check"})
    iou, per_video, _ = H.joint(views, T, Ln.k_for_views(n_views), vs=0.25, ref=ROLES["ref"])
    timings["held-out check"] = time.perf_counter() - t
    events.emit(conn, job_id, "stage_finished", {"stage": "held-out check", "seconds": round(timings["held-out check"], 1)})

    t = time.perf_counter()
    events.emit(conn, job_id, "stage_started", {"stage": "report, review, cards"})
    ans = Mo.default_answers(feats, Mo.part_height(field, g))
    a_full = Mo.resolve_answers(ans, feats, Mo.part_height(field, g))
    f_full = Mo.apply(field, g, feats, a_full)
    v0, f0 = Mo.mesh(field, g)
    v, fc = Mo.rebuild({"field": field, "grid": g, "features": feats}, ans)
    out = st.path(job_id, "outputs")
    out.mkdir(parents=True, exist_ok=True)
    Mo.save(out / "model", field, g, feats)
    st.put(job_id, "outputs/model.stl", results.write_stl_bytes(v, fc))
    pv, pf = results.preview_mesh(f_full, g, v, fc)
    st.put(job_id, "outputs/preview.stl", results.write_stl_bytes(pv, pf))
    glb = results.glb_bytes(pv, pf)
    st.put(job_id, "outputs/preview.glb", glb)
    st.put(job_id, "outputs/preview_before.glb", glb)                  # the model before any answer or correction; rebuilds keep it
    vv = [(n, x.moved(T[n])) for n in views for x in views[n]]
    if part_only:                                                       # recordings: pictures of the part alone (background white)
        imgs = [np.where(cv2.dilate((masks[x.name] > 0).astype(np.uint8), np.ones((3, 3), np.uint8))[..., None] > 0, crops[x.name], 255)
                .astype(np.uint8) if x.name in masks else np.full_like(crops[x.name], 255) for _, x in vv]
    else:
        imgs = [crops[x.name] for _, x in vv]                          # BGR crops, in the views' order (photos: kept in job storage only)
    occ = (pv, pf)                                                      # coarse occluder for the visibility test (review.best_view)
    items = Rv.find(v, fc, vv)
    rep = Rp.build((v0, f0), feats, ans, user_given=(), held_out_iou=round(iou, 4), views_moved=[x for _, x in vv],
                   timings=None, review_items=[{k: x for k, x in it.items() if k != "triangles"} | {"triangles": int(len(it["triangles"]))} for it in items],
                   details=Dt.mesh_candidates(v, fc), quality=G.check(v, fc, f_full, iou, g))
    for it in items:
        img, _ = Q.region_card(it["triangles"], v, fc, it["normal"], vv, imgs, occ, f"{it['id']}: {it['kind']}")
        st.put(job_id, f"outputs/review/{it['id']}.png", cv2.imencode(".png", img)[1].tobytes())
    rep["cards_missing"] = []
    for q in rep["questions"]:
        try:
            geom = Q.anchors(q["id"], feats, ans)
        except KeyError as e:                                           # the card code draws v4 ridges; the pipeline's bump is the v3 form
            rep["cards_missing"].append({"question": q["id"], "reason": f"no picture: the card cannot place this feature (missing {e})"})
            continue
        img, crec = Q.frame_card(geom, vv, imgs, occ)
        Q.write_card(out / "cards" / q["id"], q, img, crec or {}, Q.diagram(q["id"]))
    timings["report, review, cards"] = time.perf_counter() - t
    events.emit(conn, job_id, "stage_finished", {"stage": "report, review, cards", "seconds": round(timings["report, review, cards"], 1)})
    rep.update(mode="real", synthetic=False, meta_models_ran=sorted({r[0] for r in conn.execute(
        "SELECT model FROM model_calls WHERE job_id = ? AND ok = 1", (job_id,))}),
        **extra, held_out_iou_per_video=per_video,
        timings={"per_stage_s": {k: round(x, 1) for k, x in timings.items()}, "total_s": round(time.perf_counter() - t_all, 1)})
    st.put(job_id, "outputs/report.json", json.dumps(rep, indent=1, default=float).encode())
    events.emit(conn, job_id, "build_finished", {"outputs": st.list(job_id, "outputs/"), "questions": len(rep["questions"]), "mock": False})
    return ("needs_input", "questions the camera cannot answer") if rep["questions"] else ("reviewing", "ready for review")
