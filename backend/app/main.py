"""FastAPI app. Run: `uvicorn backend.app.main:app` from the repo root (or `python -m backend.app.main`); workers run separately
(`python -m backend.app.worker`). The API holds no state of its own beyond the database and storage, so several replicas can run.

Every response carries X-SnapPart-Mode (mock | real) and X-Request-ID; every error uses the envelope in errors.py. Database and file work
runs in the thread pool, so the event loop (SSE streams, uploads) stays responsive."""
import asyncio
import hashlib
import io
import json
import logging
import secrets

from fastapi import FastAPI, Path as P, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from . import capture_check, db, events, jobqueue, jobs, models, ratelimit, resume
from .errors import ApiError, envelope, install
from .settings import Settings, get_settings
from .storage import JOB_ID, LocalStorage

log = logging.getLogger("snappart.api")
TERMINAL = {"failed", "cancelled", "expired"}


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    with db.session(s.db_path) as c:
        db.migrate(c)
    app = FastAPI(title="SnapPart API", version="0.2.0", docs_url=None, redoc_url=None)
    st = LocalStorage(s.data_dir)
    pep = resume.pepper(s)
    app.state.settings, app.state.storage = s, st
    install(app)

    @app.middleware("http")
    async def frame(request: Request, call_next):
        request.state.request_id = secrets.token_hex(8)                 # server-made; a client-sent id is not trusted
        try:
            resp = await call_next(request)
        except Exception:                                               # never leak a trace, path or SQL
            log.exception("unhandled error, request %s", request.state.request_id)
            resp = envelope(request, 500, "internal", "Something went wrong on our side. Please try again.")
        resp.headers["X-SnapPart-Mode"] = s.mode
        resp.headers["X-Request-ID"] = request.state.request_id
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    def conn():
        return db.session(s.db_path)

    def job_id_ok(job_id: str) -> str:
        if not JOB_ID.match(job_id):
            raise ApiError(404, "not_found", "No such job.")
        return job_id

    def who(request: Request) -> str:
        return ratelimit.ip_hash(ratelimit.client_ip(request, s.trusted_proxies), pep)

    def job_view(c, job_id: str) -> dict:
        row = jobs.get(c, job_id)
        out = jobs.public(row, st.list(job_id, "outputs/"))
        out.update(jobqueue.position(c, job_id, s))
        n = c.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM uploads WHERE job_id = ?", (job_id,)).fetchone()
        out["uploads"] = {"frames": n[0], "bytes": n[1]}
        if st.exists(job_id, "capture_check.json"):
            cc = json.loads(st.get(job_id, "capture_check.json"))
            out["capture_check"] = {"verdict": cc["verdict"], "videos": {v: {k: x[k] for k in ("verdict", "usable_frames", "frames", "tips")}
                                                                         for v, x in cc["videos"].items()}}
        else:
            out["capture_check"] = None
        return out

    # ------------------------------------------------------------------ health

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        checks = {}
        try:
            with conn() as c:
                checks["db"] = c.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] is not None
        except Exception:
            checks["db"] = False
        probe = s.data_dir / ".ready-probe"
        try:
            probe.write_bytes(b"ok")
            probe.unlink()
            checks["storage"] = True
        except OSError:
            checks["storage"] = False
        body = {"ready": all(checks.values()), "mode": s.mode, "mock": s.mock, "environment": s.environment, "checks": checks}
        return JSONResponse(body, 200 if body["ready"] else 503)

    # ------------------------------------------------------------------ jobs

    @app.post("/v1/jobs", response_model=models.JobCreated, status_code=201)
    def create_job(request: Request, body: models.JobCreate | None = None):
        ip = who(request)
        code = resume.new_code()
        with conn() as c:
            ratelimit.limit(c, f"create:{ip}", s.rate_create_per_hour, what="new parts")
            jid = jobs.create(c, s.mode, ip)
            with db.tx(c):
                c.execute("UPDATE jobs SET resume_hash = ?, part_name = ? WHERE id = ?",
                          (resume.hash_code(code, pep), body.part_name if body else None, jid))
        return {"job_id": jid, "resume_code": code, "state": "created", "mode": s.mode, "limits": s.public_limits()}

    @app.get("/v1/jobs/{job_id}", response_model=models.Job)
    def get_job(job_id: str):
        with conn() as c:
            return job_view(c, job_id_ok(job_id))

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str):
        """Deletes frames, outputs, answers, corrections, events and the job row; keeps only an anonymous tombstone (hash, time)."""
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
            st.delete_job(job_id)
            with db.tx(c):
                c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))       # uploads, events, answers, corrections cascade
                c.execute("INSERT OR REPLACE INTO tombstones (id_hash, deleted_at) VALUES (?, ?)",
                          (hashlib.sha256(job_id.encode()).hexdigest(), db.now()))
        st.delete_job(job_id)                                           # again: a build may have written meanwhile
        return {"deleted": True}

    # ------------------------------------------------------------------ uploads

    def store_frame(job_id: str, v: int, i: int, data: bytes) -> dict:
        from PIL import Image
        if data[:3] != b"\xff\xd8\xff":
            raise ApiError(415, "unsupported_media_type", "Frames must be JPEG images.")
        try:
            with Image.open(io.BytesIO(data)) as im:                     # header only: no full decode here
                w, h, fmt = im.width, im.height, im.format
        except Exception:
            raise ApiError(422, "bad_image", "This frame could not be read as a JPEG image.") from None
        if fmt != "JPEG" or max(w, h) > s.max_frame_long_edge_px or min(w, h) < 240:
            raise ApiError(422, "bad_image", f"Frames must be JPEG, at most {s.max_frame_long_edge_px} px on the long edge and at least 240 px.")
        sha = hashlib.sha256(data).hexdigest()
        key = f"videos/{v}/frames/{i:04d}.jpg"
        with conn() as c:
            with db.tx(c):
                row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise ApiError(404, "not_found", "No such job.")
                state = row["state"]
                old = c.execute("SELECT sha256, size FROM uploads WHERE job_id = ? AND video = ? AND frame_index = ?", (job_id, v, i)).fetchone()
                if old is not None and old["sha256"] == sha:
                    return {"video": v, "index": i, "sha256": sha, "stored": False, "reason": "already uploaded"}
                if state not in ("created", "uploading", "ready_to_build"):
                    raise ApiError(409, "uploads_closed", "Frames can be added only before the build starts.")
                total = c.execute("SELECT COALESCE(SUM(size), 0) FROM uploads WHERE job_id = ?", (job_id,)).fetchone()[0]
                if total - (old["size"] if old else 0) + len(data) > s.max_job_bytes:
                    raise ApiError(413, "job_too_large", f"This part's uploads would exceed {s.max_job_bytes // 2**20} MB.")
                st.put(job_id, key, data)
                c.execute("INSERT INTO uploads (job_id, video, frame_index, sha256, size, created_at) VALUES (?, ?, ?, ?, ?, ?) "
                          "ON CONFLICT(job_id, video, frame_index) DO UPDATE SET sha256 = excluded.sha256, size = excluded.size, "
                          "created_at = excluded.created_at", (job_id, v, i, sha, len(data), db.now()))
                if state in ("created", "ready_to_build"):              # a new frame after the check invalidates it
                    t = db.now()
                    c.execute("UPDATE jobs SET state = 'uploading', updated_at = ?, state_changed_at = ? WHERE id = ?", (t, t, job_id))
                    events.append(c, job_id, "state_changed", {"from": state, "to": "uploading", "reason": "frames received"})
        return {"video": v, "index": i, "sha256": sha, "stored": True}

    @app.put("/v1/jobs/{job_id}/videos/{video}/frames/{index}", openapi_extra={"requestBody": {"required": True, "content": {
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}}}, "description": "One JPEG frame, at most max_frame_bytes."}})
    async def put_frame(request: Request, job_id: str, video: int = P(ge=1, le=3), index: int = P(ge=0, lt=10_000)):
        job_id_ok(job_id)
        if index >= s.max_frames_per_video or video > s.videos_per_job:
            raise ApiError(422, "too_many_frames", f"At most {s.max_frames_per_video} frames per video and {s.videos_per_job} videos.")
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "image/jpeg":
            raise ApiError(415, "unsupported_media_type", "Send each frame as Content-Type: image/jpeg.")
        if int(request.headers.get("content-length") or 0) > s.max_frame_bytes:
            raise ApiError(413, "frame_too_large", f"A frame may be at most {s.max_frame_bytes // 2**20} MB.")
        buf = bytearray()
        async for chunk in request.stream():                            # bounded read, even without or with a false Content-Length
            buf += chunk
            if len(buf) > s.max_frame_bytes:
                raise ApiError(413, "frame_too_large", f"A frame may be at most {s.max_frame_bytes // 2**20} MB.")
        return await run_in_threadpool(store_frame, job_id, video, index, bytes(buf))

    @app.get("/v1/jobs/{job_id}/uploads")
    def get_uploads(job_id: str):
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
            rows = c.execute("SELECT video, frame_index, size FROM uploads WHERE job_id = ? ORDER BY video, frame_index", (job_id,)).fetchall()
        vids = {str(v): [] for v in range(1, s.videos_per_job + 1)}
        for r in rows:
            vids[str(r["video"])].append(r["frame_index"])
        return {"job_id": job_id, "videos": vids, "frames": len(rows), "bytes": sum(r["size"] for r in rows),
                "max_frames_per_video": s.max_frames_per_video}

    # ------------------------------------------------------------------ capture check and start

    @app.post("/v1/jobs/{job_id}/capture-check")
    def run_capture_check(job_id: str):
        with conn() as c:
            row = jobs.get(c, job_id_ok(job_id))
            if row["state"] not in ("uploading", "ready_to_build"):
                raise ApiError(409, "illegal_transition", "Upload frames first; the check runs before the build starts." if row["state"] == "created"
                               else f"This part is {row['state'].replace('_', ' ')}; the capture check is over.")
            jobs.transition(c, job_id, "checking", "capture check started")
            try:
                frames = {}
                for r in c.execute("SELECT video, frame_index FROM uploads WHERE job_id = ?", (job_id,)).fetchall():
                    frames.setdefault(r["video"], {})[r["frame_index"]] = st.get(job_id, f"videos/{r['video']}/frames/{r['frame_index']:04d}.jpg")
                res = capture_check.check(frames, s)
            except Exception:
                jobs.transition(c, job_id, "uploading", "capture check could not run")
                raise
            st.put(job_id, "capture_check.json", json.dumps(res).encode())
            jobs.transition(c, job_id, "ready_to_build", f"capture check: {res['verdict']}")
        return {k: v for k, v in res.items() if k != "videos"} | {
            "videos": {v: {k: x for k, x in d.items() if k != "frame_stats"} for v, d in res["videos"].items()}}

    @app.post("/v1/jobs/{job_id}/start", response_model=models.Job)
    def start(request: Request, job_id: str, body: models.StartRequest | None = None):
        job_id_ok(job_id)
        ip = who(request)
        anyway = bool(body and body.build_anyway)
        with conn() as c:
            row = jobs.get(c, job_id)
            ratelimit.limit(c, f"start:{ip}", s.rate_start_per_hour, what="build starts")
            warning = None
            if row["state"] == "ready_to_build":
                if not st.exists(job_id, "capture_check.json"):
                    raise ApiError(409, "check_needed", "Run the capture check before starting the build.")
                cc = json.loads(st.get(job_id, "capture_check.json"))
                if not cc["usable"]:
                    reasons = " ".join(t for v in cc["videos"].values() for t in v["tips"][:1])
                    if not anyway:
                        raise ApiError(409, "capture_unusable", f"The footage is not good enough to build from yet. {reasons} "
                                       "You can film more, or start anyway.")
                    warning = f"Built although the capture check said the footage is not good enough: {reasons}"
            jobqueue.enqueue(c, job_id, s, row["ip_hash"], warning)
            if anyway:
                c.execute("UPDATE jobs SET build_anyway = 1 WHERE id = ?", (job_id,))
            return job_view(c, job_id)

    # ------------------------------------------------------------------ events: SSE with replay, long-poll fallback

    def fetch(job_id: str, after: int):
        with conn() as c:
            row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return (row["state"] if row else None), events.after(c, job_id, after)

    async def stream(request: Request, job_id: str, after: int):
        last, quiet = after, 0.0
        while True:
            if await request.is_disconnected():
                return
            state, evs = await run_in_threadpool(fetch, job_id, last)
            for e in evs:
                yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e | {'job_id': job_id})}\n\n"
                last, quiet = e["seq"], 0.0
            if state is None:
                yield "event: deleted\ndata: {}\n\n"
                return
            if state in TERMINAL and not evs:
                return
            if quiet >= s.sse_heartbeat_s:
                yield ": heartbeat\n\n"
                quiet = 0.0
            await asyncio.sleep(s.sse_poll_s)
            quiet += s.sse_poll_s

    @app.get("/v1/jobs/{job_id}/events")
    async def get_events(request: Request, job_id: str, after: int = Query(0, ge=0), wait: float = Query(0, ge=0)):
        job_id_ok(job_id)
        state, evs = await run_in_threadpool(fetch, job_id, after)
        if state is None:
            raise ApiError(404, "not_found", "No such job.")
        if "text/event-stream" in request.headers.get("accept", ""):
            last = request.headers.get("last-event-id", "")
            start_at = int(last) if last.isdigit() else after
            return StreamingResponse(stream(request, job_id, start_at), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        t = 0.0
        while not evs and t < min(wait, s.longpoll_max_s) and state not in TERMINAL:   # long-poll: wait for the next event
            await asyncio.sleep(s.sse_poll_s)
            t += s.sse_poll_s
            state, evs = await run_in_threadpool(fetch, job_id, after)
        return {"job_id": job_id, "events": evs, "last_seq": evs[-1]["seq"] if evs else after}

    # ------------------------------------------------------------------ resume codes

    @app.get("/v1/resume/{code}")
    def resume_job(request: Request, code: str = P(max_length=80)):
        """Same answer and similar time for a wrong, malformed or missing code; per-IP lockout after repeated misses."""
        ip = who(request)
        norm = resume.normalise(code)
        h = resume.hash_code(norm or code[:80], pep)                     # always hash: timing does not reveal malformed codes
        miss = ApiError(404, "not_found", "No part matches this code. Check the words and the number.")
        with conn() as c:
            for key in (f"resume_fail:{ip}", f"resume_fail_code:{h[:16]}"):
                left = ratelimit.locked_for(c, key)
                if left > 0:
                    raise ApiError(429, "locked", f"Too many wrong codes. Try again in {int(left) + 1} s.", retry_after_s=int(left) + 1)
            ratelimit.limit(c, f"resume:{ip}", s.rate_resume_per_hour, what="code lookups")
            row = c.execute("SELECT id, state, resume_hash FROM jobs WHERE resume_hash = ?", (h,)).fetchone() if norm else None
            if row is None or not resume.same(row["resume_hash"], h):
                for key in (f"resume_fail:{ip}", f"resume_fail_code:{h[:16]}"):
                    ratelimit.fail(c, key, s.resume_max_fails, s.resume_lock_base_s, s.resume_lock_max_s)
                raise miss
        return {"job_id": row["id"], "state": row["state"]}

    # ------------------------------------------------------------------ Meta model state (truthful, from recorded calls)

    @app.get("/v1/meta/models", response_model=models.MetaModels)
    def meta_models(job_id: str | None = None):
        with conn() as c:
            if job_id is not None:
                jobs.get(c, job_id_ok(job_id))
            st_ = jobs.model_status(c, s.meta_models, s.mode, s.model_running_window_s, job_id)
        return {"mode": s.mode, "job_id": job_id, "models": st_}

    return app


def __getattr__(name):                                                  # `uvicorn backend.app.main:app` builds the app on first use
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
