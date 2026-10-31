"""FastAPI app. Run: `uvicorn backend.app.main:app` from the repo root (or `python -m backend.app.main`); workers run separately
(`python -m backend.app.worker`). The API holds no state of its own beyond the database and storage, so several replicas can run.

Every response carries X-SnapPart-Mode (mock | real) and X-Request-ID; every error uses the envelope in errors.py. Database and file work
runs in the thread pool, so the event loop (SSE streams, uploads) stays responsive."""
import asyncio
import hashlib
import io
import json
import logging
import re
import secrets
import shutil
import sqlite3
import threading
import warnings
from collections import Counter

import httpx

from fastapi import FastAPI, File, Form, Path as P, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import capture_check, db, events, imagesafe, jobqueue, jobs, meta_client, models, ratelimit, replay, results, resume, retention, spark, voice
from .guard import BodyGuard
from .errors import ApiError, envelope, install
from .settings import Settings, get_settings
from .storage import JOB_ID, LocalStorage

log = logging.getLogger("snappart.api")
TERMINAL = {"failed", "cancelled", "expired"}
UNMETERED = {"/healthz", "/readyz"}
HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
           "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
           "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",   # the API serves no HTML; files set their own
           "Cross-Origin-Opener-Policy": "same-origin", "Permissions-Policy": "camera=(), microphone=(), geolocation=()"}


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
    rec = replay.note(s) if s.replay else None
    if rec:
        print(f"SnapPart API: replay mode (no MODEL_API_KEY set). Serving the recorded part B run of {rec['recorded_at'][:10]} "
              f"({rec['label']}); no Meta model is called.", flush=True)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = s.max_image_pixels                       # Pillow raises on images over twice this (decompression bombs)
    app.add_middleware(BodyGuard, settings=s)                          # inner: runs after the request id and the rate gate

    gate = ratelimit.Gate(s)
    inflight = [0]                                                     # requests being handled by this process (one event loop)

    @app.middleware("http")
    async def frame(request: Request, call_next):
        request.state.request_id = secrets.token_hex(8)                 # server-made; a client-sent id is not trusted
        metered = request.url.path not in UNMETERED and request.method != "OPTIONS"
        counts = metered and not request.url.path.endswith("/events")    # long-polls and streams only sleep on the event loop
        counted = False
        try:
            wait, what = gate.check(who(request)) if metered else (0, "")
            if wait > 0:
                n = max(1, int(wait) + 1)
                resp = envelope(request, 429, "rate_limited", f"Too many {what} from your connection. Try again in {n} s." if what != "requests to the service"
                                else f"The service is very busy. Try again in {n} s.", retry_after_s=n)
            elif counts and inflight[0] >= s.max_inflight:            # shed at once rather than queue work clients will give up on
                resp = envelope(request, 503, "busy", "The service is very busy. Please try again in a moment.", retry_after_s=2)
            else:
                inflight[0] += counts
                counted = counts
                resp = await call_next(request)
        except sqlite3.OperationalError as e:                           # the database stayed locked past busy_timeout: busy, not broken
            if "locked" not in str(e):
                log.exception("unhandled error, request %s", request.state.request_id)
                resp = envelope(request, 500, "internal", "Something went wrong on our side. Please try again.")
            else:
                resp = envelope(request, 503, "busy", "The service is very busy. Please try again in a moment.", retry_after_s=2)
        except Exception:                                               # never leak a trace, path or SQL
            log.exception("unhandled error, request %s", request.state.request_id)
            resp = envelope(request, 500, "internal", "Something went wrong on our side. Please try again.")
        finally:
            inflight[0] -= counted
        resp.headers["X-SnapPart-Mode"] = s.mode
        if rec:
            resp.headers["X-SnapPart-Recorded"] = rec["label"]
        resp.headers["X-Request-ID"] = request.state.request_id
        for k, v in HEADERS.items():
            resp.headers.setdefault(k, v)
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["GET", "POST", "PUT", "DELETE"],
                       allow_headers=["Content-Type", "Last-Event-ID", "CF-Turnstile-Response"], expose_headers=["Retry-After", "X-Request-ID",
                       "X-SnapPart-Mode", "X-SnapPart-Draft", "Content-Disposition"], max_age=600)   # outermost: errors carry CORS headers too

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
        if rec:
            out["recorded"] = rec
        return out

    def disk_ok():
        if shutil.disk_usage(s.data_dir).free < s.min_free_bytes:
            raise ApiError(503, "storage_full", "The service is out of storage space right now. Please try again later.", retry_after_s=600)

    def turnstile_ok(request: Request):
        """Optional bot check on new parts (Cloudflare Turnstile), only when a secret is configured."""
        token = request.headers.get("cf-turnstile-response", "")
        if not token or len(token) > 2048:
            raise ApiError(403, "challenge_failed", "Please complete the check on the page, then try again.")
        try:
            with httpx.Client(timeout=5.0, transport=getattr(app.state, "turnstile_transport", None)) as h:
                ok = h.post(s.turnstile_url, data={"secret": s.turnstile_secret.get_secret_value(), "response": token}).json().get("success") is True
        except (httpx.HTTPError, ValueError):
            ok = False
        if not ok:
            raise ApiError(403, "challenge_failed", "Please complete the check on the page, then try again.")

    def meta(job_id: str | None):
        """The Meta client for real mode with a key, else None (callers then use their deterministic fallback)."""
        if s.mock or s.replay or s.model_api_key is None:
            return None
        return meta_client.MetaClient(s, job_id)

    def unavailable_reason() -> str:
        return "mock mode: no Meta model is called" if s.mock else f"replay mode: {rec['label']}, no Meta model is called now" if rec \
            else "no Meta API key is configured"

    def no_edits():
        if rec and not replay.editable(s):
            raise ApiError(409, "replay_read_only", replay.EDITS_OFF)

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
        disk_ok()
        if s.turnstile_secret is not None:
            turnstile_ok(request)
        code = resume.new_code()
        with conn() as c:
            ratelimit.limit(c, f"create:{ip}", s.rate_create_per_hour, what="new parts")
            ratelimit.limit(c, f"jobs_day:{ip}", s.daily_jobs_per_ip / 24, burst=s.daily_jobs_per_ip, what="new parts today")
            ratelimit.limit(c, "jobs_day:all", s.daily_jobs_global / 24, burst=s.daily_jobs_global, what="new parts on the service today")
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
            retention.purge(c, st, job_id)
        return {"deleted": True}

    # ------------------------------------------------------------------ uploads

    def store_frame(job_id: str, v: int, i: int, data: bytes) -> dict:
        from PIL import Image
        if data[:3] != b"\xff\xd8\xff":
            raise ApiError(415, "unsupported_media_type", "Frames must be JPEG images.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as im:                 # header only: no full decode here
                    w, h, fmt = im.width, im.height, im.format
            data = imagesafe.strip_metadata(data)                       # no EXIF / GPS is ever stored
        except Exception:
            raise ApiError(422, "bad_image", "This frame could not be read as a JPEG image.") from None
        if fmt != "JPEG" or max(w, h) > s.max_frame_long_edge_px or min(w, h) < 240:
            raise ApiError(422, "bad_image", f"Frames must be JPEG, at most {s.max_frame_long_edge_px} px on the long edge and at least 240 px.")
        disk_ok()
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
                cl_ = meta(job_id)
                if cl_ is not None and s.spark_second_opinion:          # optional: the deterministic result stands on its own
                    pick = [(v, fr[sorted(fr)[len(fr) // 2]]) for v, fr in sorted(frames.items()) if fr][:3]
                    pick += [(v, fr[sorted(fr)[0]]) for v, fr in sorted(frames.items()) if fr][:4 - len(pick)]
                    res["second_opinion"] = spark.capture_opinion(cl_, pick)
                else:
                    res["second_opinion"] = {"status": "unavailable", "reason": unavailable_reason()}
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
            if st.exists(job_id, "more_footage.json"):                  # after "film more": the new frames must show the asked-for area
                cov = results.footage_covers(st, job_id, c)
                if not cov["ok"]:
                    raise ApiError(409, "footage_missing_area", cov["message"])
                if not cov["verified"]:
                    warning = ((warning + " ") if warning else "") + ("The new footage could not be checked against the requested area "
                                                                     "(only video 1 can be); it is used as filmed.")
                st.put(job_id, "more_footage_used.json", st.get(job_id, "more_footage.json"))
                st.delete(job_id, "more_footage.json")
            jobqueue.enqueue(c, job_id, s, row["ip_hash"], warning)
            if anyway:
                c.execute("UPDATE jobs SET build_anyway = 1 WHERE id = ?", (job_id,))
            return job_view(c, job_id)

    # ------------------------------------------------------------------ events: SSE with replay, long-poll fallback

    def fetch(job_id: str, after: int):
        with conn() as c:
            row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return (row["state"] if row else None), events.after(c, job_id, after)

    streams: Counter = Counter()                                        # open event streams per job and per client, in this process
    app.state.streams = streams

    def stream_slot(job_id: str, ip: str):
        if streams[job_id] >= s.sse_per_job or streams[ip] >= s.sse_per_ip:
            raise ApiError(429, "too_many_streams", "Too many open progress streams. Close another tab, or try again shortly.", retry_after_s=5)
        streams[job_id] += 1
        streams[ip] += 1
        done = []

        def release():
            if not done:
                done.append(1)
                for k in (job_id, ip):
                    streams[k] -= 1
                    if streams[k] <= 0:
                        del streams[k]
        return release

    async def stream(request: Request, job_id: str, after: int, release):
        try:
            async for chunk in stream_body(request, job_id, after):
                yield chunk
        finally:
            release()

    async def stream_body(request: Request, job_id: str, after: int):
        last, quiet, age = after, 0.0, 0.0
        while True:
            if await request.is_disconnected() or age >= s.sse_max_s:     # the client reconnects with Last-Event-ID
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
            age += s.sse_poll_s

    @app.get("/v1/jobs/{job_id}/events")
    async def get_events(request: Request, job_id: str, after: int = Query(0, ge=0), wait: float = Query(0, ge=0)):
        job_id_ok(job_id)
        state, evs = await run_in_threadpool(fetch, job_id, after)
        if state is None:
            raise ApiError(404, "not_found", "No such job.")
        if "text/event-stream" in request.headers.get("accept", ""):
            last = request.headers.get("last-event-id", "")
            start_at = int(last) if last.isdigit() else after
            release = stream_slot(job_id, who(request))
            from starlette.background import BackgroundTask
            return StreamingResponse(stream(request, job_id, start_at, release), media_type="text/event-stream", background=BackgroundTask(release),
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
        miss = ApiError(404, "not_found", "No part matches this code. Check the words and the number.")

        def locked(c, key):
            left = ratelimit.locked_for(c, key)
            if left > 0:
                raise ApiError(429, "locked", f"Too many wrong codes. Try again in {int(left) + 1} s.", retry_after_s=int(left) + 1)
        with conn() as c:                                               # per-IP checks first: a locked-out guesser costs no scrypt
            locked(c, f"resume_fail:{ip}")
            ratelimit.limit(c, f"resume:{ip}", s.rate_resume_per_hour, what="code lookups")
        h = resume.hash_code(norm or code[:80], pep)                     # always hash: timing does not reveal malformed codes
        with conn() as c:
            locked(c, f"resume_fail_code:{h[:16]}")
            row = c.execute("SELECT id, state, resume_hash FROM jobs WHERE resume_hash = ?", (h,)).fetchone() if norm else None
            if row is None or not resume.same(row["resume_hash"], h):
                for key in (f"resume_fail:{ip}", f"resume_fail_code:{h[:16]}"):
                    ratelimit.fail(c, key, s.resume_max_fails, s.resume_lock_base_s, s.resume_lock_max_s)
                raise miss
        return {"job_id": row["id"], "state": row["state"]}

    # ------------------------------------------------------------------ questions, answers, corrections, rebuild, review

    EDITABLE = ("needs_input", "reviewing", "done")
    rebuild_slots = threading.BoundedSemaphore(s.max_concurrent_rebuilds)

    def editable(c, job_id: str) -> dict:
        row = jobs.get(c, job_id_ok(job_id))
        if row["state"] not in EDITABLE:
            raise ApiError(409, "not_ready", "Answers and changes are possible once the build has finished.")
        return row

    def touched(c, job_id: str):
        """A change after the part was marked done reopens it for review."""
        if jobs.get(c, job_id)["state"] == "done":
            jobs.transition(c, job_id, "reviewing", "changed after it was done")

    @app.get("/v1/jobs/{job_id}/report")
    def get_report(job_id: str):
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
        return results.current_report(st, job_id)

    @app.get("/v1/jobs/{job_id}/questions")
    def get_questions(job_id: str):
        with conn() as c:
            editable(c, job_id)
            ans = results.stored_answers(c, job_id)
            items = results.review_items(c, st, job_id)
        qs = results.questions_view(st, job_id, ans)
        wording = {q["id"]: q.get("friendly") and {**q["friendly"], "recorded": rec["label"]} for q in replay.questions(s)["questions"]} if rec \
            else friendly_wording(job_id, qs)
        owner = (replay.answers(s) or {}).get("answers", {}) if rec else {}
        for q in qs:                                                    # replay: what the owner answered in the recording ("Use this answer")
            if q["id"] in owner:
                q["recorded_answer"] = {"value": owner[q["id"]], "status": "user_supplied", "recorded": rec["label"]}
        for q in qs:
            q["friendly"] = wording.get(q["id"])
        with conn() as c:
            cov = results.coverage_items(c, st, job_id)
        return {"job_id": job_id, "questions": qs, "open_questions": sum(not q["answered"] for q in qs), "review_items": items,
                "coverage_requests": cov, **({"recorded": rec} if rec else {})}

    def friendly_wording(job_id: str, qs: list) -> dict:
        """Muse Spark's wording per question, made once per job and cached (outputs/wording.json); the original text otherwise."""
        key = "outputs/wording.json"
        cache = json.loads(st.get(job_id, key)) if st.exists(job_id, key) else {}
        todo = [q for q in qs if q["id"] not in cache]
        cl_ = meta(job_id)
        if todo and cl_ is not None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(6, len(todo))) as ex:
                for q, w in zip(todo, ex.map(lambda q: spark.word_question(cl_, q), todo)):
                    cache[q["id"]] = w
            st.put(job_id, key, json.dumps(cache).encode())
        elif todo:
            return {q["id"]: {"title": q["title"], "detail": q["detail"], "hint": q["hint"], "source": "fallback",
                              "reason": unavailable_reason()} for q in qs}
        return cache

    @app.post("/v1/jobs/{job_id}/answers")
    def post_answers(job_id: str, body: models.AnswersRequest):
        no_edits()
        with conn() as c:
            editable(c, job_id)
            out = results.save_answers(c, st, job_id, body.answers)
            touched(c, job_id)
        return out

    @app.get("/v1/jobs/{job_id}/corrections")
    def get_corrections(job_id: str):
        with conn() as c:
            editable(c, job_id)
            return {"job_id": job_id, "corrections": results.correction_rows(c, job_id, include_reverted=True)}

    @app.post("/v1/jobs/{job_id}/corrections", status_code=201)
    def post_correction(job_id: str, body: models.CorrectionRequest):
        no_edits()
        if len(json.dumps(body.correction)) > s.max_json_bytes:
            raise ApiError(413, "payload_too_large", "This correction is too large.")
        with conn() as c:
            editable(c, job_id)
            cid = results.add_correction(c, st, job_id, body.correction)
            touched(c, job_id)
        return {"correction_id": cid, "rebuild_needed": True}

    @app.delete("/v1/jobs/{job_id}/corrections/{correction_id}")
    def undo_correction(job_id: str, correction_id: str = P(pattern=r"^c_[0-9a-f]{12}$")):
        no_edits()
        with conn() as c:
            editable(c, job_id)
            results.revert_correction(c, job_id, correction_id)
            touched(c, job_id)
        return {"undone": correction_id, "rebuild_needed": True}

    def replay_rebuild(job_id: str) -> dict:
        """Replay: rebuild from the bundled model (no model call) and label the report so."""
        if not rebuild_slots.acquire(blocking=False):
            raise ApiError(503, "busy", "Several rebuilds are running. Please try again in a moment.", retry_after_s=2)
        try:
            with conn() as c:
                out = results.rebuild(c, st, job_id)
            r_ = results.current_report(st, job_id)
            r_.setdefault("rebuild", {})["label"] = replay.OFFLINE
            st.put(job_id, "outputs/report.json", json.dumps(r_, indent=1, default=float).encode())
            return out | {"label": replay.OFFLINE}
        finally:
            rebuild_slots.release()

    @app.post("/v1/jobs/{job_id}/rebuild")
    def post_rebuild(job_id: str):
        no_edits()
        if not rebuild_slots.acquire(blocking=False):
            raise ApiError(503, "busy", "Several rebuilds are running. Please try again in a moment.", retry_after_s=2)
        try:
            with conn() as c:
                editable(c, job_id)
                if st.size(job_id) - c.execute("SELECT COALESCE(SUM(size), 0) FROM uploads WHERE job_id = ?", (job_id,)).fetchone()[0] \
                        > s.max_job_output_bytes:
                    raise ApiError(507, "job_storage_full", "This part has used all its storage. Delete it and start again.")
                out = results.rebuild(c, st, job_id)
                row = jobs.get(c, job_id)
                if row["state"] == "needs_input" and out["open_questions"] == 0:
                    jobs.transition(c, job_id, "reviewing", "all questions answered")
                out["state"] = jobs.get(c, job_id)["state"]
            if rec:                                                     # replay: say where this result came from
                out["label"] = replay.OFFLINE
                r_ = results.current_report(st, job_id)
                r_.setdefault("rebuild", {})["label"] = replay.OFFLINE
                st.put(job_id, "outputs/report.json", json.dumps(r_, indent=1, default=float).encode())
            return out
        finally:
            rebuild_slots.release()

    @app.get("/v1/jobs/{job_id}/review")
    def get_review(job_id: str):
        with conn() as c:
            editable(c, job_id)
            return {"job_id": job_id, "items": results.review_items(c, st, job_id)}

    @app.post("/v1/jobs/{job_id}/review/{item_id}")
    def post_review_choice(job_id: str, body: models.ReviewChoice, item_id: str = P(pattern=r"^[A-Z]\d{2}$")):
        no_edits()
        with conn() as c:
            editable(c, job_id)
            out = results.choose(c, st, job_id, item_id, body.choice)
            touched(c, job_id)
        return out

    @app.post("/v1/jobs/{job_id}/marks", status_code=201)
    def post_mark(job_id: str, body: models.MarkRequest):
        no_edits()
        with conn() as c:
            editable(c, job_id)
            if len(results.user_marks(st, job_id)) >= 20:
                raise ApiError(422, "too_many_marks", "At most 20 marked spots per part.")
            return results.add_mark(st, job_id, body.point, body.radius_mm, body.normal, body.note)

    @app.post("/v1/jobs/{job_id}/accept", response_model=models.Job)
    def accept(job_id: str):
        with conn() as c:
            row = jobs.get(c, job_id_ok(job_id))
            open_q = sum(not q["answered"] for q in results.questions_view(st, job_id, results.stored_answers(c, job_id))) \
                if row["state"] in EDITABLE else 0
            if open_q:
                raise ApiError(409, "questions_open", f"{open_q} question(s) still need an answer (or 'I don't know').")
            q = results.quality_status(st, job_id)
            if row["state"] in EDITABLE and q["status"] == "draft":
                raise ApiError(409, "draft_model", "The model did not pass its quality checks: " + " ".join(q["reasons"]) +
                               " Review it (or answer the open questions) and rebuild.")
            if row["state"] in EDITABLE and results.open_blocking(c, st, job_id):
                raise ApiError(409, "footage_requested", "Part of the model was barely seen. Film it, or choose to skip it, before finishing.")
            if row["state"] == "needs_input":
                jobs.transition(c, job_id, "reviewing", "all questions answered")
            jobs.transition(c, job_id, "done", "accepted by the user", expect="reviewing")
            return job_view(c, job_id)

    @app.post("/v1/jobs/{job_id}/more-footage")
    def more_footage(job_id: str, body: models.MoreFootage):
        """Answer a "film N more seconds of this area" request: film (the job goes back to uploading; add frames after the existing ones,
        then capture-check and start: the build re-carves only with footage that shows the area) or skip (the area stays not seen)."""
        if rec:
            raise ApiError(409, "replay_read_only", replay.FILM_OFF)
        with conn() as c:
            editable(c, job_id)
            item = results.decide_coverage(c, st, job_id, body.request_id, body.action)
            if body.action == "film":
                if jobs.get(c, job_id)["state"] == "done":
                    jobs.transition(c, job_id, "reviewing", "more footage requested")
                jobs.transition(c, job_id, "uploading", f"filming more of an area ({body.request_id})")
            nxt = {str(r["video"]): r["m"] + 1 for r in c.execute("SELECT video, MAX(frame_index) m FROM uploads WHERE job_id = ? GROUP BY video", (job_id,))}
            return {"request": item, "state": jobs.get(c, job_id)["state"], "next_frame_index": nxt}

    # ------------------------------------------------------------------ Muse Spark / Muse Voice: proposals, voice, summary

    def answer_proposal(c, job_id: str, qid: str, text: str) -> dict:
        """What the user said, read as the answer to the open question qid (checked against the part like any typed answer)."""
        q = next((x for x in results.questions_view(st, job_id, results.stored_answers(c, job_id)) if x["id"] == qid), None)
        if q is None or q["answered"]:
            raise ApiError(409, "question_not_open", "That question is not open for this part.")
        from capture import question_cards as Q
        part = results.part_dims(results.base_report(st, job_id))

        def check(v):
            why = Q.check_answer(qid, v, part)
            if why is None and Q.bounds(qid, part) and not (Q.bounds(qid, part)[0] <= float(v) < Q.bounds(qid, part)[1]):
                why = f"it must be between {Q.bounds(qid, part)[0]} and {Q.bounds(qid, part)[1]} mm"
            return why
        return spark.propose_answer(meta(job_id), text, q, check)

    def proposal_for(c, job_id: str, text: str, item_id: str | None) -> dict:
        item = None
        if item_id:
            item = next((x for x in results.review_items(c, st, job_id) if x["id"] == item_id), None)
            if item is None:
                raise ApiError(404, "not_found", "No such review item.")
        feats = results.load_model(st, job_id)["features"]
        return spark.propose(meta(job_id), text, item, list(feats) + ["bump"])

    @app.post("/v1/jobs/{job_id}/proposals")
    def post_proposal(request: Request, job_id: str, body: models.ProposalRequest):
        """User text -> a correction proposal from the engine's ops. Nothing is applied: the user confirms with POST .../corrections."""
        no_edits()
        with conn() as c:
            editable(c, job_id)
            ratelimit.limit(c, f"proposals:{who(request)}", s.rate_proposals_per_hour, what="requests to change the model")
            out = proposal_for(c, job_id, body.text, body.item_id)
            return out | {"label": replay.OFFLINE_PROPOSAL, "recorded": rec} if rec else out

    @app.post("/v1/jobs/{job_id}/voice")
    async def post_voice(request: Request, job_id: str, audio: UploadFile = File(...), item_id: str | None = Form(None, pattern=r"^[A-Z][0-9]{2}$"),
                         question_id: str | None = Form(None, pattern=r"^[a-z0-9_]{1,40}$")):
        """Audio (WAV; other formats only if ffmpeg is installed) -> transcript (Muse Voice Transcribe) -> a proposal: with question_id (a
        question that is open), a proposed answer to it; otherwise a correction proposal. The audio is held in memory only and dropped as
        soon as it is transcribed. Nothing is applied: the user confirms. If transcription is not possible the error says to type instead."""
        job_id_ok(job_id)
        if int(request.headers.get("content-length") or 0) > s.max_audio_bytes + 64 * 1024:
            raise ApiError(413, "payload_too_large", f"The recording is larger than {s.max_audio_bytes // 2**20} MB.")
        data = await audio.read(s.max_audio_bytes + 1)
        await audio.close()

        def work():
            nonlocal data
            with conn() as c:
                editable(c, job_id)
                ratelimit.limit(c, f"voice:{who(request)}", s.rate_voice_per_hour, what="voice messages")
            cl_ = meta(job_id)
            if cl_ is None:
                raise ApiError(503, "voice_unavailable", f"Voice is not available here ({unavailable_reason()}). Please type instead.")
            try:
                wav, dur = voice.to_wav(data, s.max_audio_bytes)
            except voice.AudioError as e:
                raise ApiError(422, "bad_audio", f"{e} You can also type instead.") from None
            finally:
                data = None                                             # drop the upload as early as possible
            try:
                tr = meta_client.run(cl_.transcribe(wav, voice.KEYWORDS))
            except meta_client.BudgetExceeded as e:
                raise ApiError(503, "daily_capacity_used", e.message, retry_after_s=e.retry_after_s) from None
            except meta_client.MetaError:
                raise ApiError(502, "transcription_failed", "The recording could not be transcribed. Please type instead.") from None
            finally:
                del wav
            with conn() as c:
                if not tr["transcript"].strip():
                    prop = {"correction": None, "explanation": "Nothing was heard in the recording.", "action": "none", "text": ""}
                elif question_id:
                    prop = answer_proposal(c, job_id, question_id, tr["transcript"])
                else:
                    prop = proposal_for(c, job_id, tr["transcript"], item_id)
            return {"transcript": tr["transcript"], "audio_seconds": round(dur, 2), "source": meta_client.VOICE, "audio_kept": False, "proposal": prop}
        return await run_in_threadpool(work)

    @app.get("/v1/jobs/{job_id}/summary")
    def get_summary(job_id: str):
        """Plain-language summary of the current report (Muse Spark, numbers checked against the report; a template otherwise)."""
        with conn() as c:
            editable(c, job_id)
        if rec:
            return {"job_id": job_id, **replay.summary(s), "recorded": rec}
        rep = results.current_report(st, job_id)
        return {"job_id": job_id, **spark.summarize(meta(job_id), rep)}

    # ------------------------------------------------------------------ downloads (server-chosen names only)

    FILES = re.compile(r"^(model\.stl|preview\.stl|preview\.glb|preview_before\.glb|report\.json|cards/[a-z0-9_]{1,40}/(card\.png|card\.json|diagram\.svg)"
                       r"|review/[A-Z][0-9]{2}\.png|inside/[1-3]_[0-9]{4}\.jpg)$")
    TYPES = {".stl": "model/stl", ".glb": "model/gltf-binary", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
             ".jpg": "image/jpeg"}

    @app.get("/v1/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str):
        if not FILES.match(path):
            raise ApiError(404, "not_found", "No such file.")
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
        key = "outputs/" + path
        made = None
        if rec and path in ("model.stl", "preview.stl") and not st.exists(job_id, key) and replay.editable(s):
            made = replay_rebuild(job_id)                               # replay: the STLs are made here from the bundled model
        if not st.exists(job_id, key):
            raise ApiError(404, "not_found", "No such file.")
        ext = path[path.rfind("."):]
        draft = path in ("model.stl", "preview.stl", "preview.glb") and results.quality_status(st, job_id)["status"] == "draft"
        name = f"snappart-{job_id[:8]}-{path.replace('/', '-')}"
        if draft:
            name = name[:-len(ext)] + "-DRAFT" + ext                     # a model that failed a quality gate is labelled draft
        disp = "inline" if ext in (".png", ".svg", ".jpg") else "attachment"
        return Response(st.get(job_id, key), media_type=TYPES[ext], headers={
            "Content-Disposition": f'{disp}; filename="{name}"', "Cache-Control": "no-store", "X-SnapPart-Draft": "true" if draft else "false",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            **({"X-SnapPart-Label": replay.OFFLINE} if made else {})})

    @app.get("/v1/recorded/calipers")
    def recorded_calipers():
        """The recorded part B run's sizes against the owner's calipers (any mode: the bundle is part of the repository)."""
        t_ = replay.caliper_table(s)
        if t_ is None:
            raise ApiError(404, "not_found", "No recorded run with caliper readings is available.")
        return t_

    @app.get("/v1/jobs/{job_id}/recorded-answers")
    def recorded_answers(job_id: str):
        """Replay only: the owner's answers saved with the recording, labelled user_supplied, so the app can offer "Use this answer"."""
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
        ans = replay.answers(s) if rec else None
        if ans is None:
            raise ApiError(404, "not_found", "Recorded answers exist only in replay mode, for a recording that saved them.")
        return {"job_id": job_id, "answers": {k: {"value": v, "status": "user_supplied"} for k, v in ans["answers"].items()},
                "source": ans["source"], "recorded": rec}

    @app.get("/v1/jobs/{job_id}/inside")
    def get_inside(job_id: str):
        """How this build worked: every uploaded frame as a small JPEG with SAM's accepted outline, the camera pose per frame, per-stage
        timings and this job's Meta call log. Kept with the job (same retention and delete); empty for mock builds."""
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
            calls = [{"timestamp": r["ts"], "model": r["model"], "endpoint": r["endpoint"], "status": r["status_code"], "ok": bool(r["ok"]),
                      "latency_ms": None if r["latency_ms"] is None else round(r["latency_ms"], 1)}
                     for r in c.execute("SELECT ts, model, endpoint, ok, status_code, latency_ms FROM model_calls WHERE job_id = ? ORDER BY id", (job_id,))]
        if not st.exists(job_id, "outputs/inside/frames.json"):
            return {"job_id": job_id, "available": False, "reason": "no real or recorded build for this part yet" if not s.mock
                    else "mock mode: nothing was built from your frames", "frames": [], "poses": None, "timings": None, "meta_calls": calls}
        frames = json.loads(st.get(job_id, "outputs/inside/frames.json"))
        for f in frames:
            f["url"] = f["image"] = f"/v1/jobs/{job_id}/files/{f['file']}" if st.exists(job_id, "outputs/" + f["file"]) else None
        rep = results.current_report(st, job_id)
        poses = json.loads(st.get(job_id, "outputs/inside/poses.json"))
        out = {"job_id": job_id, "available": True, "frames": frames, "poses": poses, "timings": rep.get("timings"),
               "cameras": [{"video": p_["video"], "index": p_["index"], "position": p_["camera_centre_mm"]} for p_ in poses.get("poses", [])],
               "stages": (rep.get("timings") or {}).get("per_stage_s"), "meta_calls": calls}
        if rec:                                                         # the recorded run's calls, each marked recorded (no live call now)
            out.update(meta_calls=[c_ | {"recorded": rec["label"]} for c_ in replay.calls(s)], live_meta_calls=calls, recorded=rec)
        return out

    # ------------------------------------------------------------------ Meta model state (truthful, from recorded calls)

    @app.get("/v1/meta/models", response_model=models.MetaModels)
    def meta_models(job_id: str | None = None):
        with conn() as c:
            if job_id is not None:
                jobs.get(c, job_id_ok(job_id))
            st_ = jobs.model_status(c, s.meta_models, s.mode, s.model_running_window_s, job_id)
        return {"mode": s.mode, "job_id": job_id, "models": st_, **({"recorded": rec} if rec else {})}

    return app


def __getattr__(name):                                                  # `uvicorn backend.app.main:app` builds the app on first use
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, limit_concurrency=2000, timeout_keep_alive=5,
                h11_max_incomplete_event_size=16 * 1024, server_header=False)  # behind the reverse proxy (deploy/nginx.conf)
