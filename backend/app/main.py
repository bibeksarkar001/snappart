"""FastAPI app. Run: `uvicorn backend.app.main:app` from the repo root (or `python -m backend.app.main`).

Every response carries X-SnapPart-Mode (mock | real) and X-Request-ID; every error uses the envelope in errors.py. Stage 1 routes:
health, readiness, create / read job, start (enqueue), events (replay by seq), Meta model state."""
import logging
import secrets

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import db, events, jobs, models
from .errors import ApiError, envelope, install
from .settings import Settings, get_settings
from .storage import JOB_ID, LocalStorage

log = logging.getLogger("snappart.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    with db.session(s.db_path) as c:
        db.migrate(c)
    app = FastAPI(title="SnapPart API", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.settings, app.state.storage = s, LocalStorage(s.data_dir)
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

    @app.post("/v1/jobs", response_model=models.JobCreated, status_code=201)
    def create_job(body: models.JobCreate | None = None):
        with conn() as c:
            jid = jobs.create(c, s.mode)
        return {"job_id": jid, "state": "created", "mode": s.mode, "limits": s.public_limits()}

    @app.get("/v1/jobs/{job_id}", response_model=models.Job)
    def get_job(job_id: str):
        with conn() as c:
            row = jobs.get(c, job_id_ok(job_id))
        return jobs.public(row, app.state.storage.list(job_id, "outputs/"))

    @app.post("/v1/jobs/{job_id}/start", response_model=models.Job)
    def start(job_id: str):
        with conn() as c:
            row = jobs.transition(c, job_id_ok(job_id), "queued", "build requested", expect="ready_to_build")
        return jobs.public(row)

    @app.get("/v1/jobs/{job_id}/events", response_model=models.Events)
    def get_events(job_id: str, after: int = Query(0, ge=0)):
        with conn() as c:
            jobs.get(c, job_id_ok(job_id))
            ev = events.after(c, job_id, after)
        return {"job_id": job_id, "events": ev, "last_seq": ev[-1]["seq"] if ev else after}

    @app.get("/v1/meta/models", response_model=models.MetaModels)
    def meta_models(job_id: str | None = None):
        with conn() as c:
            if job_id is not None:
                jobs.get(c, job_id_ok(job_id))
            st = jobs.model_status(c, s.meta_models, s.mode, s.model_running_window_s, job_id)
        return {"mode": s.mode, "job_id": job_id, "models": st}

    return app


def __getattr__(name):                                                  # `uvicorn backend.app.main:app` builds the app on first use
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
