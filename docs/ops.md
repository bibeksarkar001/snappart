# SnapPart backend: operations

This file is extended in stage 7 (Docker, compose, deployment sketch).

## Run locally

From the repo root, with the `snappart` conda environment (`backend/environment.yml`):

```bash
SNAPPART_MOCK=1 uvicorn backend.app.main:app --port 8000
```

```bash
SNAPPART_MOCK=1 python -m backend.app.worker
```

- Run the API and each worker as separate processes. Start as many workers as the machine allows.
- `SNAPPART_MOCK=1` serves canned results of a synthetic part, which is enough for frontend work. Every response then says `X-SnapPart-Mode: mock`.
- The first mock build on a fresh data directory takes about 20 s, because it generates the canned bundle once.
- Mock mode refuses to start with `SNAPPART_ENVIRONMENT=production` unless `SNAPPART_ALLOW_MOCK_IN_PRODUCTION=1` is also set.
- Real builds are not wired into the worker yet (planned with the end-to-end run, stage 8), so real-mode jobs fail with `real_build_not_available`. The Meta model calls themselves work (stage 4).
- **Meta API key:** read from `MODEL_API_KEY`, either in the environment or in the repo's `.env`, which settings load with pydantic-settings. It never appears in a response, a log, an error or `repr(settings)`.
- **Model access:** `python scripts/meta_models_check.py` lists the model ids the key can use (it prints ids only).
- **Live check:** `python scripts/meta_live_check.py OUT_DIR` makes real calls (about 40 SAM calls, 6 Spark calls and 1 transcription; small cost) and writes `OUT_DIR/live_check.json`.
- **Voice input:** WAV is converted in memory. Other formats (webm, ogg, mp4 from the browser's recorder) need `ffmpeg` on the PATH; without it they are refused with "please send WAV / type instead".

## Settings (environment, prefix `SNAPPART_`)

List values are given as JSON, for example `SNAPPART_TRUSTED_PROXIES='["10.0.0.2"]'`.

**Owner decisions, with safe defaults:**

| Setting | Default |
|---|---|
| `SNAPPART_RETENTION_DAYS` | 7 |
| `SNAPPART_CORS_ORIGINS` | none |
| `SNAPPART_TRUSTED_PROXIES` | none |

Host, domain, contact email and governing law are not backend settings. They are listed in the launch checklist.

**Storage and mode:**

| Setting | Default | Meaning |
|---|---|---|
| `SNAPPART_DATA_DIR` | `backend/storage` (git-ignored) | Job files under `jobs/{id}/`, and the SQLite database |
| `SNAPPART_MOCK` | 0 | Mock mode |
| `SNAPPART_ENVIRONMENT` | development | `development`, `test` or `production` |

**Uploads:**

| Setting | Default |
|---|---|
| `SNAPPART_MAX_FRAMES_PER_VIDEO` | 60 |
| `SNAPPART_MAX_FRAME_BYTES` | 3 MB |
| `SNAPPART_MAX_JOB_BYTES` | 250 MB |
| `SNAPPART_MAX_FRAME_LONG_EDGE_PX` | 1920 |

**Queue:**

| Setting | Default | Meaning |
|---|---|---|
| `SNAPPART_MAX_QUEUE_LEN` | 500 | |
| `SNAPPART_MAX_CONCURRENT_BUILDS` | CPU cores ÷ 3 | Across all workers |
| `SNAPPART_MAX_QUEUED_PER_IP` | 2 | |
| `SNAPPART_BUILD_TIMEOUT_S` | 720 | Watchdog |
| `SNAPPART_BUILD_MEM_MB` | 12000 | rlimit; not enforced on macOS, where the watchdog still applies |
| `SNAPPART_LEASE_S` | 30 | |
| `SNAPPART_HEARTBEAT_S` | 10 | |
| `SNAPPART_MAX_ATTEMPTS` | 3 | |

**SAM limits:**

| Setting | Default |
|---|---|
| `SNAPPART_SAM_CONCURRENCY` | 16 |
| `SNAPPART_SAM_RPS` | 40 (the API's limit is 3,000 per minute) |
| `SNAPPART_BREAKER_FAILURES` | 5 |
| `SNAPPART_BREAKER_OPEN_S` | 30 |

**Meta models:**

| Setting | Default | Meaning |
|---|---|---|
| `SNAPPART_META_BASE_URL` | `https://api.meta.ai/v1` | |
| `SNAPPART_META_TIMEOUT_S` | 60 | Per request; transcription gets 4x |
| `SNAPPART_META_RETRIES` | 4 | Retries for 429 / 500 / 503 / network errors only; never 504 or other 4xx |
| `SNAPPART_META_CONCURRENCY` | 16 | Per process, per model |
| `SNAPPART_SPARK_REASONING_EFFORT` | low | Muse Spark reasoning counts against the token cap |
| `SNAPPART_SPARK_MAX_TOKENS` | 4000 | Reasoning plus answer |
| `SNAPPART_SPARK_TEMPERATURE` | 0.1 | |
| `SNAPPART_SPARK_SECOND_OPINION` | on | Capture-check second opinion (real mode) |
| `SNAPPART_DAILY_BUDGET_CALLS` | 20000 | All Meta calls per UTC day; then "today's free capacity is used up" |
| `SNAPPART_RATE_VOICE_PER_HOUR` | 30 | Per client |
| `SNAPPART_RATE_PROPOSALS_PER_HOUR` | 120 | Per client |

**Capture check:**

| Setting | Default |
|---|---|
| `SNAPPART_MIN_USABLE_FRAMES` | 12 |
| `SNAPPART_BLUR_MIN_LAPLACIAN_VAR` | 60 |
| `SNAPPART_MIN_COVERAGE_BINS` | 9 of 12 |
| `SNAPPART_ORBIT_DEG_PER_S` | 12 |

**Resume codes and rate limits:**

| Setting | Default | Meaning |
|---|---|---|
| `SNAPPART_RESUME_PEPPER` | generated once into `DATA_DIR/.resume_pepper` | Pepper for resume-code hashes |
| `SNAPPART_RESUME_MAX_FAILS` | 5 | |
| `SNAPPART_RESUME_LOCK_BASE_S` | 60 | |
| `SNAPPART_RESUME_LOCK_MAX_S` | 86400 | |
| `SNAPPART_RATE_CREATE_PER_HOUR` | 20 | Per client |
| `SNAPPART_RATE_START_PER_HOUR` | 20 | Per client |
| `SNAPPART_RATE_RESUME_PER_HOUR` | 60 | Per client |
| `SNAPPART_RATE_ANY_PER_MIN` / `SNAPPART_RATE_ANY_BURST` | 600 / 300 | Every route except health checks, per client |
| `SNAPPART_DAILY_REQUESTS_PER_IP` | 50000 | Per client per day |
| `SNAPPART_DAILY_JOBS_PER_IP` / `SNAPPART_DAILY_JOBS_GLOBAL` | 60 / 20000 | New parts per day |
| `SNAPPART_GLOBAL_RPS` | 1000 | All clients together, per database |
| `SNAPPART_TURNSTILE_SECRET` | none | Set it to require a Cloudflare Turnstile token on `POST /v1/jobs` (owner decision) |

**Resource limits and retention (docs/security.md):**

| Setting | Default | Meaning |
|---|---|---|
| `SNAPPART_UPLOAD_STALL_S` | 20 | A request body that sends nothing this long gets 408 |
| `SNAPPART_SSE_PER_JOB` / `SNAPPART_SSE_PER_IP` | 4 / 8 | Open event streams, per API process |
| `SNAPPART_SSE_MAX_S` | 1800 | A stream closes after this; clients reconnect with `Last-Event-ID` |
| `SNAPPART_MIN_FREE_BYTES` | 2 GiB | Below this free disk, new parts and uploads get 503 |
| `SNAPPART_MAX_JOB_OUTPUT_BYTES` | 300 MiB | Outputs one part may hold |
| `SNAPPART_MAX_IMAGE_PIXELS` | 1920 x 1920 | Pillow's decompression-bomb limit |
| `SNAPPART_RETENTION_DAYS` | 7 | Owner decision. Every job is deleted this long after its last change |
| `SNAPPART_DELETE_FRAMES_WHEN_DONE` | false | Owner decision. Also delete the frames once a part is finished |
| `SNAPPART_SWEEP_EVERY_S` | 3600 | Each worker runs the sweep this often; `python -m app.retention` (from `backend/`) runs it once |

The reverse proxy in `deploy/nginx.conf` repeats the body, time and rate limits as a second layer and terminates TLS. Set
`SNAPPART_TRUSTED_PROXIES` to its address so the API reads the client from `X-Forwarded-For`. The server name and certificate paths there
are owner decisions.

## How it scales

- **API:** stateless apart from the database and storage. Run several replicas behind the reverse proxy, all pointed at the same `DATA_DIR`.
- **Workers:** independent processes that coordinate only through the database. A worker claims a queued job in one write transaction:
  - oldest first;
  - at most one active build per client;
  - at most `MAX_CONCURRENT_BUILDS` builds across all workers.
- **Leases:** the worker renews its lease every `HEARTBEAT_S`. If a worker dies, its lease runs out, and the next worker requeues the job. The job resumes from its last finished stage (checkpoints under `jobs/{id}/work/`). After `MAX_ATTEMPTS`, the job fails with a plain reason.
- **Isolation:** each build runs in its own subprocess, with memory and CPU rlimits and a wall-clock watchdog.
- **SAM limits:** shared by all workers on one machine.
  - The concurrency semaphore uses file locks under `DATA_DIR/locks/`, and the OS releases them if a process dies.
  - The token bucket and circuit breaker live in the database.
- **One machine:** SQLite in WAL mode with short write transactions, and local disk. This is the supported setup today.
- **Beyond one machine:**
  - Replace SQLite with Postgres. The SQL is plain; the queue claim needs `SELECT ... FOR UPDATE SKIP LOCKED`.
  - Replace `LocalStorage` with an S3-compatible store implementing the same six methods (`app/storage.py`).
  - Replace the file-lock SAM semaphore with a database or Redis semaphore.
- **Real build throughput** per worker is measured in stage 7 (`docs/load-test.md`). One real build used about 3 to 4 minutes in the v3/v4 runs.
