# SnapPart backend: load and failure test

**What this test is:** real API processes (uvicorn), real worker processes and a real SQLite database. The build is the **mock pipeline**: canned outputs of a synthetic part, about 3 s per build. Meta calls go to a **fake Meta server** with 0.4 s latency. No real Meta model was called and no real key was used: every process got a fake key, and the fake server counted 0 requests carrying any other key. **10,000 real builds were not run.** A real build is CPU-heavy and is not wired into the worker yet (stage 8).

**Machine:** one MacBook, 10 cores, 16 GB, macOS. The load generator ran on the same machine as the API and the workers, so they competed for CPU.

**Not tested:** Docker and nginx. **The Docker image has not been built** (there is no Docker on this machine), and `deploy/nginx.conf` has not been run. Every number below is uvicorn directly, with no proxy in front.

Script: `python scripts/loadtest.py` (defaults as below). The raw JSON of both runs is in the git-ignored `backend/storage/`, not in the repo.

## Setup

- **Part 1:** 10,000 simulated users arrive evenly over 60 s, each from its own client IP (sent as `X-Forwarded-For` through a trusted proxy address).
  - Each user creates a part, uploads 3 frames (synthetic mat views, about 50 KB each), runs the capture check, starts the build and reads the job. One user in ten also long-polls for events.
  - The API ran 6 uvicorn processes. There were 6 workers, a 3 s mock build, and a 15 s lease (heartbeat 3 s).
  - The client allowed 1,000 open connections and gave up on any request after 60 s, as a user would.
- **Hostile mix, at the same time:**
  - 200 oversized frames (5 MB);
  - 600 malformed JPEGs or archives;
  - 2,000 resume-code guesses from 20 IPs;
  - an SSE storm (300 streams from 10 IPs on 5 jobs, each held up to 10 s);
  - 300 slowloris sockets (headers dripped every 5 s for the whole minute);
  - 50 stalled bodies (headers say 100 KB, 1 KB is sent, then nothing).
- **Worker kill:** at 30 s, the worker holding a build is killed with SIGKILL, together with its build process, as in a crash. A replacement worker starts, as compose's restart policy would. The job must resume from its checkpoint on another worker.
- **Afterwards:** the queue drains. Then every job and frame the API acknowledged is checked against the database and disk (frames by SHA-256), and `PRAGMA integrity_check` runs.
- **Part 2 (fake Meta):** a separate API in real mode, with no workers. 1,000 users over 30 s create a part, upload 4 frames and run the capture check. The capture check asks the fake Muse Spark for a second opinion through the real Meta client.

## Run 1 (before fixes): failed the goal

The goal is that 10,000 requests never break the service: each is accepted, queued or politely refused. Run 1 did not meet it.

| | Result |
|---|---|
| Part 1 load | Took 181 s, not 60. The client finished 18,027 user requests (about 100 per second). |
| `POST /jobs` | 1,318 created, **8,679 timed out at the client's 60 s limit**, 3 returned 500. p50 60.4 s. |
| Other user routes | p95 28 to 58 s. Frames: 3,835 stored, 117 timed out, 2 returned 500. Start: 531 queued, 745 shed with 503 `queue_full`, 2 returned 500. |
| Server errors | **7 responses with status 500**, all `sqlite3.OperationalError: database is locked` after the 30 s busy timeout |
| Builds | **3 queued builds failed** (`build_interrupted` after 3 attempts), the killed job among them. Workers waiting on the locked database missed their leases (6 s in run 1), so their builds were taken from them as lost. |
| Health | 7 of 326 health checks failed. p95 354 ms, max 10.9 s. |
| Integrity | OK. 0 of 1,318 jobs and 0 of 3,835 frames lost or corrupt. |
| Hostile mix | Oversized frames: 192 got 413. Malformed: 600 got 415 or 422. Resume guesses: 1,897 got 429. SSE storm: 177 got 429. Stalled bodies: 50 of 50 got 408. |
| Slowloris | 300 of 300 sockets were still open at the end. Uvicorn has no timeout for slow headers. |

Causes found:

1. **Every request wrote to SQLite**, because the shared rate limiter kept its buckets in the database. That added to each route's own writes. With 6 API processes and 6 workers all needing the single SQLite write lock, some waits exceeded the busy timeout. The same waits made workers miss their lease renewals.
2. **No early shedding.** The API took every request it was sent and queued the work inside, so clients gave up before their turn came.
3. **Resume guesses still cost a 45 ms scrypt hash** after the guessing IP was locked out. That is a CPU amplifier for attackers.
4. (Found in the smoke run before run 1.) **Several API processes starting at once on a fresh database** could fail while switching to WAL. One uvicorn process died at startup.

## Fixes between the runs

| Fix | Where | Test |
|---|---|---|
| The every-route limiter now lives in each process's memory: no database write per request. The stricter limits (create, start, voice, proposals, resume, daily jobs) stay in the database. | `app/ratelimit.py` (`Gate`), `app/main.py` | `test_security.py::test_rate_limits_and_proxies` |
| A cap on in-flight requests per process (`max_inflight`, default 64). Beyond it, the API answers 503 `busy` with `Retry-After: 2` at once. Long-polls and streams do not count, because they only wait. | `app/main.py` | `test_security.py::test_overload_sheds_at_once` |
| A database still locked after the busy timeout gives 503 `busy`, not 500 | `app/main.py` | same |
| SQLite `synchronous=NORMAL` (safe in WAL mode; a power cut can lose the last few commits, but the database is never corrupted) | `app/db.py` | full suite |
| A worker heartbeat that hits a locked database retries on the next beat instead of crashing | `app/worker.py` | not unit-tested; covered by run 2 |
| The per-IP lockout and rate limit are checked before the resume code is hashed | `app/main.py` | `test_security.py::test_locked_out_guesser_costs_no_hash` |
| Switching to WAL is retried while another process holds the lock | `app/db.py` | `test_db_startup.py` |
| Load-test lease raised from 6 s to 15 s (heartbeat 3 s). The production default is still 30 s, heartbeat 10 s. | `scripts/loadtest.py` | |

## Run 2 (after fixes): met the goal on this machine

| | Result |
|---|---|
| Part 1 load | Took 67 s. The client finished 16,198 user requests (about 243 per second). The API was up the whole time. |
| `POST /jobs` | 1,390 created (201). **8,610 shed at once with 503 `busy`** and `Retry-After`. 0 timeouts and 0 server errors. p50 0.4 ms (mostly instant refusals), p95 3.6 s, p99 10.8 s. |
| Frames | 1,068 stored, 3,102 shed with 503. p95 3.4 s, p99 9.7 s. |
| Capture check | 306 ran, 982 shed with 503, 102 got 409. The 409s were parts whose frames had all been shed, so there was nothing to check. p95 7.0 s, p99 14.6 s. |
| Start | 217 queued, 89 shed with 503. p95 8.5 s, p99 15.7 s. |
| Server errors and timeouts | **0** of either, across all user routes |
| Health | 156 of 156 health checks OK. p95 1.2 s, max 3.5 s. |
| Queue and builds | Peak queue 205 (cap 500). Every one of the 217 queued builds finished (`needs_input`). The queue drained in 115 s. |
| Mock build throughput | About 12 builds per worker per minute: a 3 s mock build plus about 2 s of process start, claim and copying. This is **not** a real build rate. |
| Worker kill | The killed job resumed on another worker (attempt 2). Only the interrupted stage ran twice, and the job finished. |
| Integrity | OK. 0 of 1,390 jobs and 0 of 1,068 frames lost or corrupt. Every shed start left its part in `ready_to_build`, so it can be started again. |
| Memory (peak RSS) | API, all 6 processes: 1.7 GB. Workers, all 6: 510 MB. |
| Hostile mix | Oversized frames: 143 got 413 and 56 got 503. Malformed: 14 got 415 or 422 and 586 got 503. Resume guesses: 1,874 got 429, 100 got 404, 26 got 503. Stalled bodies: 25 got 408 after 20 s, 25 got 503. SSE storm: 47 got 429 and 13 streams opened. 240 got 404 because the target parts had never been created: their create requests were shed. |
| Slowloris | **Still not handled:** 300 of 300 sockets were open at the end. This is left to the reverse proxy (`client_header_timeout 10s` in `deploy/nginx.conf`, not yet run). |

**Part 2 (fake Meta), run 2:**
- 563 parts created, 437 shed with 503.
- 413 capture checks ran. Each made a real Meta-client call to the fake Muse Spark, and all 413 calls succeeded (413 rows in `model_calls`).
- 149 capture checks were shed with 503.
- p95: create 4.9 s, capture check 9.8 s.
- In run 1 (before the fixes), 970 calls succeeded, but 30 requests died with connection errors and create p95 was 24.8 s.

## What this means

- **The service holds up under 10,000 users a minute on one laptop.** It stays up and loses nothing, and the work it cannot take is refused at once with 503 and a retry hint, instead of timing out. On this machine it **admitted about 14 %** of new parts. Capacity grows with cores and API processes. Builds are limited by workers: the queue holds 500 and sheds beyond that.
- **Main per-request CPU costs:** the resume-code hash (scrypt, about 45 ms per new part; it releases the GIL, so it runs in parallel) and the capture check (OpenCV on each frame). Both were measured on this machine only.
- **Real build throughput per worker has not been measured through the API**, because real builds are not wired in yet (stage 8). The best measured figure is a single process. Part B took about 3 to 4 minutes per build in the v3/v4 runs, and the golden test's shorter fold took 117 s in the last full suite. For sizing, allow roughly one real build per worker per 3 to 4 minutes, with several cores per worker. Verify this in stage 8.

## Trade-off: the every-route rate limit is per process

The every-route limiter (per client per minute and per day, plus the whole-service cap) now lives in each API process's memory:

- **Effect:** with N API processes, a client can make up to N times the configured rate.
- **Accepted because:** the alternative, a database write on every request, was what broke run 1, and the reverse proxy is the first layer for request rate (`limit_req` in `deploy/nginx.conf`).
- **What stays shared:** the limits that matter most for cost and abuse are still in the database and shared by every process: new parts (hourly and daily, per client and global), build starts, voice, proposals, resume-code lookups and lockouts, and the daily Meta call budget.
- **Memory:** the in-memory buckets are cleared when they pass 500,000 entries, which also resets every client's count.
- **Beyond one machine:** move this limiter to Redis or the proxy.

## Known limits of this test

- The client ran on the same 10 cores as the server, so the server had less CPU than it would alone.
- All traffic used loopback networking: no TLS, no real network delay, no proxy.
- The fake Meta server answers instantly after a fixed 0.4 s. Real latency varies more (the stage 4 live check measured slower SAM calls).
- A per-process cap shed many requests while other processes still had room. uvicorn spreads connections between processes but does not balance them.
