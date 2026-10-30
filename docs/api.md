# SnapPart API (v1)

JSON over HTTPS. The machine-readable contract is `docs/openapi.json` (regenerate with `python scripts/export_openapi.py`). Mock mode (`SNAPPART_MOCK=1`) and real mode share this contract. This document covers backend stages 1 and 2. Questions, answers, corrections, rebuild and downloads arrive in stage 3.

## Every response

- `X-SnapPart-Mode: mock | real`. In mock mode no upload is processed and no Meta model is called.
- `X-Request-ID`: quote it when reporting a problem.
- `X-Content-Type-Options: nosniff` and `Cache-Control: no-store`.
- Errors always use this envelope. A stack trace, path or SQL never appears in it.

```json
{"error": {"code": "capture_unusable", "message": "plain words for the user", "retry_after_s": null, "request_id": "8e97714a6d754644"}}
```

When `retry_after_s` is set, the `Retry-After` header carries the same value.

Error codes used so far:

| Code | Status | Meaning |
|---|---|---|
| `not_found` | 404 | Unknown, deleted or malformed job id; also any wrong resume code. |
| `validation_error` | 422 | Bad body, query or path, including unknown JSON fields. |
| `illegal_transition` | 409 | The job's state does not allow this action. |
| `check_needed` | 409 | Start was called before a capture check. |
| `capture_unusable` | 409 | Start was refused because the footage is not usable; `message` gives the tips. |
| `uploads_closed` | 409 | A frame was sent after the build started. |
| `unsupported_media_type` | 415 | The frame is not JPEG, or `Content-Type` is not `image/jpeg`. |
| `bad_image` | 422 | Unreadable JPEG, or its dimensions are out of range. |
| `too_many_frames` | 422 | Frame index or video number beyond the limits. |
| `frame_too_large`, `job_too_large` | 413 | A size limit was exceeded. |
| `rate_limited`, `too_many_builds`, `locked` | 429 | Too many requests, too many active builds, or a resume lockout. |
| `queue_full` | 503 | The build queue is full. |
| `internal` | 500 | Anything unexpected. |

## Job states

`created → uploading → checking → ready_to_build → queued → building → needs_input → reviewing → done`, plus `failed`, `cancelled` and `expired`.

- Every transition is stored as a `state_changed` event with a timestamp.
- A transition the state machine does not allow returns 409 and changes nothing.
- A frame uploaded after the capture check moves the job back to `uploading`, so the check must run again.

## Endpoints

### `POST /v1/jobs` → 201

Body (optional): `{"part_name": "buckle"}` (at most 80 characters, unknown fields rejected).

```json
{"job_id": "sb__GMi77R4BE96WVgUo1BXLwZZzLTri", "resume_code": "main-tray-blind-99", "state": "created", "mode": "mock",
 "limits": {"videos_per_job": 3, "max_frames_per_video": 60, "max_frame_bytes": 3145728, "max_job_bytes": 262144000,
            "max_frame_long_edge_px": 1920, "max_audio_bytes": 33554432}}
```

- `job_id` has 192 random bits.
- `resume_code` is three words from the BIP-39 English list plus two digits. It is shown once; the server keeps only a hash.
- Rate limited per client.

### `PUT /v1/jobs/{job_id}/videos/{video}/frames/{index}`

- **Body:** one JPEG, sent with `Content-Type: image/jpeg`.
- **Videos:** 1 = high orbit, 2 = low orbit, 3 = part flipped. Frame `index` runs from 0 to `max_frames_per_video` − 1.
- **Idempotent:** the same bytes again is a no-op (`"stored": false`). Different bytes replace the frame, but only until the build starts.
- **Checks:** the JPEG magic bytes and the header size (long edge at most `max_frame_long_edge_px`, short edge at least 240 px) are checked without decoding the image.

```json
{"video": 1, "index": 0, "sha256": "94be50ff...", "stored": true}
```

### `GET /v1/jobs/{job_id}/uploads`

Which frame indices exist, so an interrupted browser can resume exactly where it stopped.

```json
{"job_id": "sb__...", "videos": {"1": [0, 1, 2], "2": [0, 1], "3": []}, "frames": 5, "bytes": 1130000, "max_frames_per_video": 60}
```

### `POST /v1/jobs/{job_id}/capture-check`

Deterministic and fast: 0.7 s for 120 frames at 1920 px, measured on synthetic frames on a 10-core laptop.

- **Per frame:** mat markers, sharpness, exposure, an approximate pose, and whether the part box is in view.
- **Per video:** a verdict (`ok`, `weak`, `unusable` or `missing`) and concrete tips.
- **Coverage map:** frame counts per 30° azimuth sector × elevation band.
- **Sides** are named relative to the printed mat: right = +x of the page, far = +y.
- **`second_opinion`** is `null` until the optional Muse Spark check (stage 4).
- The job moves to `ready_to_build` either way. Start is refused while the verdict is `unusable`.

```json
{"usable": false, "verdict": "unusable",
 "videos": {"1": {"role": "high orbit", "verdict": "ok", "frames": 18, "usable_frames": 18, "azimuth_bins_covered": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "median_elevation_deg": 60.0, "tips": []},
            "2": {"role": "low orbit", "verdict": "weak", "frames": 10, "usable_frames": 10, "azimuth_bins_covered": [0, 1, 2, 3, 4, 5, 6], "median_elevation_deg": 25.5,
                  "tips": ["Video 2 (low orbit) is missing the near side of the part (about 150 deg of the orbit): film about 13 more seconds from there."]},
            "3": {"role": "flipped part", "verdict": "missing", "frames": 0, "usable_frames": 0, "azimuth_bins_covered": [], "median_elevation_deg": null,
                  "tips": ["Video 3 (flipped part) is missing. Film it: with the part turned over, all the way round."]}},
 "coverage_map": {"azimuth_bin_deg": 30, "elevation_edges_deg": [0, 20, 35, 50, 65, 90], "counts": [[0, 0, ...], ...]},
 "method": "deterministic: ArUco markers, Laplacian sharpness, exposure, approximate pose (focal 0.75 x long edge)", "second_opinion": null}
```

### `POST /v1/jobs/{job_id}/start`

Body (optional): `{"build_anyway": true}`. This starts a build despite an `unusable` check, and the job keeps a warning saying so.

- **Admission control:**
  - The queue is bounded (`max_queue_len`): when it is full, the response is `503 queue_full` with `Retry-After` and how many builds are ahead.
  - Each client is capped at `max_queued_per_ip` waiting or running builds (`429 too_many_builds`).
  - Starts are rate limited per client.
- **Response:** the job (see `GET`) with `queue_position` and `eta_s`. `eta_s` is the rolling average of measured build times × the builds ahead per build slot. It is `null` until a build has finished on this server.

### `GET /v1/jobs/{job_id}`

```json
{"job_id": "sb__...", "state": "queued", "mode": "mock", "created_at": 1793342319.47, "updated_at": 1793342320.75, "state_changed_at": 1793342320.75,
 "warnings": ["Built although the capture check said ..."], "error": null, "queue_position": 1, "eta_s": null, "outputs": [],
 "uploads": {"frames": 28, "bytes": 6331496},
 "capture_check": {"verdict": "unusable", "videos": {"1": {"verdict": "ok", "usable_frames": 18, "frames": 18, "tips": []}, "...": {}}}}
```

On failure, `error` is `{"code": "build_timeout", "message": "This build took too long and was stopped. ..."}`.

Build error codes:

| Code | Cause |
|---|---|
| `build_timeout` | The watchdog stopped the build. |
| `build_crashed` | The build kept crashing. |
| `build_interrupted` | The worker was lost too often. |
| `real_build_not_available` | Real mode before stage 4. |

`outputs` lists stored result files (`outputs/model.stl`, `outputs/report.json`, `outputs/cards/<question>/card.png`, ...). Downloads arrive in stage 3.

### `GET /v1/jobs/{job_id}/events`

Two ways to follow a job's progress:

- **Server-Sent Events** (`Accept: text/event-stream`). Each event is `id: <seq>`, `event: <type>`, and `data:` holding the JSON event.
  - Reconnecting with `Last-Event-ID` (or `?after=seq`) replays everything after that seq.
  - A `: heartbeat` comment is sent every 15 s.
  - The stream ends when the job is failed, cancelled or expired, and sends `event: deleted` if the job is deleted.
- **Long-poll fallback** (any other `Accept`): `?after=seq&wait=s` (wait at most 25 s) returns as soon as there is a newer event.

```json
{"job_id": "sb__...", "events": [{"seq": 3, "type": "state_changed", "data": {"from": "uploading", "to": "checking", "reason": "capture check started"}, "ts": 1793342320.60}], "last_seq": 3}
```

Event types:

| Type | `data` |
|---|---|
| `state_changed` | `from`, `to`, `reason`, and `error` on failure |
| `stage_started` | `stage`, `index`, `of` |
| `stage_finished` | `stage`, `index`, `of`, `seconds` (mock builds add `"mock": true`) |
| `progress` | `fraction`, from 0 to 1 |
| `warning` | `message` |
| `queue` | `position` |
| `build_finished` | `outputs`, `questions` |

The stage names are the real pipeline's: `marker detection`, `mat self-calibration`, `poses + undistorted crops`, `SAM`, `registration`, `joint carve`, `features`, `apply answers + mesh + STL`.

### `GET /v1/resume/{code}`

Takes the code with spaces or hyphens, in any case. Returns `{"job_id", "state"}`.

- A wrong, malformed or deleted code gives the same 404 envelope, and the server hashes the input either way so timing does not tell them apart.
- After `resume_max_fails` misses (default 5), from one client or for one code, lookups lock for 60 s. The lock doubles with each further miss, up to a day (`429 locked`, with `Retry-After`).

### `DELETE /v1/jobs/{job_id}`

Deletes the frames, outputs, check, events, answers, corrections and the job row. Only an anonymous tombstone remains: a hash of the id and the time. A running build is stopped at its next heartbeat. Returns `{"deleted": true}`.

### `GET /v1/meta/models[?job_id=]`

Truthful model state, computed only from recorded calls.

- `running` is true only when a call to that model succeeded: in that job, or recently (globally).
- In mock mode every model shows `running: false`.

```json
{"mode": "mock", "job_id": null, "models": [{"id": "sam-3.1", "running": false, "reason": "mock mode: no Meta model is called", "calls_ok": 0, "calls_failed": 0, "last_ok_at": null}, "..."]}
```

### `GET /healthz`, `GET /readyz`

`/readyz` returns 200 or 503 with `{"ready", "mode", "mock", "environment", "checks": {"db", "storage"}}`, so mock mode is always visible here.
