# SnapPart API (v1)

JSON over HTTPS. The machine-readable contract is `docs/openapi.json` (regenerate with `python scripts/export_openapi.py`). Mock mode (`SNAPPART_MOCK=1`) and real mode share this contract, so the frontend can be wired now against mock mode. This document covers backend stages 1 to 5.

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
| `frame_too_large`, `job_too_large`, `payload_too_large` | 413 | A size limit was exceeded (frame 3 MB, voice 32 MB, any other body 64 KB). |
| `bad_request` | 400 | A malformed `Content-Length`. |
| `upload_stalled` | 408 | The request body stopped arriving for `upload_stall_s` (20 s). |
| `challenge_failed` | 403 | Only when Turnstile is configured: `POST /v1/jobs` without a valid `CF-Turnstile-Response` header. |
| `rate_limited`, `too_many_builds`, `locked`, `too_many_streams` | 429 | Too many requests (every route except health checks is metered per client, per day and for the whole service), too many active builds, a resume lockout, or too many open event streams. `Retry-After` is always set. |
| `queue_full`, `storage_full`, `busy` | 503 | The build queue is full, the server's disk is below its free-space floor, or the server is overloaded right now (`busy`: retry after `Retry-After`). |
| `job_storage_full` | 507 | A part's outputs reached `max_job_output_bytes`; delete it and start again. |
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
- **`second_opinion`:** in real mode, Muse Spark looks at up to 4 frames and returns `{"status": "ok", "source": "muse-spark-1.3", "frames": [{"video", "index", "mat_visible", "part_fully_visible", "blurry", "glare", "hand_in_frame"}], "overall": "looks_fine" | "has_problems", "tip"}`. If it is down, in mock mode, or its answer does not validate, this is `{"status": "unavailable", "reason"}`, and the deterministic verdict stands on its own either way.
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
| `real_build_not_available` | Real mode on a server where the real build is not switched on. |

`outputs` lists stored result files (`outputs/model.stl`, `outputs/report.json`, `outputs/cards/<question>/card.png`, ...). Download them with `GET /v1/jobs/{job_id}/files/{path}`.

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

## After the build (stage 3)

These routes work once the job is `needs_input`, `reviewing` or `done`. Before that they return `409 not_ready`. Any answer, correction or review choice on a `done` job moves it back to `reviewing`.

### `GET /v1/jobs/{job_id}/questions`

Returns the questions the camera cannot answer, plus the review items. Each question has:

- **`kind`:** `number`, `choice` or `conflict`.
  - A `conflict` question also lists `candidates`: the video's value and the earlier answer.
- **What it is about:** `feature`, which the 3D viewer highlights (`grooves`, `lock_bump`, `plate`), and `label`.
- **Text:** `title`, plus `detail` (built only from measured report values) and `hint`.
- **Bounds:** `min` and `max` in mm, taken from the part's measured size.
- **Answer state:** `default`; `applies_to` (how many slots one answer covers); `answered`, `answer` and `status` (`user_supplied`, or `estimated` after "I don't know").
- **`card`:** links to the "show me where" image, its SVG diagram and its `card.json`.

```json
{"job_id": "xC7m...", "open_questions": 5,
 "questions": [{"id": "groove_depth_mm", "feature": "grooves", "label": "Slots", "kind": "number",
   "title": "How deep are the rectangular slots? (one answer for all of them)",
   "detail": "The 2 openings measured 10.0 x 3.0 mm. A camera cannot see how far down they go.",
   "hint": "Push the depth rod of your calipers into a slot until it touches the bottom.", "unit": "mm", "default": 3.7,
   "min": 0.2, "max": 6.07, "choices": null, "candidates": null, "applies_to": 2, "impact": null,
   "answered": false, "answer": null, "status": null,
   "card": {"image": "/v1/jobs/xC7m.../files/cards/groove_depth_mm/card.png", "diagram": ".../diagram.svg", "meta": ".../card.json"}}],
 "review_items": [{"id": "M01", "kind": "lump", "reason": "MOCK: a canned example of a flagged area, not a real finding",
   "question": "This area looks bumpy. On the real part, is it a flat surface?", "choices": ["Keep", "Remove", "Make flat"],
   "centre": [6.0, 0.0, 6.0], "normal": [0, 0, 1.0], "polygon_xy": [[3.0, -6.0], [9.0, -6.0], [9.0, 6.0], [3.0, 6.0]], "z_mm": [5.9, 6.0, 6.1],
   "area_mm2": 72.0, "feature_id": null, "note": null, "image": "/v1/jobs/xC7m.../files/review/M01.png", "decision": null}]}
```

### `POST /v1/jobs/{job_id}/answers`

Body: `{"answers": {"groove_depth_mm": 2.0, "bump_top_shape": "skip"}}`.

- **Value types:** a number in mm, one of the question's `choices`, or `"skip"` ("I don't know": the default is kept and labelled `estimated`).
- **Validation:** answers are checked against the bounds and the part's measured size, so a slot deeper than the part is tall is rejected with a plain reason (`422 answer_out_of_range`). Unknown question ids return `422 unknown_question`.
- **Rebuild:** saving answers does not rebuild by itself.

```json
{"saved": ["bump_top_shape", "groove_depth_mm"], "open_questions": 3, "rebuild_needed": true}
```

### `POST /v1/jobs/{job_id}/corrections`, `GET .../corrections`, `DELETE .../corrections/{id}`

- **POST** body: `{"correction": {...}}`. It must be exactly one of the corrections engine's five ops, with typed parameters (schema in `backend/capture/corrections.py`): `flatten_region`, `remove_region`, `move_feature_to_opposite_face`, `set_feature_shape`, `add_feature`. Feature and plane names must exist in the model; anything else returns `422 invalid_correction`. The response is `{"correction_id": "c_111755f0efcf", "rebuild_needed": true}`.
- **GET** lists all corrections in order, including undone ones (`"reverted": true`).
- **DELETE** undoes one correction: `{"undone": "c_...", "rebuild_needed": true}`.

### `POST /v1/jobs/{job_id}/rebuild`

Applies the answers and the corrections that are not undone to the stored model, with no AI call. It rewrites `model.stl`, `preview.stl`, `preview.glb` and `report.json`.

- **Speed:** 1.8 s on part B (571k triangles; preview 141k), 0.2 s on the synthetic mock part.
- **Response:** the changes since the previous report, and the new state. The job moves to `reviewing` once no question is open.
- **Concurrency:** at most `max_concurrent_rebuilds` run at once per API process; above that the response is `503 busy` with `Retry-After: 2`.

```json
{"seconds": 0.24, "changes": [{"id": "grooves.depth", "from": 3.7, "to": 2.0, "status_from": "estimated", "status_to": "user_supplied"},
  {"id": "lock_ridge.shape", "from": "hill", "to": "rounded", "status_from": "user_supplied", "status_to": "user_supplied"}],
 "open_questions": 3, "watertight": true, "volume_mm3": 7161.1, "model_triangles": 108192, "preview_triangles": 108192, "state": "needs_input"}
```

### `GET /v1/jobs/{job_id}/report`

The current `report.json` (after the last rebuild), or the build's report if nothing was rebuilt yet. Every number keeps its status label: `measured`, `user_supplied`, `estimated` or `not_detected`.

Fields after a rebuild:

| Field | Meaning |
|---|---|
| `answers` | the answers used |
| `corrections` | the build's corrections plus yours |
| `user_corrections` | yours, with ids |
| `rebuild` | `seconds`, `watertight`, `volume_mm3`, `model_size_mm`, triangle counts |

Mock reports carry `"mode": "mock"` and `"synthetic": true`.

### Review: `GET /v1/jobs/{job_id}/review`, `POST .../review/{item_id}`, `POST .../marks`

Each review item has an image, a region (`centre`, `normal`, `polygon_xy`, `z_mm`) and `choices`. `POST .../review/{item_id}` takes `{"choice": "Make flat"}` and records the decision. What each choice does:

| Choice | Effect |
|---|---|
| Keep, Current side | no edit |
| Remove | cuts the flagged region's height band |
| Make flat | trims what stands above (or below) the region's middle height; it never adds material |
| Opposite side | moves the feature through the part |
| Remove (single-video feature) | removes the feature |

- These areas must face up or down; sideways areas return `422 choice_not_available` for Remove and Make flat.
- Changing your mind undoes the earlier choice's edit.
- The response is `{"item": "M01", "decision": "Make flat", "correction_id": "c_...", "rebuild_needed": true}`.

`POST .../marks` adds your own spot (at most 20 per part): `{"point": [x, y, z], "radius_mm": 3, "normal": [0, 0, 1], "note": "scratch"}`. The point is in the part frame in mm, as the 3D viewer reports it. The spot becomes item `U01`, `U02`, ..., with the same three choices.

### `POST /v1/jobs/{job_id}/accept`

Moves the job to `done` once no question is open (`409 questions_open` otherwise). Returns the job.

### Downloads: `GET /v1/jobs/{job_id}/files/{name}`

Only these names exist. Every other name is a 404, and no user-supplied path reaches the disk.

| Name | Content-Type | Notes |
|---|---|---|
| `model.stl` | `model/stl` | binary, millimetres, full resolution |
| `preview.stl` | `model/stl` | at most 250k triangles (the same model meshed coarser) |
| `preview.glb` | `model/gltf-binary` | the preview for three.js; positions in mm with node scale 0.001 and a rotation from the part's z-up to glTF's y-up |
| `report.json` | `application/json` | |
| `cards/<question>/card.png`, `card.json`, `diagram.svg` | png / json / svg | |
| `review/<item>.png` | `image/png` | |

- **Headers:** `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` and `Content-Security-Policy: default-src 'none'; sandbox`.
- **Content-Disposition:** `attachment` for models and the report, `inline` for images. File names are `snappart-<first 8 of the job id>-<name>`.

## What the approved design needs that the backend cannot provide yet

| Design element | Status |
|---|---|
| Question `impact` ("impact high") | `impact` is `null`. Not computed yet; a measured version would rebuild with the answer changed and report how much the model moves. |
| Plate "conflict" question (video 0.90 mm vs earlier 2.6 mm) | Supported: after an answer that disagrees with the measured separation by more than 0.3 mm, the question becomes `kind: "conflict"` with `candidates` (video, your answer), a `conflict_message` and `status: "conflict"`, and counts as open again. It needs a report that stores the measured separation (real builds); the v4 part B report does not. |
| Review chips "9 measured / n you answered / n estimated / n waiting" | Derive them from `report.json` `features[].status` and the questions' `status`. |
| Result row "Inside the slots: can't be seen" | `report.not_detected` (with reasons), and each depth that only an image saw (`details[].not_detected`). |
| Per-dimension `±` error bars, accuracy estimate at capture | `gates.error_bars` is available for builds that measure held-out folds (real builds, stage 8). On part B the bars are ±0.10 mm and do not cover the caliper errors, so the UI should present them as "at least". There is no accuracy estimate at capture time and no accuracy percentage anywhere. |
| "Fits well / Too tight / Too loose" and the clearance rebuild (−0.3 to +0.4 mm on mating faces) | Not supported: the corrections engine has no clearance op, and the prompt allows only its five ops. Needs a decision. |
| "Tell SnapPart something" box and the mic | `POST .../proposals` and `POST .../voice` (stage 4). |
| Meta model badges | `GET /v1/meta/models?job_id=`, from recorded calls. Real builds (SAM in a job) arrive with stage 8. |
| Capture coverage ring | `capture_check.coverage_map` (azimuth × elevation counts); the ring's drawing is the frontend's. |
| QR link and short link `/p/<code>` | The frontend builds it from the resume code; the backend serves `GET /v1/resume/{code}`. The prototype's "amber river hinge" is not a valid real code (those words are not on the BIP-39 list). |
| GLB of the part | `preview.glb` (preview resolution). |

## Meta models (stage 4)

**Rules for every Meta call:**
- **Running state:** a model counts as running only when a call to it succeeded. Every call is recorded in `model_calls`, and `GET /v1/meta/models` reads those rows.
- **Mock mode** never calls a model, and every endpoint below then uses its deterministic fallback.
- **Daily budget:** all calls share one daily count. When it is used up, endpoints that need a model answer `503 daily_capacity_used`, with the time until midnight UTC in `Retry-After`. Endpoints with a fallback use the fallback and say so.

### Friendlier question wording

Each question from `GET /questions` carries `friendly: {"title", "detail", "hint", "source"}`.

- The wording comes from Muse Spark, made once per job and cached.
- Spark may not introduce any number that is not in the question; an answer that does is rejected.
- Otherwise the original text is used, with `"source": "fallback"` and a `reason`.

### `POST /v1/jobs/{job_id}/proposals`

Body: `{"text": "this part is flat like the bottom", "item_id": "M01"}`. `item_id` is the review item or marked spot the user is looking at, and is optional.

The answer is a **proposal**; nothing is applied. The user confirms by sending its `correction` to `POST .../corrections`.

**How a proposal is made:**
- Muse Spark only picks typed options:
  - `action`: `make_flat`, `remove`, `move_to_opposite_side`, `set_shape` or `none`;
  - the ridge `shape` and `top`;
  - a short `explanation`.
- The region comes from the selected item, so the model never writes geometry, code or paths.
- The correction is built from those choices and validated against the corrections-engine schema.
- **Untrusted text:** the user's text is cut to 1000 characters and placed in a delimited field that the fixed prompt says to treat as data.
- **Fallback:** keyword rules, with `"source": "fallback (keyword rules)"`.

```json
{"correction": {"op": "remove_region", "region": {"polygon_xy": [[149.5, 86.0], [153.0, 86.0], [153.0, 117.0], [149.5, 117.0]], "z_mm": [4.5, 8.5]},
                "source": "review", "review_item": "R01", "reason": "user chose make flat"},
 "explanation": "User confirms the lump area should be flat like the bottom surface.", "action": "make_flat",
 "text": "this part is flat like the bottom", "confirm_with": "POST /v1/jobs/{job_id}/corrections", "source": "muse-spark-1.3"}
```

If the action needs an area and none is selected, the response is `correction: null` with an explanation asking the user to select one. Rate limited per client.

### `POST /v1/jobs/{job_id}/voice`

`multipart/form-data` with `audio` (the recording) and an optional `item_id`. Rate limited per client.

**Audio handling:**
- At most 32 MB and 10 minutes.
- WAV in any PCM format is converted in memory to mono 16-bit 24 kHz. Other formats need `ffmpeg` on the server.
- The audio is transcribed by Muse Voice Transcribe, with part vocabulary as keywords (slot, ridge, plate, depth, flat, ...).
- The audio is dropped at once and never stored.

**Response:**

```json
{"transcript": "please make this area flat, like the bottom of the part.", "audio_seconds": 3.08, "source": "muse-voice-transcribe-1.0",
 "audio_kept": false, "proposal": {"...": "as POST .../proposals"}}
```

**Errors** (each message says "type instead", so the UI can fall back to the text box):

| Code | Status | When |
|---|---|---|
| `bad_audio` | 422 | Unreadable or too short audio |
| `voice_unavailable` | 503 | Mock mode, or no key |
| `transcription_failed` | 502 | The transcription call failed |
| `daily_capacity_used` | 503 | The daily budget is used up |

### `GET /v1/jobs/{job_id}/summary`

`{"job_id", "summary", "source"}`: a plain-language summary of the current report by Muse Spark.

- Every number in it must exist in the report. Otherwise, or if Spark is unavailable, a template summary is used (`"source": "fallback (template)"`).

## Accuracy and completeness (stage 5)

### Per-job defaults

Each question has `default` and `default_source`:
- `measured`: from the video, and at least 1 mm (for example the ridge's height out of its wall);
- `estimated`: a stated estimate, for example slot depth = half the part's height.

No part-specific values live in the generic code. In the report, features keep `status` `measured`, `user_supplied`, `estimated` or `not_detected`.

### New report fields

| Field | Meaning |
|---|---|
| `not_detected` | Recesses: weakly seen areas that 3 or more cameras faced, so more footage would not help. Each has `centre_mm`, `area_mm2` and `reason`. |
| `coverage_requests` | Areas that few cameras faced: `{"id": "film_1", "message": "The camera barely saw the left side of the part. Film about 8 more seconds of it (a low orbit, near table height).", "blocking", "area_mm2", "normal", "cameras_facing", ...}`. |
| `low_confidence_regions.coverage_map` | Every 8 mm cell of the surface with its median number of silhouettes. |
| `checks` | Learnings rules applied to per-video detections (registration gap, location agreement). |
| `details` | Small-detail candidates (holes, slots, bosses, fillets, chamfers, steps, text or marks), each with `size_mm`, `evidence` (views or surface points) and `status` (`measured` or `question`). Image-only openings say their depth is `not_detected`. |
| `quality` | `{"status": "ready" or "draft", "gates": [...], "reasons": [...], "not_run": [...], "roughness": [...]}`. |

### `GET /v1/jobs/{job_id}/questions`

The response also has:
- **`coverage_requests`:** each request with its `decision` (`filming`, `skipped` or `null`).
- **Detail questions** among `review_items`, with ids `D01`, `D02`, ...:
  - choices are `["It is a hole", "Just a mark"]` or `["Text", "Just a mark"]`;
  - the answer is recorded and does not change the geometry.

### `POST /v1/jobs/{job_id}/more-footage`

Body: `{"request_id": "film_1", "action": "film" | "skip"}`.

- **`skip`** accepts the area as not seen.
- **`film`** moves the job back to `uploading` and returns `next_frame_index` per video. Upload new frames from those indices on, run the capture check, then `start`.
  - **The footage is checked:** `start` refuses with `409 footage_missing_area` if no new usable frame faces the requested area. Only video 1's frames can be checked against the part frame; new frames of other videos are accepted with a warning that they were not checked.
  - **Only then is the part re-carved:** the build runs again with all frames.

### Drafts

A model that fails a quality gate is a draft:
- **Downloads:** `model.stl`, `preview.stl` and `preview.glb` get `X-SnapPart-Draft: true` and a file name ending in `-DRAFT`. Otherwise the header is `false`.
- **Accept:** `POST .../accept` refuses with `409 draft_model` (giving the reasons). It also refuses with `409 footage_requested` while a blocking coverage request has no decision.
- **Rebuild:** the response includes `quality: {"status", "reasons"}`.

