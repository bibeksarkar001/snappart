# SnapPart backend: security

What the API defends against, how, where the code is, and which test proves it. Tests are in `backend/tests/` (`test_security.py` unless
named otherwise). Everything here runs in the API itself. `deploy/nginx.conf` repeats the body, time and rate limits as a second layer, but
it has **not been run yet** because there is no nginx or Docker on the dev machine. The stage 7 load test ran uvicorn directly.

## Threats

| Threat | Mitigation | Where in code | Test |
|---|---|---|---|
| Oversized bodies (memory, disk) | Per-route caps, checked against `Content-Length` and against the bytes actually received: frame 3 MB, voice 32 MB, any other body 64 KB. 413 in the error envelope. | `app/guard.py` (`BodyGuard`), `main.put_frame` | `test_body_limits_per_route` |
| Slow bodies (slowloris) | A body that sends nothing for `upload_stall_s` (20 s) gets 408. Slow headers and connection counts are handled by the proxy (`client_header_timeout`, `limit_conn`) and uvicorn (`timeout_keep_alive`, `limit_concurrency`, a 16 KB header cap). | `app/guard.py`, `main.__main__`, `deploy/nginx.conf` | `test_upload_stall_timeout` (proxy and uvicorn parts not tested) |
| Unknown or malformed fields | Pydantic schemas with `extra="forbid"`. Answer keys must match `^[a-z0-9_]{1,40}$` and values are at most 40 characters. Part names are at most 80. | `app/models.py` | `test_body_limits_per_route`, `test_hostile_strings` |
| Too many frames or bytes per job | 3 videos x 60 frames, 250 MB per job, checked in the same transaction as the write | `main.store_frame` | `test_app_stage2.py::test_uploads_idempotent` |
| Wrong file types, archives | `Content-Type: image/jpeg` plus the JPEG magic bytes (`FF D8 FF`) for frames. RIFF/WAVE for voice. Archives, PNGs and anything else get 415. | `main.store_frame`, `app/voice.py` | `test_image_safety`, `test_app_stage4.py::test_voice_conversion` |
| Path traversal | Storage keys are made by the server and validated by regex. Download paths must match a whitelist. Job ids must match `^[A-Za-z0-9_-]{16,64}$`. | `app/storage.py`, `main.get_file` | `test_path_traversal`, `test_app_stage1.py::test_storage_keys` |
| Decompression bombs, huge images | `Image.MAX_IMAGE_PIXELS` is 1920 x 1920. The header is read first, without decoding, and a bomb warning counts as an error. The long edge must be 1920 px or less, so OpenCV never gets more pixels than that. | `main.create_app`, `main.store_frame` | `test_image_safety` (forged 40000 x 40000 header) |
| Location and identity in photos | EXIF, GPS, XMP, IPTC, maker notes and comments are removed losslessly before the frame is stored or hashed. The pixels do not change. | `app/imagesafe.py` | `test_image_safety` |
| Unsafe deserialization | The app never uses pickle. Model fields load with `np.load(..., allow_pickle=False)`. The only pickle in the repo is the marker-scan cache of the owner's own videos (`capture/frames.scan`). The API never calls it. | `capture/model.py`, `capture/frames.py` | `test_no_pickle_on_user_data` |
| Request floods | Every route except health checks shares an in-memory bucket per client per minute, one per client per day, and one for the whole service. These are per API process (see `docs/load-test.md` for the trade-off). Stricter buckets in the database, shared by every process, apply to `POST /jobs` (hourly, daily per client, daily global), `start`, `voice`, `proposals` and `resume`. Overload sheds at once with 503 `busy` (in-flight cap per process; a locked database is also `busy`). Every 429 and 503 sets `Retry-After`. | `app/ratelimit.py` (`Gate`, `limit`), `main.frame` | `test_rate_limits_and_proxies`, `test_overload_sheds_at_once`, `test_app_stage2.py::test_shared_limits` |
| Spoofed client IP | `X-Forwarded-For` is read only when the peer is in `trusted_proxies`. The client is the right-most hop that is not a trusted proxy. | `ratelimit.client_ip` | `test_rate_limits_and_proxies` |
| Resume-code guessing | Codes have about 40 bits of entropy (3 BIP-39 words and 2 digits). They are stored as a peppered scrypt hash. Lookups use a constant-time compare and give the same 404 for wrong, malformed and missing codes. There is a lockout per IP and per code (5 misses, then 60 s doubling up to a day). The per-IP checks run before the hash, so a locked-out guesser costs no scrypt. | `app/resume.py`, `main.resume_job` | `test_app_stage2.py::test_resume_codes_and_lockout`, `test_locked_out_guesser_costs_no_hash` |
| Bots creating parts | Optional Cloudflare Turnstile check on `POST /v1/jobs` when `SNAPPART_TURNSTILE_SECRET` is set. It fails closed if Cloudflare cannot be reached. | `main.turnstile_ok` | `test_turnstile_hook` |
| Too many event streams | Capped per job (4) and per client (8) in each API process. Each stream ends after 30 min and the client resumes with `Last-Event-ID`. | `main.stream_slot` | `test_sse_caps_and_lifetime` |
| Queue flooding | The queue is bounded (503 with `Retry-After`). Each client may have at most 2 queued builds and one active build. | `app/jobqueue.py` | `test_app_stage2.py::test_queue_admission_fairness_and_leases` |
| Runaway builds | Builds run in a separate process with memory and CPU rlimits, plus a watchdog and leases | `app/worker.py`, `app/buildproc.py` | `test_app_stage2.py::test_worker_mock_build_resume_and_watchdog` |
| Disk exhaustion | Free-space floor (`min_free_bytes`): new parts and uploads get 503 below it. Outputs are capped per job (507). Retention sweep. | `main.disk_ok`, `main.post_rebuild`, `app/retention.py` | `test_disk_guard`, `test_retention_sweep_with_fake_clock` |
| Cross-origin abuse | CORS allowlist from settings. `*` is refused at startup. | `main.create_app`, `settings._no_wildcard_cors` | `test_headers_cors_and_json_errors` |
| Browser-side attacks on API responses | HSTS, `nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `CSP default-src 'none'; frame-ancestors 'none'`, `Cache-Control: no-store` on every response. Downloads add a sandboxing CSP and a server-chosen file name. | `main.frame`, `main.get_file` | `test_headers_cors_and_json_errors`, `test_hostile_strings` |
| Information leaks in errors | One JSON envelope for every error. Unexpected exceptions become a generic 500 with a request id. The trace is logged on the server only. | `app/errors.py`, `main.frame` | `test_logs_carry_no_ip`, `test_app_stage1.py::test_error_envelope_never_leaks` |
| Prompt injection (part name, answers, proposals, voice transcripts) | Models see user text only inside `<user_text>` delimiters, and the text is cut to 1000 characters with the delimiters removed. Models answer only through a strict JSON schema that names typed operations. Nothing a model says is executed, used as a path, or applied without the user confirming. Answers are validated against bounds. | `app/spark.py`, `app/results.save_answers` | `test_hostile_strings`, `test_app_stage4.py::test_proposals_are_typed_ops_only` |
| Key leaks | The key comes only from the environment, as a `SecretStr`, and is redacted in logs. A test scans every tracked file and every `*.log` under `backend/` for key-shaped strings and for the configured key itself. It reports file names only. | `app/settings.py`, `app/meta_client.py` | `test_no_key_shaped_strings_in_tracked_files_or_logs`, `test_app_stage4.py::test_retries_logging_and_no_key_leak` |
| Vulnerable dependencies | Pip dependencies are pinned with `==` in `backend/environment.yml` (`test_dependencies_are_pinned`). Licences are in `docs/licences.md`. | `backend/environment.yml`, `scripts/licences.py` | see "Supply chain" below |
| Tracking users | No analytics and no third-party scripts. Logs hold request ids, never IPs. Rate-limit keys use a salted IP hash. | `app/ratelimit.py` | `test_logs_carry_no_ip` |
| Data kept too long | `DELETE` removes the files, every row and the resume code, and leaves only a hash tombstone. The sweep deletes every job 7 days after its last change (owner setting), plus the frames of finished parts if that is enabled. Voice audio is never written to disk. | `app/retention.py`, `app/voice.py` | `test_delete_end_to_end`, `test_retention_sweep_with_fake_clock`, `test_app_stage4.py::test_api_voice_real_mode_with_fake_server` |

## Supply chain

- `pip-audit` 2.10.1 was run on 2026-10-30 against all 152 Python packages in the `snappart` environment (`pip list --format=freeze`, `--no-deps`). Result: **no known vulnerabilities**. It does not cover the 192 native conda packages (openssl, ffmpeg, OpenCASCADE and others). Those need a separate scanner (for example an image scanner once the Docker image is built).
- Licences (`docs/licences.md`, regenerated with `python scripts/licences.py`):
  - `meta-sam-parser` is under Meta's SAM License, not an open-source licence.
  - The environment includes GPL-licensed native packages, notably `ffmpeg` (GPL-2.0-or-later), which comes in through `vtk-io-ffmpeg`. That matters if a Docker image is distributed; the image has not been built yet.

## Not covered (known limits)

- The per-process caps (every-route rate limit, in-flight cap, event streams, rebuild slots) multiply with the number of API processes. The database rate limits do not, because all processes share them.
- Stripping metadata stops at the first scan. Metadata placed after the image data (rare vendor trailers) is kept.
- Slow-header protection depends on the proxy. In the stage 7 load test, uvicorn alone kept all 300 slowloris sockets open (`docs/load-test.md`), and the proxy has not been run yet.
- The key scan can only find key shapes it knows, plus the configured key itself.
