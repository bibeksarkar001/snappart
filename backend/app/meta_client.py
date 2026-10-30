"""The one place that calls Meta models (https://api.meta.ai/v1): sam-3.1 (Responses API, streamed), muse-spark-1.3 (Chat Completions with
a JSON schema), muse-voice-transcribe-1.0 (POST /asr/transcribe, multipart WAV).

Every call: daily budget check (hard stop with a friendly message), per-model concurrency limit, for SAM also the machine-wide slots,
token bucket and circuit breaker (limits.py), timeouts, retries with jittered exponential backoff honouring Retry-After (429 / 500 / 503 /
network errors only, per the API's error-handling docs; never 504 or other 4xx), and one row in model_calls (job, model, endpoint, ok,
latency, status, tokens). The key is sent only in the Authorization header; it never appears in URLs, logs, errors or responses."""
import asyncio
import json
import random
import sqlite3
import time
from datetime import datetime, timezone

import httpx

from . import db, limits, ratelimit

SAM, SPARK, VOICE = "sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0"
RETRYABLE = {429, 500, 503}


class MetaError(Exception):
    """A Meta call failed for good. `message` is safe to show (no key, no URL with secrets)."""

    def __init__(self, model: str, message: str, status: int | None = None):
        super().__init__(message)
        self.model, self.message, self.status = model, message, status


class BudgetExceeded(MetaError):
    def __init__(self, retry_after_s: int):
        super().__init__("*", "Today's free capacity is used up. Please try again tomorrow.")
        self.retry_after_s = retry_after_s


def utc_day(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def spend(conn, model: str, limit: int) -> None:
    """Count one call against today's budget (all models together), or raise BudgetExceeded. Atomic across processes."""
    with db.tx(conn):
        t = db.now()
        day = utc_day(t)
        used = conn.execute("SELECT COALESCE(SUM(calls), 0) FROM budget_ledger WHERE day = ?", (day,)).fetchone()[0]
        if used >= limit:
            midnight = datetime.fromtimestamp(t, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86400
            raise BudgetExceeded(max(1, int(midnight - t)))
        conn.execute("INSERT INTO budget_ledger (day, model, calls) VALUES (?, ?, 1) ON CONFLICT(day, model) DO UPDATE SET calls = calls + 1",
                     (day, model))


class MetaClient:
    def __init__(self, settings, job_id: str | None = None, transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep):
        if settings.model_api_key is None:
            raise MetaError("*", "No Meta API key is configured on this server.")
        self.s, self.job_id, self.sleep = settings, job_id, sleep
        self._key = settings.model_api_key.get_secret_value()
        self._transport = transport
        self._sem: dict[str, asyncio.Semaphore] = {}

    def _http(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.s.meta_base_url, transport=self._transport, timeout=httpx.Timeout(timeout, connect=10.0),
                                 headers={"Authorization": f"Bearer {self._key}"})

    def _record(self, model, endpoint, ok, latency_ms, status, tokens=None):
        sql = "INSERT INTO model_calls (job_id, model, endpoint, ok, latency_ms, status_code, tokens, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        row = (self.job_id, model, endpoint, int(ok), latency_ms, status, tokens, db.now())
        with db.session(self.s.db_path) as c:
            try:
                with db.tx(c):
                    c.execute(sql, row)
            except sqlite3.IntegrityError:                              # the job was deleted meanwhile: keep the global count
                with db.tx(c):
                    c.execute(sql, (None,) + row[1:])

    async def _call(self, model: str, endpoint: str, attempt):
        """Run attempt() (one HTTP exchange, returning (result, status, tokens)) under the budget, limits and retry policy; log each try."""
        with db.session(self.s.db_path) as c:
            spend(c, model, self.s.daily_budget_calls)
        sem = self._sem.setdefault(model, asyncio.Semaphore(self.s.meta_concurrency))
        async with sem:
            for k in range(self.s.meta_retries + 1):
                t0 = time.perf_counter()
                status, wait = None, None
                try:
                    result, status, tokens = await attempt()
                    self._record(model, endpoint, True, (time.perf_counter() - t0) * 1000, status, tokens)
                    return result
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
                    retry = status in RETRYABLE
                    msg = f"{model} returned HTTP {status}"
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    retry, msg = True, f"{model}: network error ({type(e).__name__})"
                except MetaError:
                    self._record(model, endpoint, False, (time.perf_counter() - t0) * 1000, status)
                    raise
                self._record(model, endpoint, False, (time.perf_counter() - t0) * 1000, status)
                if not retry or k == self.s.meta_retries:
                    raise MetaError(model, msg, status)
                await self.sleep(wait if wait is not None else min(20.0, 0.75 * 2 ** k) + random.random())

    # ------------------------------------------------------------ Muse Spark: Chat Completions with a JSON schema

    async def chat_json(self, purpose: str, messages: list, schema: dict, max_tokens: int | None = None) -> tuple[dict, dict]:
        """One structured-output call. Returns (parsed JSON, usage). The output is parsed here; schema validation is the caller's.
        max_tokens covers reasoning plus the answer (docs: reasoning counts against it), so it is generous; reasoning_effort keeps it small."""
        body = {"model": SPARK, "messages": messages, "temperature": self.s.spark_temperature, "max_tokens": max_tokens or self.s.spark_max_tokens,
                "reasoning_effort": self.s.spark_reasoning_effort,
                "response_format": {"type": "json_schema", "json_schema": {"name": purpose, "schema": schema, "strict": True}}}

        async def attempt():
            async with self._http(self.s.meta_timeout_s) as h:
                r = await h.post("/chat/completions", json=body)
                r.raise_for_status()
                j = r.json()
                usage = j.get("usage") or {}
                if (j.get("choices") or [{}])[0].get("finish_reason") == "length":
                    raise MetaError(SPARK, "Muse Spark's answer was cut off (token limit).", r.status_code)
                try:
                    out = json.loads(j["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError, ValueError):
                    raise MetaError(SPARK, "Muse Spark returned an unreadable answer.", r.status_code) from None
                return (out, usage), r.status_code, usage.get("total_tokens")
        return await self._call(SPARK, "/chat/completions", attempt)

    # ------------------------------------------------------------ SAM 3.1: Responses API, streamed, parsed by meta_sam_parser

    async def segment(self, jpeg: bytes, concept: str):
        """Records (boxes and masks with confidence) for one concept on one image, plus image (w, h)."""
        from capture import sam as S
        import openai
        from meta_sam_parser import image_segmentation_format, parse_responses_stream

        def breaker(ok=None):
            with db.session(self.s.db_path) as c:
                b = limits.Breaker(c, "sam", self.s.breaker_failures, self.s.breaker_open_s)
                if ok is None:
                    b.check()                                            # raises limits.Unavailable while paused
                    return ratelimit.take(c, "sam:rps", float(self.s.sam_concurrency), self.s.sam_rps)   # 0, or seconds to wait
                b.record(ok)

        async def attempt():
            try:
                while (w := breaker()) > 0:                              # token bucket: wait on the event loop, not in a thread
                    await asyncio.sleep(w)
            except limits.Unavailable as e:
                raise MetaError(SAM, e.reason, 503) from None
            async with self._http(self.s.meta_timeout_s) as h:
                cl = openai.AsyncOpenAI(base_url=self.s.meta_base_url, api_key=self._key, http_client=h, max_retries=0)
                try:
                    events = await cl.responses.create(**S.request(jpeg, concept), stream=True)
                    parsed = parse_responses_stream(events, image_segmentation_format())
                    async for _ in parsed:
                        pass
                    res = await parsed.final_result()
                except openai.APIStatusError as e:
                    breaker(False)
                    raise httpx.HTTPStatusError("sam", request=e.response.request, response=e.response) from None
                except openai.APIConnectionError as e:
                    breaker(False)
                    raise httpx.TransportError(type(e).__name__) from None
            breaker(True)
            return res, 200, None

        lock_dir = self.s.data_dir / "locks" / "sam"
        while (fd := limits.try_slot(lock_dir, self.s.sam_concurrency)) is None:   # machine-wide SAM slot; wait without blocking a thread
            await asyncio.sleep(0.05)
        try:
            return await self._call(SAM, "/responses", attempt)
        finally:
            limits.release_slot(fd)

    # ------------------------------------------------------------ Muse Voice Transcribe: multipart WAV

    async def transcribe(self, wav: bytes, keywords: list[str] | None = None) -> dict:
        req = {"model": VOICE, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK", **({"keywords": keywords} if keywords else {})}

        async def attempt():
            async with self._http(self.s.meta_timeout_s * 4) as h:
                r = await h.post("/asr/transcribe", files={"request": (None, json.dumps(req), "application/json"),
                                                           "audio": ("audio.wav", wav, "audio/wav")}, headers={"Accept": "application/json"})
                r.raise_for_status()
                j = r.json()
                return {"transcript": j.get("transcript", ""), "audio_ms": j.get("audioDurationMs")}, r.status_code, None
        return await self._call(VOICE, "/asr/transcribe", attempt)


def run(coro):
    """Run a client coroutine from synchronous code (the build subprocess, threads of the API)."""
    return asyncio.run(coro)
