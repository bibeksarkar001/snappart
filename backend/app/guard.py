"""Request-body guard (pure ASGI, so it sees every byte before FastAPI parses anything): a per-route size cap (frame 3 MB, audio 32 MB,
everything else 64 KB of JSON), checked against Content-Length up front and against the bytes actually received, and an upload stall
timeout (a body that sends nothing for upload_stall_s is cut off with 408, the slow-body half of slowloris). Slow headers and connection
counts are the server's and the reverse proxy's job (docs/security.md)."""
import asyncio
import json
import re

FRAME = re.compile(r"^/v1/jobs/[^/]+/videos/[^/]+/frames/[^/]+$")
VOICE = re.compile(r"^/v1/jobs/[^/]+/voice$")


class Stop(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def cap_for(method: str, path: str, s) -> int:
    if method == "PUT" and FRAME.match(path):
        return s.max_frame_bytes
    if method == "POST" and VOICE.match(path):
        return s.max_audio_bytes + 64 * 1024                            # multipart framing
    return s.max_json_bytes


class BodyGuard:
    def __init__(self, app, settings):
        self.app, self.s = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cap = cap_for(scope["method"], scope["path"], self.s)
        too_big = Stop(413, "payload_too_large", f"This request body is larger than {cap // 1024} KB." if cap < 2 ** 20
                       else f"This request body is larger than {cap // 2 ** 20} MB.")
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length", b"")
        if cl and (not cl.isdigit() or int(cl) > cap):
            return await self.reply(scope, send, too_big if cl.isdigit() else Stop(400, "bad_request", "Bad Content-Length."))
        state = {"seen": 0, "done": False, "hit": None, "started": False}

        async def guarded():
            if state["done"]:                                           # body complete: later receives wait for disconnect (SSE)
                return await receive()
            try:
                msg = await asyncio.wait_for(receive(), self.s.upload_stall_s)
            except asyncio.TimeoutError:
                state["hit"] = Stop(408, "upload_stalled", "The upload stopped sending data. Please try again.")
                raise state["hit"] from None
            if msg["type"] == "http.request":
                state["seen"] += len(msg.get("body", b""))
                if state["seen"] > cap:
                    state["hit"] = too_big
                    raise too_big
                state["done"] = not msg.get("more_body", False)
            else:
                state["done"] = True
            return msg

        async def guarded_send(msg):
            if state["hit"] is not None:                                # the app turned our error into its own response: drop it
                return
            state["started"] = state["started"] or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, guarded, guarded_send)
        except Stop:
            pass
        if state["hit"] is not None and not state["started"]:
            await self.reply(scope, send, state["hit"])

    @staticmethod
    async def reply(scope, send, e: Stop):
        rid = (scope.get("state") or {}).get("request_id")
        body = json.dumps({"error": {"code": e.code, "message": e.message, "retry_after_s": None, "request_id": rid}}).encode()
        await send({"type": "http.response.start", "status": e.status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"connection", b"close"),
            (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]})
        await send({"type": "http.response.body", "body": body})
