"""One error envelope for every failure: {"error": {"code", "message" (plain words), "retry_after_s", "request_id"}}. Stack traces, paths and
SQL never reach a client: unexpected exceptions become a generic 500 and are logged server-side with the request id."""
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("snappart.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after_s: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after_s = status, code, message, retry_after_s


def not_found(what="job"):
    return ApiError(404, "not_found", f"No such {what}.")


def envelope(request: Request | None, status: int, code: str, message: str, retry_after_s: int | None = None) -> JSONResponse:
    rid = getattr(getattr(request, "state", None), "request_id", None)
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s is not None else {}
    return JSONResponse({"error": {"code": code, "message": message, "retry_after_s": retry_after_s, "request_id": rid}}, status, headers=headers)


HTTP_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed", 409: "conflict",
              413: "payload_too_large", 415: "unsupported_media_type", 429: "rate_limited", 503: "unavailable"}


def install(app):
    @app.exception_handler(ApiError)
    async def _api(request: Request, e: ApiError):
        return envelope(request, e.status, e.code, e.message, e.retry_after_s)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ())[1:]) or "request"
        return envelope(request, 422, "validation_error", f"Invalid {where}: {first.get('msg', 'check the request')}.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException):
        code = HTTP_CODES.get(e.status_code, "error")
        msg = {404: "Not found.", 405: "This method is not allowed here."}.get(e.status_code, "The request could not be handled.")
        return envelope(request, e.status_code, code, msg)
