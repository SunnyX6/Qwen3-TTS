from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from runtime.executor import ConflictError, QueueFullError, RequestCanceledError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        message = "; ".join(item["msg"] for item in exc.errors()) or "Invalid request body"
        return JSONResponse({"ok": False, "error": message}, status_code=422)

    @app.exception_handler(KeyError)
    async def key_error(_: Request, exc: KeyError):
        return JSONResponse({"ok": False, "error": f"Missing field: {exc.args[0]}"}, status_code=400)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.exception_handler(QueueFullError)
    async def queue_full_error(_: Request, exc: QueueFullError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)

    @app.exception_handler(ConflictError)
    async def conflict_error(_: Request, exc: ConflictError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)

    @app.exception_handler(RequestCanceledError)
    async def request_canceled(_: Request, exc: RequestCanceledError):
        payload = {
            "ok": False,
            "status": "canceled",
            "error": str(exc),
        }
        request_id = getattr(exc, "request_id", None)
        if request_id:
            payload["requestId"] = request_id
        kind = getattr(exc, "kind", None)
        if kind:
            payload["kind"] = kind
        return JSONResponse(payload, status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found(_: Request, exc: FileNotFoundError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.exception_handler(PermissionError)
    async def permission_error(_: Request, exc: PermissionError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=403)

    @app.exception_handler(Exception)
    async def generic_error(_: Request, exc: Exception):
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)


__all__ = ["register_exception_handlers"]
