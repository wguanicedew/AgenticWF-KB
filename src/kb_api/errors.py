"""
The response envelope and error codes of the API (docs/api.md, Conventions).

    {"success": true, "data": …}
    {"success": false, "error": {"code": 1003, "message": "…", "details": null}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from kb_core.common.exceptions import DatabaseException
from kb_runtime import errors as runtime_errors

LOG = logging.getLogger(__name__)

BAD_REQUEST, UNAUTHORIZED, FORBIDDEN, NOT_FOUND, CONFLICT, INTERNAL, VALIDATION = 1000, 1001, 1002, 1003, 1005, 1007, 1010
CODE_BY_STATUS = {400: BAD_REQUEST, 401: UNAUTHORIZED, 403: FORBIDDEN, 404: NOT_FOUND, 405: BAD_REQUEST,
                  409: CONFLICT, 413: BAD_REQUEST, 422: VALIDATION, 429: BAD_REQUEST, 500: INTERNAL, 503: INTERNAL}
STATUS_BY_CODE = {BAD_REQUEST: 400, UNAUTHORIZED: 401, FORBIDDEN: 403, NOT_FOUND: 404, CONFLICT: 409,
                  INTERNAL: 500, VALIDATION: 422}


class KBError(Exception):
    """An error answered in the envelope. `status` defaults from the code."""

    def __init__(self, code: int, message: str, details: Any = None, status: int | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details
        self.status = status or STATUS_BY_CODE.get(code, 400)


def ok(data: Any = None, status: int = 200) -> JSONResponse:
    return JSONResponse({"success": True, "data": data}, status_code=status)


def ok_list(items: list, total: int, page: int, page_size: int) -> JSONResponse:
    return JSONResponse({"success": True, "data": items, "total": total, "page": page, "page_size": page_size})


def page_args(page: int, page_size: int) -> tuple[int, int]:
    if page < 1 or not 1 <= page_size <= 100:
        raise KBError(VALIDATION, "page must be ≥ 1 and page_size between 1 and 100")
    return page, page_size


def error_body(code: int, message: str, details: Any = None) -> dict:
    return {"success": False, "error": {"code": code, "message": message, "details": details}}


def error_response(code: int, message: str, details: Any = None, status: int | None = None,
                   headers: dict | None = None) -> JSONResponse:
    return JSONResponse(error_body(code, message, details), status_code=status or STATUS_BY_CODE.get(code, 400),
                        headers=headers)


def install(app: FastAPI) -> None:
    @app.exception_handler(KBError)
    async def _kb_error(request: Request, exc: KBError):
        return error_response(exc.code, exc.message, exc.details, exc.status)

    @app.exception_handler(runtime_errors.RuntimeError_)
    async def _runtime_error(request: Request, exc: runtime_errors.RuntimeError_):
        status = 429 if isinstance(exc, runtime_errors.Busy) else None
        response = error_response(exc.code, exc.message, exc.details, status)
        if isinstance(exc, runtime_errors.Conflict) and isinstance(exc.details, dict) and "id" in exc.details:
            # a duplicate: `data` holds the existing object (OpenKB's wire format)
            import json
            body = json.loads(response.body)
            body["data"] = exc.details
            response = JSONResponse(body, status_code=response.status_code)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "error"
        return error_response(CODE_BY_STATUS.get(exc.status_code, BAD_REQUEST), message, status=exc.status_code,
                              headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        return error_response(VALIDATION, "invalid request", details)

    @app.exception_handler(DatabaseException)
    async def _database_error(request: Request, exc: DatabaseException):
        LOG.error("database error on %s %s: %s", request.method, request.url.path, exc)
        return error_response(INTERNAL, "database error", status=503)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        LOG.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(INTERNAL, "internal error")
