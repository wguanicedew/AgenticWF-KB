"""
Helpers for the ASGI fronts of standalone and service mode (kb_api.standalone, kb_api.service).

Both fronts authenticate a request their own way, then sign an AuthContext with the app's
internal secret and pass the request on, so the app behind them always runs the same
`AuthContextMiddleware` as in authsvc mode. A front never trusts identity headers from the
client: it removes every `X-Auth-*` (and `X-KB-Principal`, except where it reads it) first.
"""

from __future__ import annotations

import json
from typing import Any

from authsvc.context import AuthContext, encode_context

from kb_api.errors import error_body

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
SIGNED_TTL = 60   # one request; the AuthContext's own expires_at may end it sooner


def headers_of(scope: dict) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}


def strip_identity_headers(scope: dict, keep: tuple[bytes, ...] = ()) -> dict:
    headers = [(k, v) for k, v in scope["headers"]
               if not (k.lower().startswith(b"x-auth-") or (k.lower() == b"x-kb-principal" and k.lower() not in keep))]
    return dict(scope, headers=headers)


def signed_scope(scope: dict, ctx: AuthContext, secret: bytes) -> dict:
    """The request with an AuthContext signed for exactly this method and path."""
    payload, signature = encode_context(ctx, secret, SIGNED_TTL, scope["method"], scope["path"])
    headers = [*scope["headers"], (b"x-auth-context", payload.encode()), (b"x-auth-context-sig", signature.encode())]
    return dict(scope, headers=headers)


def cookies_of(header: str) -> dict[str, str]:
    """Name → value from a Cookie header; lenient, as other apps on the host set cookies too."""
    found: dict[str, str] = {}
    for part in header.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name and name not in found:
            found[name] = value.strip().strip('"')
    return found


def same_origin(scope: dict) -> bool:
    """A write with a cookie must come from this site: Origin (or Sec-Fetch-Site) says so."""
    headers = headers_of(scope)
    origin = headers.get("origin")
    if origin is None:
        return headers.get("sec-fetch-site", "same-origin") in ("same-origin", "none")
    return origin == f"{scope.get('scheme', 'http')}://{headers.get('host', '')}"


def safe_return_to(value: str, base: str) -> str:
    """A path of this service only (no control characters, no other host)."""
    if any(ord(c) < 0x21 or ord(c) == 0x7f for c in value):
        return base + "/"
    if value.startswith(base + "/") and not value.startswith("//") and "\\" not in value:
        return value
    return base + "/"


async def read_body(receive: Any, limit: int = 64 * 1024) -> bytes:
    chunks, size, more = [], 0, True
    while more:
        message = await receive()
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            break
        chunks.append(chunk)
        more = message.get("more_body", False)
    return b"".join(chunks)


async def respond(send: Any, status: int, body: bytes, headers: list[tuple[bytes, bytes]] | None = None,
                  content_type: bytes = b"text/plain; charset=utf-8") -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type), (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store"), *(headers or [])]})
    await send({"type": "http.response.body", "body": body})


async def respond_json(send: Any, status: int, body: dict, headers: list | None = None) -> None:
    await respond(send, status, json.dumps(body).encode(), headers, b"application/json")


async def respond_error(send: Any, status: int, code: int, message: str, headers: list | None = None) -> None:
    await respond_json(send, status, error_body(code, message), headers)


async def respond_html(send: Any, status: int, page: str, headers: list | None = None) -> None:
    await respond(send, status, page.encode(), [(b"x-frame-options", b"DENY"), *(headers or [])],
                  b"text/html; charset=utf-8")
