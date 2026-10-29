"""
Service mode: the agent's knowledge backend (docs/run-modes.md, `service`).

The agent calls the KB directly, over a Unix socket or a private port, with no credential. It
names the user it acts for in

    X-KB-Principal: base64url(JSON of the task's AuthContext.to_dict())

`ServicePrincipal` decodes that header, caps its scopes (KB_SERVICE_SCOPES_CAP; `admin` is never
accepted unless listed), signs it with the internal secret and passes the request on, so the app
filters every KB and item by that user exactly as in the other modes. There is deliberately no
implicit "service superuser": without the header only the health check and the system info
answer.

The network is the trust boundary: the listener must be reachable by the agent only
(`agenticwf.kb.api.settings.service_bind_allowed`).
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from agenticwf.auth.context import AuthContext

from agenticwf.kb.api.errors import BAD_REQUEST
from agenticwf.kb.api.fronts import headers_of, respond_error, signed_scope, strip_identity_headers
from agenticwf.kb.api.settings import Settings

HEADER = "x-kb-principal"


class PrincipalError(ValueError):
    pass


def encode_principal(ctx: AuthContext) -> str:
    """What the agent sends (and tests use)."""
    raw = json.dumps(ctx.to_dict(), separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def decode_principal(value: str, scopes_cap: frozenset[str]) -> AuthContext:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        data = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise PrincipalError("X-KB-Principal is not base64url JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("subject"), str) or not data["subject"].strip():
        raise PrincipalError("X-KB-Principal has no subject")
    data["scopes"] = sorted(set(data.get("scopes") or ()) & scopes_cap)
    # The agent's user was approved by authsvc already; expiry is the agent's business (its
    # background tasks outlive the user's request).
    data["expires_at"] = None
    try:
        return AuthContext.from_dict(data)
    except (KeyError, TypeError) as exc:
        raise PrincipalError("X-KB-Principal is not an AuthContext") from exc


class ServicePrincipal:
    """ASGI front of service mode: X-KB-Principal in, a signed AuthContext out."""

    def __init__(self, app: Any, settings: Settings):
        self.app = app
        self.settings = settings
        base = settings.base_path
        self.open_paths = {base + "/health"}
        self.optional_paths = {base + "/api/v1/system/info"}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        value = headers_of(scope).get(HEADER)
        scope = strip_identity_headers(scope)   # X-Auth-* and X-KB-Principal: only this front signs
        path = scope["path"]
        if path in self.open_paths or (path in self.optional_paths and not value):
            await self.app(scope, receive, send)
            return
        if not value:
            await respond_error(send, 400, BAD_REQUEST, "principal_required: service mode needs X-KB-Principal")
            return
        try:
            ctx = decode_principal(value, self.settings.service_scopes_cap)
        except PrincipalError as exc:
            await respond_error(send, 400, BAD_REQUEST, f"invalid_principal: {exc}")
            return
        await self.app(signed_scope(scope, ctx, self.settings.context_secret), receive, send)
