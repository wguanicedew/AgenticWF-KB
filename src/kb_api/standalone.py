"""
Standalone mode: a personal KB for one user on their own machine (docs/run-modes.md, `standalone`).

Like the agent's local mode, there is one user and one access token:

    agenticwf-kb token                  print the token (for curl, scripts, a personal agent)
    agenticwf-kb token --rotate         replace it (signs every browser out)
    <public url>/kb/?token=…            the sign-in link printed at start

`StandaloneAuth` stands in front of the unchanged app:

1. strip every incoming `X-Auth-*` (and `X-KB-Principal`) header;
2. accept the token from `Authorization: Bearer`, or `X-API-Key` (OpenKB clients), or the
   browser cookie;
3. sign an AuthContext for the local user (`local:<KB_LOCAL_USER>`, every scope) with the internal
   secret and pass the request on, so `AuthContextMiddleware` runs as in authsvc mode.

The token lives in a file (KB_TOKEN_FILE, default secrets/token), read again when it changes, so
a rotation applies without a restart. The cookie holds `HMAC(internal secret, token digest +
expiry)`, never the token: rotating the token ends every browser session. Cookie writes must be
same-origin. Nothing here touches the database.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode

from authsvc.context import AuthContext

from kb_api.errors import FORBIDDEN, NOT_FOUND, UNAUTHORIZED
from kb_api.fronts import (SAFE_METHODS, cookies_of, headers_of, read_body, respond, respond_error, respond_html,
                           respond_json, safe_return_to, same_origin, signed_scope, strip_identity_headers)
from kb_api.settings import Settings

COOKIE_AGE = 30 * 86400
CONTEXT_TTL = 8 * 3600
SCOPES = frozenset({"app:access", "knowledge:read", "knowledge:write", "admin"})


# --- the token -------------------------------------------------------------------------------------
def generate_token() -> str:
    return secrets.token_urlsafe(32)


def ensure_token(path: Path, given: str | None = None, rotate: bool = False) -> tuple[str, bool]:
    """The token in `path`, created (or replaced by `given`, or rotated) as needed.
    Returns (token, changed)."""
    current = path.read_text().strip() if path.is_file() else ""
    if given:
        if len(given) < 16:
            raise ValueError("KB_TOKEN must be at least 16 characters")
        if given == current and not rotate:
            return current, False
        token = given
    elif current and not rotate:
        return current, False
    else:
        token = generate_token()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token + "\n")
    return token, True


class TokenFile:
    """The current token, read again when the file changes (rotation without a restart)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._stamp: tuple[float, int] | None = None
        self._token = ""

    def current(self) -> str:
        try:
            stat = self.path.stat()
        except OSError:
            return ""
        stamp = (stat.st_mtime, stat.st_size)
        with self._lock:
            if stamp != self._stamp:
                self._token = self.path.read_text().strip()
                self._stamp = stamp
            return self._token

    def valid(self, candidate: str | None) -> bool:
        token = self.current()
        return bool(candidate) and bool(token) and hmac.compare_digest(candidate.encode(), token.encode())


# --- the front --------------------------------------------------------------------------------------
class StandaloneAuth:
    """ASGI front of standalone mode: the token in, a signed AuthContext for the local user out."""

    def __init__(self, app: Any, settings: Settings):
        self.app = app
        self.settings = settings
        self.base = settings.base_path
        self.secret = settings.context_secret
        self.tokens = TokenFile(Path(settings.token_file))
        self.subject = f"local:{settings.local_user}"

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":   # no websocket routes; never pass one unauthenticated
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope = strip_identity_headers(scope)
        method, path = scope["method"], scope["path"]
        if path == self.base + "/health":
            await self.app(scope, receive, send)
            return
        if path.startswith(self.base + "/auth/"):
            await self._auth_route(scope, receive, send)
            return

        via = self._credential(scope)
        query = parse_qs(scope.get("query_string", b"").decode())
        if via is None and method == "GET" and self.tokens.valid(query.get("token", [""])[0]):
            # The printed sign-in link: keep a cookie, drop the token from the address bar.
            rest = {k: v for k, v in query.items() if k != "token"}
            target = path + ("?" + urlencode(rest, doseq=True) if rest else "")
            await respond(send, 303, b"", headers=[(b"location", target.encode()), self._set_cookie(scope)])
            return
        if via is None:
            if method == "GET" and not path.startswith(self.base + "/api/"):
                location = f"{self.base}/auth/login?return_to={quote(path)}".encode()
                await respond(send, 302, b"", headers=[(b"location", location)])
            else:
                await respond_error(send, 401, UNAUTHORIZED, "missing or wrong access token",
                                    headers=[(b"www-authenticate", b'Bearer realm="agenticwf-kb"')])
            return
        if via == "cookie" and method not in SAFE_METHODS and not same_origin(scope):
            await respond_error(send, 403, FORBIDDEN, "csrf: cross-origin request with the session cookie")
            return
        await self.app(signed_scope(scope, self.context(via), self.secret), receive, send)

    def context(self, via: str) -> AuthContext:
        return AuthContext(subject=self.subject, user_id=self.subject, scopes=SCOPES,
                           expires_at=int(time.time()) + CONTEXT_TTL,
                           auth_method="session" if via == "cookie" else "bearer")

    # -- credentials -----------------------------------------------------------------------------
    def _credential(self, scope: dict) -> str | None:
        """"bearer" or "cookie" when the request carries the token, else None."""
        headers = headers_of(scope)
        auth = headers.get("authorization", "")
        if auth[:7].lower() == "bearer " and self.tokens.valid(auth[7:].strip()):
            return "bearer"
        if self.tokens.valid(headers.get("x-api-key", "").strip()):
            return "bearer"
        return "cookie" if self._cookie_valid(cookies_of(headers.get("cookie", "")).get(self.cookie_name(scope), "")) \
            else None

    def cookie_name(self, scope: dict) -> str:
        port = (scope.get("server") or ("", 0))[1]
        return f"agenticwf_kb_{port}"   # cookies ignore the port: one name per instance

    def _cookie_sig(self, expires: int) -> str:
        token_digest = hashlib.sha256(self.tokens.current().encode()).hexdigest()
        digest = hmac.new(self.secret, f"{token_digest}.{expires}".encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def _cookie_valid(self, value: str) -> bool:
        expires, _, sig = value.partition(".")
        if not expires.isdigit() or int(expires) < time.time() or not self.tokens.current():
            return False
        return hmac.compare_digest(sig, self._cookie_sig(int(expires)))

    def _set_cookie(self, scope: dict, clear: bool = False) -> tuple[bytes, bytes]:
        secure = "; Secure" if scope.get("scheme") == "https" else ""
        if clear:
            value, age = "", 0
        else:
            expires = int(time.time()) + COOKIE_AGE
            value, age = f"{expires}.{self._cookie_sig(expires)}", COOKIE_AGE
        return (b"set-cookie", f"{self.cookie_name(scope)}={value}; Path={self.base}; Max-Age={age}; HttpOnly; "
                               f"SameSite=Strict{secure}".encode())

    # -- <base>/auth/* (what authsvc serves in a deployment) ---------------------------------------
    async def _auth_route(self, scope: dict, receive: Any, send: Any) -> None:
        method, path = scope["method"], scope["path"][len(self.base):]
        query = parse_qs(scope.get("query_string", b"").decode())
        return_to = safe_return_to(query.get("return_to", [self.base + "/"])[0], self.base)
        if path == "/auth/session" and method == "GET":
            if self._credential(scope):
                body = {"authenticated": True, "approved": True,
                        "user": {"id": self.subject, "name": self.settings.local_user}, "scopes": sorted(SCOPES)}
            else:
                body = {"authenticated": False, "login_url": f"{self.base}/auth/login"}
            await respond_json(send, 200, body)
            return
        if path == "/auth/login" and method == "GET":
            await respond_html(send, 200, login_page(self.base, return_to, error=False))
            return
        if path == "/auth/login" and method == "POST":
            if not same_origin(scope):
                await respond_error(send, 403, FORBIDDEN, "csrf")
                return
            form = parse_qs((await read_body(receive)).decode())
            target = safe_return_to(form.get("return_to", [self.base + "/"])[0], self.base)
            if not self.tokens.valid(form.get("token", [""])[0].strip()):
                await respond_html(send, 401, login_page(self.base, target, error=True))
                return
            await respond(send, 303, b"", headers=[(b"location", target.encode()), self._set_cookie(scope)])
            return
        if path == "/auth/logout" and method == "POST":
            if not same_origin(scope):
                await respond_error(send, 403, FORBIDDEN, "csrf")
                return
            clear = self._set_cookie(scope, clear=True)
            if "application/json" in headers_of(scope).get("accept", ""):
                await respond_json(send, 200, {"logout_url": f"{self.base}/auth/login"}, [clear])
            else:
                await respond(send, 303, b"", headers=[(b"location", f"{self.base}/auth/login".encode()), clear])
            return
        await respond_error(send, 404, NOT_FOUND, "not found")


def login_page(base: str, return_to: str, error: bool) -> str:
    message = '<p class="err">That token is not right.</p>' if error else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticWF KB · sign in</title>
<meta name="color-scheme" content="light dark"><style>
body{{font:15px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;margin:0}}
form{{width:min(380px,90vw)}} h1{{font-size:22px;margin:0 0 6px}} p{{color:#777;margin:0 0 18px}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;border:1px solid #8886;border-radius:10px;font:inherit}}
button{{margin-top:12px;width:100%;padding:10px;border:0;border-radius:10px;background:#2f6fed;color:#fff;font:inherit;cursor:pointer}}
code{{font-size:13px}} .err{{color:#c62828}}</style></head><body>
<form method="post" action="{html.escape(base)}/auth/login"><h1>AgenticWF knowledge base</h1>
<p>Personal KB. Paste the access token: <code>agenticwf-kb token</code>, or open the link printed at start.</p>
{message}<input type="password" name="token" placeholder="Access token" autofocus autocomplete="current-password">
<input type="hidden" name="return_to" value="{html.escape(return_to)}"><button>Sign in</button></form></body></html>"""
