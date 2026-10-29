"""
Standalone mode: a personal KB for one user on their own machine (docs/run-modes.md, `standalone`).

Like the agent's local mode, there is one user and one access token:

    agenticwf-kb token                  print the token (for curl, scripts, a personal agent)
    agenticwf-kb token --rotate         replace it (signs every browser out)
    <public url>/kb/?token=…            the sign-in link printed at start

With KB_LOCAL_HINT (a phrase the user chose, at least 8 characters), the sign-in page also shows
the token to whoever gives it; wrong answers are throttled (after 5 in a row, one try a minute).

`StandaloneAuth` stands in front of the unchanged app:

1. strip every incoming `X-Auth-*` (and `X-KB-Principal`) header;
2. accept the token from `Authorization: Bearer`, or `X-API-Key`, or the
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

from agenticwf.auth.context import AuthContext

from agenticwf.kb.api.errors import FORBIDDEN, NOT_FOUND, UNAUTHORIZED
from agenticwf.kb.api.fronts import (SAFE_METHODS, cookies_of, headers_of, read_body, respond, respond_error, respond_html,
                           respond_json, safe_return_to, same_origin, signed_scope, strip_identity_headers)
from agenticwf.kb.api.settings import Settings

COOKIE_AGE = 30 * 86400
CONTEXT_TTL = 8 * 3600
SCOPES = frozenset({"app:access", "knowledge:read", "knowledge:write", "admin"})
HINT_MIN = 8                    # a shorter KB_LOCAL_HINT is ignored
HINT_TRIES, HINT_WAIT = 5, 60   # wrong hints in a row, then one try per HINT_WAIT seconds


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
        self._hint_failures, self._hint_last = 0, 0.0

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

    def _hint_answer(self, answer: str) -> str:
        """"ok", "wrong" or "wait" (too many wrong answers in a row: one try per HINT_WAIT)."""
        hint, now = local_hint(), time.monotonic()
        if self._hint_failures >= HINT_TRIES and now < self._hint_last + HINT_WAIT:
            return "wait"
        if hint and hmac.compare_digest(answer.encode(), hint.encode()):
            self._hint_failures = 0
            return "ok"
        self._hint_failures, self._hint_last = self._hint_failures + 1, now
        return "wrong"

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
            await respond_html(send, 200, login_page(self.base, return_to))
            return
        if path == "/auth/login" and method == "POST":
            if not same_origin(scope):
                await respond_error(send, 403, FORBIDDEN, "csrf")
                return
            form = parse_qs((await read_body(receive)).decode())
            target = safe_return_to(form.get("return_to", [self.base + "/"])[0], self.base)
            if not self.tokens.valid(form.get("token", [""])[0].strip()):
                await respond_html(send, 401, login_page(self.base, target, error="token"))
                return
            await respond(send, 303, b"", headers=[(b"location", target.encode()), self._set_cookie(scope)])
            return
        if path == "/auth/hint" and method == "POST":
            # The right hint signs this browser in and shows the token (for scripts and other browsers).
            if not same_origin(scope):
                await respond_error(send, 403, FORBIDDEN, "csrf")
                return
            form = parse_qs((await read_body(receive)).decode())
            target = safe_return_to(form.get("return_to", [self.base + "/"])[0], self.base)
            result = self._hint_answer(form.get("hint", [""])[0].strip())
            if result == "ok":
                await respond_html(send, 200, token_page(self.tokens.current(), target), [self._set_cookie(scope)])
            else:
                await respond_html(send, 429 if result == "wait" else 401, login_page(self.base, target, error=result))
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


def local_hint() -> str:
    """KB_LOCAL_HINT (the AgenticWF container passes AGENTICWF_LOCAL_HINT): whoever gives it on the
    sign-in page is shown the token. Off when unset or shorter than HINT_MIN."""
    value = os.environ.get("KB_LOCAL_HINT", "").strip()
    return value if len(value) >= HINT_MIN else ""


def token_help() -> str:
    """How to get the link and the token: KB_TOKEN_HELP when set (the AgenticWF container sets its
    `docker exec` commands; one per line, `#` lines explain), else this command's own."""
    lines = [line.strip() for line in os.environ.get("KB_TOKEN_HELP", "").splitlines() if line.strip()] \
        or ["# On this machine (the sign-in link was printed when it started):", "agenticwf-kb token"]
    return "\n".join(f'<p class="h">{html.escape(line.lstrip("# "))}</p>' if line.startswith("#")
                     else f"<pre>{html.escape(line)}</pre>" for line in lines)


PAGE_STYLE = """body{font:15px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;margin:0}
main{width:min(560px,92vw)} h1{font-size:22px;margin:0 0 6px} p{color:#777;margin:0 0 18px}
p.h{margin:0 0 4px;font-size:13px} pre{margin:0 0 12px;padding:8px 10px;border:1px solid #8884;border-radius:8px;
font-size:12.5px;white-space:pre-wrap;word-break:break-all;user-select:all}
input{width:100%;box-sizing:border-box;padding:10px 12px;border:1px solid #8886;border-radius:10px;font:inherit}
button,a.btn{display:block;box-sizing:border-box;margin-top:12px;width:100%;padding:10px;border:0;border-radius:10px;
background:#2f6fed;color:#fff;font:inherit;text-align:center;text-decoration:none;cursor:pointer}
button.alt{background:none;border:1px solid #8886;color:inherit} form+form,details{margin-top:28px}
summary{cursor:pointer;color:#777;margin-bottom:10px} code{font-size:13px} .err{color:#c62828}"""


def page(body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticWF KB · sign in</title>
<meta name="color-scheme" content="light dark"><style>{PAGE_STYLE}</style></head><body><main>
{body}</main></body></html>"""


def login_page(base: str, return_to: str, error: str | None = None) -> str:
    """error: "token" (a wrong token), "wrong" (a wrong hint) or "wait" (too many wrong hints)."""
    back = f'<input type="hidden" name="return_to" value="{html.escape(return_to)}">'
    hinted = error in ("wrong", "wait")
    token_form = f"""<form method="post" action="{html.escape(base)}/auth/login">
{'<p class="err">That token is not right.</p>' if error == "token" else ""}<input type="password" name="token" \
placeholder="Access token"{"" if hinted else " autofocus"} autocomplete="current-password">
{back}<button>Sign in</button></form>"""
    if not local_hint():
        return page(f"""<h1>AgenticWF knowledge base</h1>
<p>Personal KB. Open its sign-in link, or paste its access token. To get them:</p>
{token_help()}
{token_form}""")
    message = {"wrong": "That hint is not right.",
               "wait": "Too many wrong hints: try again in a minute."}.get(error, "")
    return page(f"""<h1>AgenticWF knowledge base</h1>
<p>Personal KB. Paste its access token, or give your hint to see it.</p>
{token_form}
<form method="post" action="{html.escape(base)}/auth/hint"><p class="h">Lost the token? Your hint shows it.</p>
{f'<p class="err">{message} Get the token from the container instead (below).</p>' if message else ""}\
<input type="password" name="hint" placeholder="Hint"{" autofocus" if hinted else ""} autocomplete="off">
{back}<button class="alt">Show the token</button></form>
<details{" open" if hinted else ""}><summary>Or get the sign-in link and the token from the container</summary>
{token_help()}</details>""")


def token_page(token: str, return_to: str) -> str:
    return page(f"""<h1>AgenticWF knowledge base</h1>
<p>Signed in. Your access token, for scripts and a personal agent (<code>Authorization: Bearer …</code>):</p>
<pre>{html.escape(token)}</pre>
<a class="btn" href="{html.escape(return_to)}">Continue</a>""")
