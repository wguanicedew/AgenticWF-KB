"""
KB API settings, read from environment variables (docs/run-modes.md, Configuration summary).

The variables of the authsvc integration use the same names as authsvc in the agenticwf-auth
container (AUTH_CONTEXT_SECRET[_FILE], AUTH_BIND_REQUEST), so one secret file serves the auth
container, the agent and the KB. Everything about knowledge (database, retrieval, models) is in
kb.cfg instead.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from agenticwf.auth.config import read_secret

LOG = logging.getLogger(__name__)

MODES = ("standalone", "authsvc", "service")
TRUE = {"1", "true", "yes", "on"}
DEFAULT_SERVICE_SCOPES_CAP = frozenset({"knowledge:read", "knowledge:write"})


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _scopes(value: str) -> frozenset[str]:
    return frozenset(value.replace(",", " ").split())


def _internal_secret() -> bytes:
    """The key StandaloneAuth / ServicePrincipal sign the internal AuthContext with.

    Standalone mode also signs the browser cookie with it, so it must survive restarts: the
    entrypoint keeps it in secrets/standalone_secret (KB_STANDALONE_SECRET_FILE). Without one, a
    random key per process is used (sign-ins then end with the process)."""
    value = read_secret("KB_STANDALONE_SECRET")
    if value:
        return value.encode()
    return secrets.token_urlsafe(48).encode()


@dataclass(frozen=True)
class Settings:
    # standalone | authsvc | service (docs/run-modes.md)
    mode: str = "standalone"
    # Every route lives under this prefix: Caddy forwards /app/kb/* and the path authsvc signs is
    # the path the KB sees.
    base_path: str = "/kb"
    # authsvc: the HMAC key shared with authsvc. standalone/service: the internal signing key.
    context_secret: bytes = field(default_factory=lambda: secrets.token_urlsafe(48).encode())
    # authsvc: require that a context was issued for this exact method and path.
    bind_request: bool = True
    # Account approval (policy.yaml `required_scopes`), checked again in process. Empty disables it.
    required_scopes: frozenset[str] = frozenset({"app:access"})
    # service: the scopes a forwarded principal may carry at most.
    service_scopes_cap: frozenset[str] = DEFAULT_SERVICE_SCOPES_CAP
    # standalone: the address browsers use, for the printed sign-in link.
    public_url: str = "http://localhost:8766"
    # standalone: the one user (subject `local:<local_user>`) and the file with its token.
    local_user: str = "user"
    token_file: str = field(default_factory=lambda: str(token_path()))

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"KB_AUTH_MODE must be one of {', '.join(MODES)}, not {self.mode!r}")
        if not self.base_path.startswith("/") or self.base_path.endswith("/"):
            raise ValueError("KB_BASE_PATH must start with '/' and not end with one (e.g. /kb)")

    @classmethod
    def from_env(cls, mode: str | None = None) -> "Settings":
        mode = mode or _env("KB_AUTH_MODE", "standalone")
        common = dict(
            mode=mode,
            base_path=_env("KB_BASE_PATH", "/kb"),
            service_scopes_cap=_scopes(_env("KB_SERVICE_SCOPES_CAP", " ".join(sorted(DEFAULT_SERVICE_SCOPES_CAP)))),
            public_url=_env("KB_PUBLIC_URL", "http://localhost:8766").rstrip("/"),
        )
        if mode == "authsvc":
            secret = read_secret("AUTH_CONTEXT_SECRET")
            if not secret:
                raise RuntimeError("authsvc mode needs AUTH_CONTEXT_SECRET or AUTH_CONTEXT_SECRET_FILE")
            if len(secret) < 32:
                raise RuntimeError("AUTH_CONTEXT_SECRET must be at least 32 characters")
            return cls(context_secret=secret.encode(),
                       bind_request=_env("AUTH_BIND_REQUEST", "true").lower() in TRUE,
                       required_scopes=_scopes(_env("KB_REQUIRED_SCOPES", "app:access")), **common)
        if mode == "service":
            # The agent already authorized its user; approval is not checked again here.
            return cls(context_secret=secrets.token_urlsafe(48).encode(), required_scopes=frozenset(), **common)
        return cls(context_secret=_internal_secret(), required_scopes=frozenset({"app:access"}),
                   local_user=_env("KB_LOCAL_USER", "user"), token_file=str(token_path()), **common)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def service_bind_allowed(listen: str) -> bool:
    """A service-mode listener (no authentication) must be loopback or a Unix socket, unless
    KB_SERVICE_ALLOW_REMOTE=true (e.g. a Docker network only the agent joins)."""
    if listen.startswith("unix:") or listen.startswith("/"):
        return True
    host = listen.rsplit(":", 1)[0].strip("[]") if ":" in listen else listen
    return is_loopback(host) or _env("KB_SERVICE_ALLOW_REMOTE", "false").lower() in TRUE


def data_root() -> Path:
    return Path(_env("KB_DATA_ROOT", "/var/lib/agenticwf-kb"))


def token_path() -> Path:
    """Standalone mode's token: KB_TOKEN_FILE, default $KB_DATA_ROOT/secrets/token."""
    return Path(_env("KB_TOKEN_FILE", str(data_root() / "secrets" / "token")))
