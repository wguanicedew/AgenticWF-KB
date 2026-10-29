"""
AuthContext checks and the thread for KB endpoints.

Routes reach the runtime only through a KBThread (docs/architecture.md, Threads):

    @router.get("/api/v1/me")
    async def me(thread: KBThread = Depends(thread_for(allow_pending=True))):
        return ok(thread.me())

`thread_for(...)` checks the scopes, then opens a thread for the request's principal on the
process's KBThreadManager (app.state.manager) and closes it after the response.

In authsvc mode the route policy (deploy/policy.kb.yaml) is the first gate; these FastAPI
dependencies repeat the checks inside the process, so a misconfigured policy cannot expose
knowledge on its own. In standalone and service modes they are the only route checks. Which
KBs and items a principal may touch is decided by agenticwf.kb.runtime.access.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Depends, Request

from agenticwf.auth.client import current_auth_context
from agenticwf.auth.context import AuthContext

from agenticwf.kb.api.errors import FORBIDDEN, UNAUTHORIZED, KBError
from agenticwf.kb.runtime.thread.manager import KBThreadManager
from agenticwf.kb.runtime.thread.thread import KBThread

__all__ = ["AuthContext", "KBThread", "current_auth_context", "get_principal", "optional_principal", "require",
           "thread_for", "manager_of",
           "SCOPE_READ", "SCOPE_WRITE", "SCOPE_ADMIN"]

SCOPE_READ = "knowledge:read"
SCOPE_WRITE = "knowledge:write"
SCOPE_ADMIN = "admin"


def optional_principal() -> AuthContext | None:
    return current_auth_context()


def get_principal() -> AuthContext:
    """The request's AuthContext, from any of the three run modes; 401 (1001) without one."""
    ctx = current_auth_context()
    if ctx is None:
        raise KBError(UNAUTHORIZED, "authentication required")
    return ctx


def is_approved(ctx: AuthContext, request: Request) -> bool:
    return ctx.has_scopes(*request.app.state.settings.required_scopes)


def require(*scopes: str, allow_machine: bool = True, allow_pending: bool = False):
    """FastAPI dependency: an approved principal holding all ``scopes``.

    ``allow_machine=False`` rejects machine-to-machine tokens (authsvc mode);
    ``allow_pending=True`` skips the account-approval scopes."""

    def dependency(request: Request, ctx: AuthContext = Depends(get_principal)) -> AuthContext:
        if not allow_pending and not is_approved(ctx, request):
            raise KBError(FORBIDDEN, "approval_pending")
        if not allow_machine and ctx.is_machine:
            raise KBError(FORBIDDEN, "machine tokens are not allowed here")
        missing = set(scopes) - ctx.scopes
        if missing:
            raise KBError(FORBIDDEN, f"missing scopes: {' '.join(sorted(missing))}")
        return ctx

    return dependency


def manager_of(request: Request) -> KBThreadManager:
    return request.app.state.manager


def thread_for(*scopes: str, allow_machine: bool = True, allow_pending: bool = False):
    """FastAPI dependency: `require(...)`, then a KBThread for the principal, closed afterwards."""
    check = require(*scopes, allow_machine=allow_machine, allow_pending=allow_pending)

    async def dependency(request: Request, ctx: AuthContext = Depends(check)) -> AsyncIterator[KBThread]:
        async with manager_of(request).thread(ctx, source=request.app.state.settings.mode) as thread:
            yield thread

    return dependency
