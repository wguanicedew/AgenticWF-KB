"""System routes: health, the caller's identity, system info, the management page.

Every route reaches the runtime through a thread: a KBThread for the principal
(`thread_for`), or the manager's SystemThread where there is no principal (health, and system
info in service mode).
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from agenticwf.auth.context import AuthContext

from agenticwf.kb.api.deps import KBThread, is_approved, manager_of, optional_principal, thread_for
from agenticwf.kb.api.errors import UNAUTHORIZED, KBError, ok

router = APIRouter()


@router.get("/health", include_in_schema=False)
async def health(request: Request):
    """Public: the database answers. Plain JSON (no envelope) for load balancers and probes."""
    if not await manager_of(request).system.healthy():
        return JSONResponse({"status": "unavailable", "database": "down"}, status_code=503)
    return {"status": "ok"}


@router.get("/api/v1/me")
async def me(request: Request, thread: KBThread = Depends(thread_for(allow_pending=True))):
    return ok({**(await thread.me()), "approved": is_approved(thread.auth, request),
               "mode": request.app.state.settings.mode})


@router.get("/api/v1/system/info")
async def system_info(request: Request, ctx: AuthContext | None = Depends(optional_principal)):
    settings = request.app.state.settings
    manager = manager_of(request)
    if ctx is None:
        # Service mode answers without a principal (the agent checks what the KB offers before it
        # acts for anyone); the other modes need an authenticated caller.
        if settings.mode != "service":
            raise KBError(UNAUTHORIZED, "authentication required")
        info = await manager.system.info()
    else:
        async with manager.thread(ctx, source=settings.mode) as thread:
            info = await thread.system_info()
    return ok({**info, "mode": settings.mode, "base_path": settings.base_path})


_PAGE = (Path(__file__).resolve().parent.parent / "web" / "index.html")


@lru_cache(maxsize=1)
def _page_template() -> str:
    return _PAGE.read_text(encoding="utf-8")


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def landing(request: Request, ctx: AuthContext | None = Depends(optional_principal)):
    """The management page (agenticwf/kb/api/web/index.html) for a signed-in caller; it calls the API with
    the caller's own credentials (cookie or session). Without a principal: who to ask."""
    settings = request.app.state.settings
    if ctx is not None:
        page = _page_template().replace("{{BASE}}", html.escape(settings.base_path, quote=True)) \
            .replace("{{MODE}}", html.escape(settings.mode, quote=True))
        return HTMLResponse(page, headers={"Cache-Control": "no-store",
                                           "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline' "
                                           "'self'; style-src 'unsafe-inline' 'self'; img-src 'self' data:; "
                                           "frame-ancestors 'none'"})
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticWF KB</title>
<meta name="color-scheme" content="light dark"><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:640px;
margin:10vh auto;padding:0 16px}} code{{font-size:13px}}</style></head><body>
<h1>AgenticWF knowledge base</h1><p>Not signed in ({settings.mode} mode).</p>
<p>API: <code>{html.escape(settings.base_path)}/api/v1/</code></p></body></html>""")
