"""Web search (docs/web-search.md): provider management (OpenKB's /web-search-providers),
and /web-search and /web-fetch for callers such as the agent's web tools. Behind `[features] web`.

`/web-search`, `/web-fetch` and `/web-search-providers/test` are read-only POSTs (`knowledge:read`);
managing providers needs `knowledge:write`. Operators' providers come from kb.cfg and change there.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from kb_api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from kb_api.errors import NOT_FOUND, KBError, ok
from kb_runtime.thread.thread import feature_enabled


def _enabled():
    if not feature_enabled("web"):
        raise KBError(NOT_FOUND, "web search is switched off in this deployment")


router = APIRouter(prefix="/api/v1", dependencies=[Depends(_enabled)])
READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)


@router.get("/web-search/providers")
async def kinds(thread: KBThread = Depends(READ)):
    return ok(thread.web_kinds())


@router.post("/web-search")
async def search(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.web_search(body))


@router.post("/web-fetch")
async def fetch(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.web_fetch(body))


@router.get("/web-search-providers")
async def providers(thread: KBThread = Depends(READ)):
    return ok(await thread.web_providers())


@router.post("/web-search-providers", status_code=201)
async def create(body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.web_create_provider(body), status=201)


@router.post("/web-search-providers/test")
async def test_unsaved(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.web_test_provider(None, body))


@router.get("/web-search-providers/{provider_id}")
async def get(provider_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.web_provider(provider_id))


@router.put("/web-search-providers/{provider_id}")
async def update(provider_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.web_update_provider(provider_id, body))


@router.put("/web-search-providers/{provider_id}/credentials")
async def credentials(provider_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.web_provider_credentials(provider_id, body))


@router.delete("/web-search-providers/{provider_id}")
async def delete(provider_id: str, thread: KBThread = Depends(WRITE)):
    await thread.web_delete_provider(provider_id)
    return ok(None)


@router.post("/web-search-providers/{provider_id}/test")
async def test(provider_id: str, body: dict[str, Any] = Body(default={}), thread: KBThread = Depends(READ)):
    return ok(await thread.web_test_provider(provider_id, body))
