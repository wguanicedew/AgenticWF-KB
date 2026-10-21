"""Phase 2 routes (docs/api.md, Tags, FAQ, freshness): tags, versions, staleness, conflicts,
reindex, housekeeping and the event stream."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import Response, StreamingResponse

from kb_api.deps import SCOPE_ADMIN, SCOPE_READ, SCOPE_WRITE, KBThread, manager_of, thread_for
from kb_api.errors import NOT_FOUND, KBError, ok, ok_list, page_args

router = APIRouter(prefix="/api/v1")
READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)
MAX_STREAM_SECONDS = 3600
HEARTBEAT_SECONDS = 15


# --- tags -----------------------------------------------------------------------------------------
@router.get("/knowledge-bases/{kb_id}/tags")
async def list_tags(kb_id: str, keyword: str | None = None, page: int = 1, page_size: int = 50,
                    thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.list_tags(kb_id, keyword, page, page_size)
    return ok_list(items, total, page, page_size)


@router.post("/knowledge-bases/{kb_id}/tags", status_code=201)
async def create_tag(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.create_tag(kb_id, body), status=201)


@router.put("/knowledge-bases/{kb_id}/tags/{tag_id}")
async def update_tag(kb_id: str, tag_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_tag(kb_id, tag_id, body))


@router.delete("/knowledge-bases/{kb_id}/tags/{tag_id}")
async def delete_tag(kb_id: str, tag_id: str, force: bool = False, thread: KBThread = Depends(WRITE)):
    return ok(await thread.delete_tag(kb_id, tag_id, force))


@router.put("/knowledge/tags")
async def set_knowledge_tags(body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.set_knowledge_tags(body.get("updates") or {}))


# --- versions --------------------------------------------------------------------------------------
@router.get("/knowledge/{knowledge_id}/versions")
async def versions(knowledge_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.versions(knowledge_id))


@router.get("/knowledge/{knowledge_id}/versions/{version}/download")
async def download_version(knowledge_id: str, version: int, thread: KBThread = Depends(READ)):
    name, data = await thread.download_version(knowledge_id, version)
    return Response(data, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


# --- freshness ---------------------------------------------------------------------------------------
@router.get("/knowledge-bases/{kb_id}/staleness")
async def staleness(kb_id: str, stale_after_days: int | None = None, expiring_within_days: int = 30,
                    never_retrieved_grace_days: int = 14, thread: KBThread = Depends(READ)):
    return ok(await thread.staleness(kb_id, stale_after_days=stale_after_days,
                                     expiring_within_days=expiring_within_days,
                                     never_retrieved_grace_days=never_retrieved_grace_days))


@router.get("/knowledge-bases/{kb_id}/conflicts")
async def conflicts(kb_id: str, status: str = "open", knowledge_id: str | None = None,
                    thread: KBThread = Depends(READ)):
    return ok(await thread.conflicts(kb_id, status, knowledge_id))


@router.put("/knowledge-conflicts/{conflict_id}")
async def review_conflict(conflict_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.review_conflict(conflict_id, body))


@router.post("/knowledge-bases/{kb_id}/reindex")
async def reindex(kb_id: str, body: dict[str, Any] = Body(default={}),
                  thread: KBThread = Depends(thread_for(SCOPE_WRITE, allow_machine=False))):
    return ok(await thread.reindex(kb_id, body or {}))


@router.post("/system/housekeeping")
async def housekeeping(body: dict[str, Any] = Body(default={}),
                       thread: KBThread = Depends(thread_for(SCOPE_ADMIN, allow_machine=False))):
    body = body or {}
    return ok(await thread.housekeeping(bool(body.get("dry_run", True)), body.get("retention_days")))


# --- events (SSE) --------------------------------------------------------------------------------------
@router.get("/events")
async def events(request: Request, thread: KBThread = Depends(READ)):
    """Status changes of knowledge the caller can read. Served by the main `kb` process only."""
    if request.app.state.settings.mode == "service":
        raise KBError(NOT_FOUND, "not found")
    manager = manager_of(request)
    ctx = thread.auth
    until = time.time() + MAX_STREAM_SECONDS
    if ctx.expires_at:
        until = min(until, ctx.expires_at)

    async def stream():
        sub = manager.events.subscribe(ctx)
        try:
            yield "retry: 2000\n\n"
            yield f"event: status\ndata: {json.dumps({'status': 'connected'})}\n\n"
            while time.time() < until:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(sub.queue.get(), timeout=HEARTBEAT_SECONDS)
                    yield f"event: message\ndata: {json.dumps(item)}\n\n"
                except asyncio.TimeoutError:
                    yield ": idle\n\n"
        finally:
            manager.events.unsubscribe(sub)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
