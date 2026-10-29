"""Agentic search (docs/agentic-search.md): POST /agentic-search (JSON or SSE) and the presets.

A run is bound to its request: a client disconnect cancels it, streaming or not. There is no
cancel route.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from agenticwf.kb.api.deps import SCOPE_READ, KBThread, thread_for
from agenticwf.kb.api.errors import ok

router = APIRouter(prefix="/api/v1")
READ = thread_for(SCOPE_READ)
DISCONNECT_POLL = 0.5


@router.get("/agentic-search/presets")
async def presets(thread: KBThread = Depends(READ)):
    return ok(thread.agentic_presets())


@router.get("/agentic-search/tools")
async def tools(thread: KBThread = Depends(READ)):
    return ok(await thread.agentic_tools())


@router.post("/agentic-search")
async def agentic_search(request: Request, body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    run = await thread.agentic_search(body)
    frames = run.frames()
    if body.get("stream"):
        async def events():
            try:
                async for item in frames:
                    yield f"event: message\ndata: {json.dumps(item, default=str)}\n\n"
            finally:
                await frames.aclose()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def collect():
        async for item in frames:
            if item["response_type"] == "complete":
                return item["data"]

    task = asyncio.create_task(collect())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL)
            if done:
                return ok(task.result())
            if await request.is_disconnected():
                task.cancel()
                break
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await frames.aclose()
    return ok(None)
