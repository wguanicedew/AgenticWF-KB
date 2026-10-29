"""Remote knowledge bases (docs/retrieval.md, Remote knowledge bases): a user's remote KB servers
(/remotes) and the switches on their KBs. Reading is `knowledge:read`; adding, changing and
switching remotes are people's writes (`knowledge:write`). Operators' remotes come from kb.cfg;
only their switches change here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from agenticwf.kb.api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from agenticwf.kb.api.errors import ok

router = APIRouter(prefix="/api/v1")

READ = thread_for(SCOPE_READ)
PEOPLE_WRITE = thread_for(SCOPE_WRITE, allow_machine=False)


@router.get("/remotes")
async def remotes(thread: KBThread = Depends(READ)):
    return ok(await thread.remotes())


@router.post("/remotes", status_code=201)
async def create(body: dict[str, Any] = Body(...), thread: KBThread = Depends(PEOPLE_WRITE)):
    return ok(await thread.create_remote(body), status=201)


@router.get("/remotes/{remote_id}")
async def get(remote_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.get_remote(remote_id))


@router.put("/remotes/{remote_id}")
async def update(remote_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(PEOPLE_WRITE)):
    return ok(await thread.update_remote(remote_id, body))


@router.delete("/remotes/{remote_id}")
async def delete(remote_id: str, thread: KBThread = Depends(PEOPLE_WRITE)):
    await thread.delete_remote(remote_id)
    return ok(None)


@router.get("/remotes/{remote_id}/knowledge-bases")
async def knowledge_bases(remote_id: str, refresh: bool = False, thread: KBThread = Depends(READ)):
    return ok(await thread.remote_kbs(remote_id, refresh))


@router.put("/remotes/{remote_id}/knowledge-bases/{kb_id}")
async def switch(remote_id: str, kb_id: str, body: dict[str, Any] = Body(...),
                 thread: KBThread = Depends(PEOPLE_WRITE)):
    return ok(await thread.set_remote_kb(remote_id, kb_id, bool(body.get("enabled", True))))
