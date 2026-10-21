"""Temporal memory (docs/api.md, Temporal memory). Phase 3, behind `[features] memory`.

Every route reads or writes the caller's own memory only. `POST /memory/search` is a read-only
POST (`knowledge:read`); adding and deleting episodes needs `knowledge:write`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from kb_api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from kb_api.errors import NOT_FOUND, KBError, ok, ok_list, page_args
from kb_runtime.thread.thread import feature_enabled


def _enabled():
    if not feature_enabled("memory"):
        raise KBError(NOT_FOUND, "memory is switched off in this deployment")


router = APIRouter(prefix="/api/v1/memory", dependencies=[Depends(_enabled)])
READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)


@router.post("/episodes", status_code=202)
async def add_episode(body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.memory_add_episode(body), status=202)


@router.get("/episodes")
async def episodes(group_id: str | None = None, page: int = 1, page_size: int = 20, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.memory_episodes(group_id, page, page_size)
    return ok_list(items, total, page, page_size)


@router.get("/episodes/{episode_id}")
async def episode(episode_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.memory_episode(episode_id))


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.memory_delete_episode(episode_id))


@router.get("/facts")
async def facts(group_id: str | None = None, q: str | None = None, entity: str | None = None, as_of: str | None = None,
                known_at: str | None = None, include_invalid: bool = False, limit: int = 100,
                thread: KBThread = Depends(READ)):
    return ok(await thread.memory_facts(group_id=group_id, q=q, entity=entity, as_of=as_of, known_at=known_at,
                                        include_invalid=include_invalid, limit=limit))


@router.post("/search")
async def search(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.memory_search(body))


@router.get("/entities")
async def entities(group_id: str | None = None, q: str | None = None, page: int = 1, page_size: int = 50,
                   thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.memory_entities(group_id, q, page, page_size)
    return ok_list(items, total, page, page_size)


@router.get("/entities/{entity_id}")
async def entity(entity_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.memory_entity(entity_id))
