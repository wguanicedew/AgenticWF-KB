"""Models (docs/api.md, Models): system models from kb.cfg, plus users' own.

    POST, GET          /api/v1/models                (GET ?type=Embedding|Rerank|KnowledgeQA|VLLM)
    GET, PUT, DELETE   /api/v1/models/{id}           id or system:<name>
    POST               /api/v1/models/{id}/debug     one small real call
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from kb_api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from kb_api.errors import ok

router = APIRouter(prefix="/api/v1/models")


@router.post("", status_code=201)
async def create_model(body: dict[str, Any] = Body(...),
                       thread: KBThread = Depends(thread_for(SCOPE_WRITE, allow_machine=False))):
    return ok(await thread.create_model(body), status=201)


@router.get("")
async def list_models(type: str | None = None, thread: KBThread = Depends(thread_for(SCOPE_READ))):
    return ok(await thread.list_models(type))


@router.get("/{model_id}")
async def get_model(model_id: str, thread: KBThread = Depends(thread_for(SCOPE_READ))):
    return ok(await thread.get_model(model_id))


@router.put("/{model_id}")
async def update_model(model_id: str, body: dict[str, Any] = Body(...),
                       thread: KBThread = Depends(thread_for(SCOPE_WRITE, allow_machine=False))):
    return ok(await thread.update_model(model_id, body))


@router.delete("/{model_id}")
async def delete_model(model_id: str, thread: KBThread = Depends(thread_for(SCOPE_WRITE, allow_machine=False))):
    await thread.delete_model(model_id)
    return ok({"id": model_id, "deleted": True})


@router.post("/{model_id}/debug")
async def debug_model(model_id: str, thread: KBThread = Depends(thread_for(SCOPE_WRITE))):
    return ok(await thread.debug_model(model_id))
