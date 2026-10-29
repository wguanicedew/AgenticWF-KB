"""FAQ entries (docs/faq.md, API) under /knowledge-bases/{kb_id}/faq.

Fixed paths (`/entries/export`, `/entries/fields`, `/entries/tags`) are declared before
`/entries/{entry_id}` so they are not taken for an entry id.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.responses import Response

from agenticwf.kb.api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from agenticwf.kb.api.errors import VALIDATION, KBError, ok, ok_list, page_args

router = APIRouter(prefix="/api/v1/knowledge-bases/{kb_id}/faq")
READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)


def _bool(value) -> bool | None:
    if value in (None, ""):
        return None
    return str(value).lower() in ("1", "true", "yes", "on")


@router.get("/entries")
async def list_entries(kb_id: str, page: int = 1, page_size: int = 20, keyword: str | None = None,
                       tag_id: str | None = None, is_enabled: str | None = None, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.faq_entries(kb_id, page, page_size, keyword=keyword, tag_id=tag_id,
                                            is_enabled=_bool(is_enabled))
    return ok_list(items, total, page, page_size)


@router.get("/entries/export")
async def export_entries(kb_id: str, thread: KBThread = Depends(READ)):
    name, data = await thread.faq_export(kb_id)
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


@router.post("/entries")
async def upsert_entries(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_upsert(kb_id, body.get("entries") or [], body.get("mode") or "append",
                                      bool(_bool(body.get("dry_run")))))


@router.put("/entries/fields")
async def update_fields(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_update_fields(kb_id, body))


@router.put("/entries/tags")
async def update_tags(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_update_tags(kb_id, body.get("updates") or {}))


@router.delete("/entries")
async def delete_entries(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_delete(kb_id, body.get("ids") or []))


@router.post("/entry", status_code=201)
async def create_entry(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_create(kb_id, body), status=201)


@router.get("/entries/{entry_id}")
async def get_entry(kb_id: str, entry_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.faq_entry(kb_id, entry_id))


@router.put("/entries/{entry_id}")
async def update_entry(kb_id: str, entry_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.faq_update(kb_id, entry_id, body))


@router.post("/entries/{entry_id}/similar-questions")
async def add_similar(kb_id: str, entry_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    questions = body.get("similar_questions")
    if not isinstance(questions, list) or not questions:
        raise KBError(VALIDATION, "similar_questions must be a non-empty list")
    return ok(await thread.faq_add_similar(kb_id, entry_id, questions))


@router.post("/search")
async def search(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.faq_search(kb_id, body))


@router.post("/import")
async def import_entries(kb_id: str, file: UploadFile = File(...), mode: str = Form("append"),
                         dry_run: str = Form("false"), thread: KBThread = Depends(WRITE)):
    data = await file.read()
    return ok(await thread.faq_import(kb_id, file.filename or "faq.csv", data, mode or "append",
                                      bool(_bool(dry_run))))
