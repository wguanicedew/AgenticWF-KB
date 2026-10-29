"""Knowledge bases, knowledge, chunks and search (docs/api.md).

Every route reaches the runtime through the request's KBThread. `{kb_id}` accepts an id or
`<owner>:<slug>`.
"""

from __future__ import annotations

import json
import mimetypes
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response

from agenticwf.kb.api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from agenticwf.kb.api.errors import VALIDATION, KBError, ok, ok_list, page_args
from agenticwf.kb.runtime.thread.thread import SearchOptions

router = APIRouter(prefix="/api/v1")

READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)
PEOPLE_WRITE = thread_for(SCOPE_WRITE, allow_machine=False)


def _float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise KBError(VALIDATION, f"not a number: {value!r}")


def _bool(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on") if value is not None else False


def _list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [v.strip() for v in str(value).split(",") if v.strip()]


# --- knowledge bases ---------------------------------------------------------------------------
@router.post("/knowledge-bases", status_code=201)
async def create_kb(body: dict[str, Any] = Body(...), thread: KBThread = Depends(PEOPLE_WRITE)):
    return ok(await thread.create_kb(body), status=201)


@router.get("/knowledge-bases")
async def list_kbs(page: int = 1, page_size: int = 20, type: str | None = None, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.list_kbs(page, page_size, kb_type=type)
    return ok_list(items, total, page, page_size)


@router.get("/shared-knowledge-bases")
async def shared_kbs(page: int = 1, page_size: int = 20, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.list_kbs(page, page_size, shared_only=True)
    return ok_list(items, total, page, page_size)


@router.get("/knowledge-bases/{kb_id}")
async def get_kb(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.get_kb(kb_id))


@router.put("/knowledge-bases/{kb_id}")
async def update_kb(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(PEOPLE_WRITE)):
    return ok(await thread.update_kb(kb_id, body))


@router.delete("/knowledge-bases/{kb_id}")
async def delete_kb(kb_id: str, purge: bool = False, thread: KBThread = Depends(PEOPLE_WRITE)):
    """`purge=true` erases every row at once instead of after `[maintenance] purge_after_days`."""
    await thread.delete_kb(kb_id, purge)
    return ok({"id": kb_id, "deleted": True, "purged": purge})


# --- search -----------------------------------------------------------------------------------------
def _options(params: dict) -> SearchOptions:
    top_k = params.get("match_count")
    return SearchOptions(vector_threshold=_float(params.get("vector_threshold")),
                         keyword_threshold=_float(params.get("keyword_threshold")),
                         top_k=int(top_k) if top_k not in (None, "") else None,
                         knowledge_ids=_list(params.get("knowledge_ids")),
                         tag_ids=_list(params.get("tag_ids")),
                         disable_vector=_bool(params.get("disable_vector_match")),
                         disable_keywords=_bool(params.get("disable_keywords_match")),
                         enable_rerank=None if params.get("enable_rerank") is None else _bool(params["enable_rerank"]))


@router.get("/knowledge-bases/{kb_id}/hybrid-search")
async def hybrid_search_get(kb_id: str, query_text: str = Query(...), vector_threshold: float | None = None,
                            keyword_threshold: float | None = None, match_count: int | None = None,
                            thread: KBThread = Depends(READ)):
    options = _options({"vector_threshold": vector_threshold, "keyword_threshold": keyword_threshold,
                        "match_count": match_count})
    return ok(await thread.search(query_text, [kb_id], options))


@router.post("/knowledge-bases/{kb_id}/hybrid-search")
async def hybrid_search_post(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.search(body.get("query_text") or body.get("query") or "", [kb_id], _options(body)))


@router.post("/knowledge-search")
async def knowledge_search(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    query = body.get("query") or ""
    if not query.strip():
        raise KBError(1000, "query is required")
    kbs = _list(body.get("knowledge_base_ids"))
    if body.get("knowledge_base_id"):
        kbs.append(body["knowledge_base_id"])
    return ok(await thread.search(query, kbs or None, _options(body)))


@router.post("/knowledge/{knowledge_id}/regenerate-summary")
async def regenerate_summary(knowledge_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.regenerate_summary(knowledge_id))


@router.post("/knowledge-grep")
async def knowledge_grep(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    regex = True if body.get("regex") is None else _bool(body["regex"])
    return ok(await thread.grep(body.get("pattern") or "", _list(body.get("knowledge_base_ids")) or None,
                                _list(body.get("knowledge_ids")), regex=regex,
                                case_sensitive=_bool(body.get("case_sensitive")), limit=body.get("limit") or 20))


# --- knowledge ---------------------------------------------------------------------------------
def _form_fields(**fields) -> dict:
    out = {k: v for k, v in fields.items() if v not in (None, "")}
    if "metadata" in out:
        try:
            out["metadata"] = json.loads(out["metadata"])
        except ValueError:
            raise KBError(VALIDATION, "metadata must be JSON")
    if "groups" in out:
        out["groups"] = _list(out["groups"])
    return out


@router.post("/knowledge-bases/{kb_id}/knowledge/file")
async def upload_file(kb_id: str, file: UploadFile = File(...), fileName: str | None = Form(default=None),
                      metadata: str | None = Form(default=None), mode: str | None = Form(default=None),
                      external_id: str | None = Form(default=None), folder_path: str | None = Form(default=None),
                      valid_from: str | None = Form(default=None), valid_until: str | None = Form(default=None),
                      expires_at: str | None = Form(default=None), deprecated: str | None = Form(default=None),
                      visibility: str | None = Form(default=None), groups: str | None = Form(default=None),
                      tag_ids: str | None = Form(default=None), record_type: str | None = Form(default=None),
                      column_map: str | None = Form(default=None), citation: str | None = Form(default=None),
                      citation_text: str | None = Form(default=None), authors: str | None = Form(default=None),
                      published: str | None = Form(default=None), source_url: str | None = Form(default=None),
                      doi: str | None = Form(default=None), publisher: str | None = Form(default=None),
                      version: str | None = Form(default=None), thread: KBThread = Depends(WRITE)):
    data = await file.read()
    fields = _form_fields(fileName=fileName, metadata=metadata, mode=mode, external_id=external_id,
                          folder_path=folder_path, valid_from=valid_from, valid_until=valid_until,
                          expires_at=expires_at, deprecated=deprecated, visibility=visibility, groups=groups,
                          tag_ids=tag_ids, record_type=record_type, column_map=column_map, citation=citation,
                          citation_text=citation_text, authors=authors, published=published, source_url=source_url,
                          doi=doi, publisher=publisher, version=version)
    if "tag_ids" in fields:
        fields["tag_ids"] = _list(fields["tag_ids"])
    outcome, item = await thread.add_file(kb_id, file.filename or "upload", data, fields)
    return ok({**item, "outcome": outcome}, status=201 if outcome == "created" else 200)


@router.post("/knowledge-bases/{kb_id}/knowledge/url", status_code=201)
async def add_url(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.add_url(kb_id, body), status=201)


@router.post("/knowledge-bases/{kb_id}/sites", status_code=201)
async def add_site(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    """Import every page under one URL (docs/ingestion.md, Sites)."""
    return ok(await thread.add_site(kb_id, body), status=201)


@router.get("/knowledge-bases/{kb_id}/sites")
async def list_sites(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.list_sites(kb_id))


@router.get("/sites/{site_id}")
async def get_site(site_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.get_site(site_id))


@router.put("/sites/{site_id}")
async def update_site(site_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_site(site_id, body))


@router.post("/sites/{site_id}/refresh")
async def refresh_site(site_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.refresh_site(site_id))


@router.delete("/sites/{site_id}")
async def delete_site(site_id: str, delete_pages: bool = True, purge: bool = False,
                      thread: KBThread = Depends(WRITE)):
    return ok(await thread.delete_site(site_id, delete_pages, purge))


@router.post("/knowledge-bases/{kb_id}/knowledge/manual", status_code=201)
async def add_manual(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.add_manual(kb_id, body), status=201)


@router.get("/knowledge-bases/{kb_id}/knowledge")
async def list_knowledge(kb_id: str, page: int = 1, page_size: int = 20, keyword: str | None = None,
                         file_type: str | None = None, parse_status: str | None = None,
                         tag_ids: str | None = None, tag_id: str | None = None, site_id: str | None = None,
                         thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    tags = _list(tag_ids) + ([tag_id] if tag_id else [])
    items, total = await thread.list_knowledge(kb_id, page, page_size, {"keyword": keyword, "file_type": file_type,
                                                                        "parse_status": parse_status,
                                                                        "tag_ids": tags, "site_id": site_id})
    return ok_list(items, total, page, page_size)


@router.delete("/knowledge-bases/{kb_id}/knowledge")
async def clear_knowledge(kb_id: str, purge: bool = False, thread: KBThread = Depends(WRITE)):
    """Delete every item of the KB the caller may manage."""
    deleted, page = 0, 1
    while True:
        items, _ = await thread.list_knowledge(kb_id, page, 100)
        if not items:
            break
        for item in items:
            try:
                await thread.delete_knowledge(item["id"], purge)
                deleted += 1
            except Exception:
                continue
        if len(items) < 100:
            break
    return ok({"deleted": deleted})


@router.get("/knowledge/batch")
async def batch(ids: str = Query(...), thread: KBThread = Depends(READ)):
    return ok(await thread.batch_knowledge(_list(ids)))


@router.get("/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.get_knowledge(knowledge_id))


@router.put("/knowledge/manual/{knowledge_id}")
async def update_manual(knowledge_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_manual(knowledge_id, body))


@router.put("/knowledge/{knowledge_id}")
async def update_knowledge(knowledge_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_knowledge(knowledge_id, body))


@router.put("/knowledge/{knowledge_id}/file")
async def replace_file(knowledge_id: str, file: UploadFile = File(...), thread: KBThread = Depends(WRITE)):
    item = await thread.get_knowledge(knowledge_id)
    outcome, item = await thread.add_file(item["knowledge_base_id"], file.filename or item["file_name"],
                                          await file.read(), {"mode": "replace", "external_id": item["external_id"],
                                                              "fileName": item["file_name"]})
    return ok({**item, "outcome": outcome})


@router.delete("/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str, purge: bool = False, thread: KBThread = Depends(WRITE)):
    """`purge=true` erases the row at once instead of after `[maintenance] purge_after_days`."""
    await thread.delete_knowledge(knowledge_id, purge)
    return ok({"id": knowledge_id, "deleted": True, "purged": purge})


@router.get("/knowledge/{knowledge_id}/download")
async def download(knowledge_id: str, thread: KBThread = Depends(READ)):
    name, data = await thread.download(knowledge_id)
    media = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return Response(data, media_type=media,
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


@router.get("/knowledge/{knowledge_id}/content")
async def content(knowledge_id: str, offset: int = 0, limit: int = 8000, thread: KBThread = Depends(READ)):
    return ok(await thread.knowledge_content(knowledge_id, offset, limit))


@router.post("/knowledge/{knowledge_id}/redact")
async def redact(knowledge_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    """Remove matching text from the item everywhere it is kept (docs/ingestion.md, Redaction)."""
    return ok(await thread.redact(knowledge_id, body))


@router.post("/knowledge-bases/{kb_id}/redact")
async def redact_kb(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.redact_kb(kb_id, body))


@router.post("/knowledge/{knowledge_id}/reparse")
async def reparse(knowledge_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.reparse(knowledge_id))


@router.post("/knowledge/{knowledge_id}/cancel-parse")
async def cancel(knowledge_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.cancel(knowledge_id))


# --- chunks -----------------------------------------------------------------------------------------
@router.get("/chunks/by-id/{chunk_id}")
async def chunk_by_id(chunk_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.get_chunk(chunk_id))


@router.get("/chunks/{knowledge_id}")
async def list_chunks(knowledge_id: str, page: int = 1, page_size: int = 20, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    items, total = await thread.list_chunks(knowledge_id, page, page_size)
    return ok_list(items, total, page, page_size)


@router.put("/chunks/{knowledge_id}/{chunk_id}")
async def update_chunk(knowledge_id: str, chunk_id: str, body: dict[str, Any] = Body(...),
                       thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_chunk(knowledge_id, chunk_id, body))


@router.delete("/chunks/{knowledge_id}/{chunk_id}")
async def delete_chunk(knowledge_id: str, chunk_id: str, thread: KBThread = Depends(WRITE)):
    await thread.delete_chunks(knowledge_id, chunk_id)
    return ok({"id": chunk_id, "deleted": True})


@router.delete("/chunks/{knowledge_id}")
async def delete_chunks(knowledge_id: str, thread: KBThread = Depends(WRITE)):
    await thread.delete_chunks(knowledge_id)
    return ok({"knowledge_id": knowledge_id, "deleted": True})
