"""Structured records (docs/structured-records.md, API): record types, records, query,
aggregate, similar and plugin operations. Phase 3, behind `[features] records`.

`query`, `aggregate`, `similar`, `suggest` and `infer` are read-only POSTs (`knowledge:read`).
Fixed paths are declared before `{name}` and `{record_id}` so they are not taken for one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile

from agenticwf.kb.api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from agenticwf.kb.api.errors import NOT_FOUND, KBError, ok, ok_list, page_args
from agenticwf.kb.runtime.thread.thread import feature_enabled

router = APIRouter(prefix="/api/v1")


def _enabled():
    if not feature_enabled("records"):
        raise KBError(NOT_FOUND, "records are switched off in this deployment")


READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)


# --- record types -----------------------------------------------------------------------------------
@router.get("/record-types", dependencies=[Depends(_enabled)])
async def list_types(all_versions: bool = False, thread: KBThread = Depends(READ)):
    return ok(await thread.record_types(all_versions))


@router.post("/record-types", status_code=201, dependencies=[Depends(_enabled)])
async def register_type(body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.register_record_type(body), status=201)


@router.post("/record-types/infer", dependencies=[Depends(_enabled)])
async def infer_type(request: Request, thread: KBThread = Depends(READ)):
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise KBError(1000, "give a file or JSON samples")
        data = await upload.read()
        return ok(await thread.infer_record_type({"name": form.get("name"), "llm": form.get("llm") == "true"},
                                                 upload.filename, data))
    return ok(await thread.infer_record_type(await request.json()))


@router.get("/record-types/{name}", dependencies=[Depends(_enabled)])
async def get_type(name: str, thread: KBThread = Depends(READ)):
    return ok(await thread.record_type(name))


@router.put("/record-types/{name}", dependencies=[Depends(_enabled)])
async def update_type(name: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.update_record_type(name, body, bool(body.get("new_version"))))


@router.delete("/record-types/{name}", dependencies=[Depends(_enabled)])
async def deprecate_type(name: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.deprecate_record_type(name))


@router.get("/record-types/{name}/stats", dependencies=[Depends(_enabled)])
async def type_stats(name: str, knowledge_base_id: str | None = None, thread: KBThread = Depends(READ)):
    return ok(await thread.record_type_stats(name, knowledge_base_id))


@router.post("/record-types/{name}/migrate", dependencies=[Depends(_enabled)])
async def migrate_type(name: str, body: dict[str, Any] = Body(default={}), thread: KBThread = Depends(WRITE)):
    return ok(await thread.migrate_record_type(name, body))


@router.post("/record-types/{name}/ops/{op}", dependencies=[Depends(_enabled)])
async def type_operation(name: str, op: str, body: dict[str, Any] = Body(default={}), thread: KBThread = Depends(READ)):
    return ok(await thread.record_type_operation(name, op, body))


# --- records ---------------------------------------------------------------------------------------------
@router.post("/knowledge-bases/{kb_id}/records", dependencies=[Depends(_enabled)])
async def upsert_records(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.upsert_records(kb_id, body))


@router.get("/knowledge-bases/{kb_id}/records", dependencies=[Depends(_enabled)])
async def list_records(kb_id: str, request: Request, record_type: str | None = None, page: int = 1,
                       page_size: int = 20, thread: KBThread = Depends(READ)):
    page, page_size = page_args(page, page_size)
    facets = {k[len("facet."):]: v for k, v in request.query_params.items() if k.startswith("facet.")}
    items, total = await thread.list_records(kb_id, record_type, facets, page, page_size)
    return ok_list(items, total, page, page_size)


@router.post("/records/query", dependencies=[Depends(_enabled)])
async def query(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.query_records(body))


@router.post("/records/aggregate", dependencies=[Depends(_enabled)])
async def aggregate(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.aggregate_records(body))


@router.post("/records/similar", dependencies=[Depends(_enabled)])
async def similar(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.similar_records(body))


@router.post("/records/suggest", dependencies=[Depends(_enabled)])
async def suggest(body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.record_type_operation("system:hpo_job", "suggest", body))


@router.get("/records/{record_id}", dependencies=[Depends(_enabled)])
async def get_record(record_id: str, version: int | None = None, thread: KBThread = Depends(READ)):
    return ok(await thread.get_record(record_id, version))


@router.delete("/records/{record_id}", dependencies=[Depends(_enabled)])
async def delete_record(record_id: str, thread: KBThread = Depends(WRITE)):
    await thread.delete_record(record_id)
    return ok(None)
