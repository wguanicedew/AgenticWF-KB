"""The LLM wiki (docs/llm-wiki.md, API) under /knowledgebase/{kb_id}/wiki — OpenKB's singular
`knowledgebase`, kept. Slugs contain `/` (`entity/rucio`), so page routes take a path.

GETs and POST /global-search need `knowledge:read`; every other call `knowledge:write`. Fixed
paths come before the `{slug:path}` routes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from kb_api.deps import SCOPE_READ, SCOPE_WRITE, KBThread, thread_for
from kb_api.errors import NOT_FOUND, KBError, ok, ok_list
from kb_runtime.thread.thread import feature_enabled


def _enabled():
    if not feature_enabled("wiki"):
        raise KBError(NOT_FOUND, "the wiki is switched off in this deployment")


router = APIRouter(prefix="/api/v1/knowledgebase/{kb_id}/wiki", dependencies=[Depends(_enabled)])
READ = thread_for(SCOPE_READ)
WRITE = thread_for(SCOPE_WRITE)


@router.get("/pages")
async def list_pages(kb_id: str, page_type: str | None = None, folder_id: str | None = None,
                     status: str | None = "published", page: int = 1, page_size: int = 50,
                     thread: KBThread = Depends(READ)):
    if page < 1 or not 1 <= page_size <= 200:
        raise KBError(1010, "page must be ≥ 1 and page_size between 1 and 200")
    items, total = await thread.wiki_pages(kb_id, page_type=page_type, folder_id=folder_id, status=status or None,
                                           page=page, page_size=page_size)
    return ok_list(items, total, page, page_size)


@router.post("/pages", status_code=201)
async def create_page(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_create_page(kb_id, body), status=201)


@router.put("/move-page")
async def move_page(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_move_page(kb_id, body.get("slug") or "", body.get("folder_id") or ""))


@router.get("/folders")
async def folders(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_folders(kb_id))


@router.post("/folders", status_code=201)
async def create_folder(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_create_folder(kb_id, body), status=201)


@router.put("/folders/{folder_id}")
async def update_folder(kb_id: str, folder_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_update_folder(kb_id, folder_id, body))


@router.delete("/folders/{folder_id}")
async def delete_folder(kb_id: str, folder_id: str, thread: KBThread = Depends(WRITE)):
    await thread.wiki_delete_folder(kb_id, folder_id)
    return ok(None)


@router.get("/index")
async def index(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_index(kb_id))


@router.get("/graph")
async def graph(kb_id: str, slug: str | None = None, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_graph(kb_id, slug))


@router.get("/stats")
async def stats(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_stats(kb_id))


@router.get("/search")
async def search(kb_id: str, q: str, regex: bool = True, limit: int = 10, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_search(kb_id, q, regex, limit))


@router.get("/revisions/{slug:path}")
async def revisions(kb_id: str, slug: str, version: int | None = None, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_revisions(kb_id, slug, version))


@router.post("/revert")
async def revert(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    if body.get("version") is None:
        raise KBError(1000, "version is required")
    return ok(await thread.wiki_revert(kb_id, body.get("slug") or "", int(body["version"])))


@router.get("/issues")
async def issues(kb_id: str, status: str | None = "pending", thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_issues(kb_id, status or None))


@router.post("/issues", status_code=201)
async def flag_issue(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_flag_issue(kb_id, body), status=201)


@router.put("/issues/{issue_id}/status")
async def issue_status(kb_id: str, issue_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_issue_status(kb_id, issue_id, body.get("status")))


@router.get("/quality")
async def quality(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_quality(kb_id))


@router.get("/lint")
async def lint(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_lint(kb_id))


@router.post("/rebuild-links")
async def rebuild_links(kb_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_rebuild_links(kb_id))


@router.post("/rebuild", status_code=202)
async def rebuild(kb_id: str, thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_rebuild(kb_id), status=202)


@router.post("/global-search")
async def global_search(kb_id: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_global_search(kb_id, body.get("query") or "", int(body.get("level") or 0)))


@router.get("/relations")
async def relations(kb_id: str, slug: str | None = None, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_relations(kb_id, slug))


@router.get("/communities")
async def communities(kb_id: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_communities(kb_id))


@router.get("/pages/{slug:path}")
async def get_page(kb_id: str, slug: str, thread: KBThread = Depends(READ)):
    return ok(await thread.wiki_page(kb_id, slug))


@router.put("/pages/{slug:path}")
async def update_page(kb_id: str, slug: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(WRITE)):
    return ok(await thread.wiki_update_page(kb_id, slug, body))


@router.delete("/pages/{slug:path}")
async def delete_page(kb_id: str, slug: str, thread: KBThread = Depends(WRITE)):
    await thread.wiki_delete_page(kb_id, slug)
    return ok(None)
