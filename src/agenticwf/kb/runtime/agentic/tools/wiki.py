"""
Wiki read tools (docs/agentic-search.md, Wiki read tools; docs/llm-wiki.md): wiki_search,
wiki_read_page, wiki_global_search. Read-only; offered with the `wiki` preset, and in `deep` when
a wiki-enabled KB is in scope. Pages come back as `pN`, their source chunks as `cN` (with `dN`).
"""

from __future__ import annotations

import re

import sqlalchemy as sa

from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.ext import WikiPage
from agenticwf.kb.core.orm.models.kb import Chunk, Knowledge
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.agentic.tools import Tool, ToolContext, ToolError, attr, int_arg, list_arg
from agenticwf.kb.runtime.errors import Invalid, NotFound

SOURCES = 30
SOURCE_CHARS = 200


def page_payload(page, score=None) -> dict:
    sources = [r.split("|", 1)[-1] for r in getattr(page, "source_refs", None) or []]
    text = f"Wiki: {page.title}" + (f" (compiled from: {'; '.join(sources[:5])})" if sources else "")
    return {"kind": "wiki_page", "slug": page.slug, "title": page.title, "knowledge_base_id": page.knowledge_base_id,
            "content": f"{page.title}\n{page.summary}", "score": score, "citation_text": text}


def page_handle(tc: ToolContext, page, score=None) -> str:
    return tc.handles.issue("p", f"{page.knowledge_base_id}:{page.slug}", page_payload(page, score))


def wiki_kb_ids(tc: ToolContext, handles: list[str] | None = None) -> list[str]:
    ids = tc.kb_ids(handles)
    out = [i for i in ids if (tc.scope.kbs[i].indexing_strategy or {}).get("wiki_enabled")]
    if not out:
        raise ToolError("no wiki-enabled knowledge base is in scope")
    return out


async def wiki_search(tc: ToolContext, args: dict) -> str:
    query = str(args.get("query") or "")
    if not query.strip():
        raise ToolError("query is required")
    kb_ids = wiki_kb_ids(tc, list_arg(args, "knowledge_base_ids"))
    async with async_session_scope(commit=False) as session:
        pages = await tc.services.wiki.search_pages(session, kb_ids, query, args.get("regex", True) is not False,
                                                    int_arg(args, "limit", 10, 1, 30))
        for p in pages:
            session.expunge(p)
    if not pages:
        return f"No wiki page matches {query!r}. Try search_knowledge."
    return tc.clip("\n".join(f'<page id="{page_handle(tc, p)}" slug="{attr(p.slug)}" type="{p.page_type}" '
                             f'title="{attr(p.title)}">{attr(p.summary)}</page>' for p in pages))


async def wiki_read_page(tc: ToolContext, args: dict) -> str:
    refs = list_arg(args, "slugs")[:5]
    if not refs:
        raise ToolError("give 1 to 5 slugs or pN handles")
    kb_ids = wiki_kb_ids(tc)
    out = []
    async with async_session_scope(commit=False) as session:
        for ref in refs:
            handle = tc.handles.lookup(ref)
            if handle is not None:
                if handle.prefix != "p":
                    raise ToolError(f"{ref} is not a wiki page")
                kb_id, _, slug = handle.key.partition(":")
                candidates = [kb_id]
            else:
                slug, candidates = ref.strip(), kb_ids
            page = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id.in_(candidates), WikiPage.slug == slug, WikiPage.deleted_at.is_(None),
                WikiPage.status == "published"))).scalars().first()
            if page is None:
                out.append(f'<wiki_page slug="{attr(slug)}" missing="true"/>')
                continue
            handle_text = page_handle(tc, page)
            chunk_ids = [r.split("|")[1] for r in (page.chunk_refs or []) if r.count("|") >= 2][:SOURCES]
            rows = (await session.execute(sa.select(Chunk, Knowledge).join(Knowledge, Knowledge.id == Chunk.knowledge_id)
                                          .where(Chunk.id.in_(chunk_ids or ["-"]), Knowledge.deleted_at.is_(None)))).all()
            sources = []
            from agenticwf.kb.runtime.agentic.tools.knowledge import chunk_handle, doc_handle

            for chunk, item in rows:
                if not access.can_read_item(tc.auth, item):
                    continue
                c = chunk_handle(tc, {"id": chunk.id, "knowledge_id": item.id, "knowledge_title": item.title,
                                      "knowledge_base_id": item.knowledge_base_id, "content": chunk.content,
                                      "score": None, "chunk_index": chunk.chunk_index, "chunk_type": chunk.chunk_type})
                d = doc_handle(tc, item.id, item.title, item.knowledge_base_id)
                sources.append(f'<source id="{c}" doc="{d}" title="{attr(item.title)}">'
                               f'{attr(chunk.content[:SOURCE_CHARS])}</source>')
            content = re.sub(r"\[src:\d+\]", "", page.content or "")
            out.append(f'<wiki_page id="{handle_text}" slug="{attr(page.slug)}" type="{page.page_type}" '
                       f'version="{page.version}">\n<metadata title="{attr(page.title)}" aliases="{attr(", ".join(page.aliases or []))}" '
                       f'folder="{attr(page.category_path)}" in_links="{len(page.in_links or [])}" '
                       f'updated="{page.updated_at.date().isoformat() if page.updated_at else ""}"/>\n'
                       f'<summary>{page.summary}</summary>\n<content>\n{content}\n</content>\n'
                       f'<sources>\n' + "\n".join(sources) + "\n</sources>\n</wiki_page>")
    return tc.clip("\n".join(out))


async def wiki_global_search(tc: ToolContext, args: dict) -> str:
    query = str(args.get("query") or "")
    kb_ids = wiki_kb_ids(tc, list_arg(args, "knowledge_base_ids"))
    level = int_arg(args, "level", 0, 0, 1)
    parts = []
    for kb_id in kb_ids:
        try:
            result = await tc.services.wiki_pipeline.global_search(tc.auth, kb_id, query, level)
        except (Invalid, NotFound) as exc:
            raise ToolError(str(exc))
        if not result["answer"]:
            continue
        answer = result["answer"]
        async with async_session_scope(commit=False) as session:
            pages = {p.slug: p for p in (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb_id, WikiPage.slug.in_(result["sources"] or ["-"])))).scalars()}
            for p in pages.values():
                session.expunge(p)

        def rekey(match):
            page = pages.get(match.group(1).strip())
            title = match.group(2) or (page.title if page else match.group(1))
            return f"{title} ({page_handle(tc, page)})" if page else title

        answer = re.sub(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]+?))?\]\]", rekey, answer)
        sources = ", ".join(f"{page_handle(tc, pages[s])}" for s in result["sources"] if s in pages)
        parts.append(f'<global_answer kb="{tc.handles.issue("b", kb_id, {"name": tc.scope.kbs[kb_id].name})}" '
                     f'sources="{sources}">\n{answer}\n</global_answer>')
    if not parts:
        return "The wiki has no community or summary pages that answer this yet. Try wiki_search or search_knowledge."
    return tc.clip("\n".join(parts))


TOOLS = [
    Tool("wiki_search", "Find wiki pages by title, alias, summary or content (case-insensitive regex). Returns "
         "pages as pN with their summaries.",
         {"type": "object", "properties": {"query": {"type": "string"}, "regex": {"type": "boolean"},
                                           "knowledge_base_ids": {"type": "array", "items": {"type": "string"}},
                                           "limit": {"type": "integer", "minimum": 1, "maximum": 30}},
          "required": ["query"]}, wiki_search, group="wiki"),
    Tool("wiki_read_page", "Read up to 5 wiki pages (slugs or pN): metadata, summary, content and up to 30 source "
         "passages as citable cN (verify exact claims with read_document).",
         {"type": "object", "properties": {"slugs": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
          "required": ["slugs"]}, wiki_read_page, group="wiki"),
    Tool("wiki_global_search", "Answer a question about the whole corpus (themes, overall risks) from the wiki's "
         "community pages: map over communities, reduce to one answer with pN sources.",
         {"type": "object", "properties": {"query": {"type": "string"}, "level": {"type": "integer", "minimum": 0,
                                                                                    "maximum": 1},
                                           "knowledge_base_ids": {"type": "array", "items": {"type": "string"}}},
          "required": ["query"]}, wiki_global_search, group="wiki"),
]
