"""
Knowledge tools (docs/agentic-search.md, Knowledge tools): search_knowledge, grep_knowledge,
list_documents, read_document. Results carry handles (`cN`, `dN`), never raw ids.
"""

from __future__ import annotations

import asyncio
import re

import sqlalchemy as sa

from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import Chunk, Knowledge, utcnow
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.agentic.tools import Tool, ToolContext, ToolError, attr, int_arg, list_arg
from agenticwf.kb.runtime.errors import Invalid
from agenticwf.kb.runtime.faq.rules import apply_priority, priority_settings
from agenticwf.kb.runtime.retrieval.grep import grep
from agenticwf.kb.runtime.retrieval.search import SearchOptions

READ_LIMIT = 8000
WINDOW = 400
MAX_WINDOWS = 50
PAGE = 30
READABLE_TYPES = ("text", "parent_text")


# --- handles for results -----------------------------------------------------------------------------
def chunk_payload(hit: dict) -> dict:
    return {"kind": "chunk", "chunk_id": hit["id"], "knowledge_id": hit["knowledge_id"],
            "knowledge_title": hit.get("knowledge_title", ""), "knowledge_base_id": hit.get("knowledge_base_id", ""),
            "content": hit.get("content", ""), "score": hit.get("score"), "chunk_index": hit.get("chunk_index"),
            "chunk_type": hit.get("chunk_type", "text"), "knowledge_date": hit.get("knowledge_date"),
            "deprecated": bool(hit.get("deprecated")), **({"match": hit["match"]} if hit.get("match") else {}),
            **({"citation_text": hit["citation_text"], "url": (hit.get("citation") or {}).get("url")}
               if hit.get("citation_text") else {})}


def doc_handle(tc: ToolContext, knowledge_id: str, title: str, kb_id: str) -> str:
    return tc.handles.issue("d", knowledge_id, {"knowledge_id": knowledge_id, "knowledge_title": title,
                                                "knowledge_base_id": kb_id})


def chunk_handle(tc: ToolContext, hit: dict) -> str:
    return tc.handles.issue("c", hit["id"], chunk_payload(hit))


# --- search_knowledge ----------------------------------------------------------------------------------
async def run_searches(tc: ToolContext, queries: list[str], kb_ids: list[str], knowledge_ids: list[str],
                       mode: str = "hybrid", top_k: int = 8) -> list[dict]:
    """Each query through hybrid search (with rerank and the freshness filters), merged by chunk
    id keeping the best score, FAQ priority applied, cut to top_k."""
    options = SearchOptions(top_k=top_k, knowledge_ids=knowledge_ids, tag_ids=list(tc.scope.tag_ids),
                            disable_vector=mode == "keyword", disable_keywords=mode == "semantic")
    batches = await asyncio.gather(*(tc.services.search.search(tc.auth, q, kb_ids, options) for q in queries))
    best: dict[str, dict] = {}
    for hits in batches:
        for hit in hits:
            if hit["id"] not in best or hit["score"] > best[hit["id"]]["score"]:
                best[hit["id"]] = {k: v for k, v in hit.items() if k != "match"}
    merged = sorted(best.values(), key=lambda h: (-h["score"], h["knowledge_id"], h.get("chunk_index", 0)))
    return apply_priority(merged, top_k, priority_settings())


def hit_handle(tc: ToolContext, hit: dict) -> str:
    """cN for a chunk, rN for a record hit, pN for a wiki passage."""
    if hit.get("chunk_type") == "wiki_page":
        from agenticwf.kb.runtime.agentic.tools.wiki import page_handle
        from types import SimpleNamespace

        page = SimpleNamespace(slug=hit.get("wiki_slug"), title=hit.get("knowledge_title"),
                               summary=hit.get("knowledge_description") or "", knowledge_base_id=hit.get("knowledge_base_id"))
        return page_handle(tc, page, hit.get("score"))
    if hit.get("chunk_type") == "record":
        from agenticwf.kb.runtime.agentic.tools.records import record_handle

        return record_handle(tc, {"id": hit["id"], "record_type": hit.get("record_type"), "name": hit.get("record_name"),
                                  "knowledge_base_id": hit.get("knowledge_base_id")}, hit.get("score"))
    return chunk_handle(tc, hit)


def format_result(tc: ToolContext, hit: dict) -> str:
    if hit.get("chunk_type") == "wiki_page":
        handle = hit_handle(tc, hit)
        return (f'<wiki_passage id="{handle}" slug="{attr(hit.get("wiki_slug"))}" title="{attr(hit.get("knowledge_title"))}" '
                f'score="{hit["score"]:.3f}">\n{hit.get("content", "")}\n</wiki_passage>')
    if hit.get("chunk_type") == "record":
        handle = hit_handle(tc, hit)
        return (f'<record id="{handle}" type="{attr(hit.get("record_type"))}" name="{attr(hit.get("record_name"))}" '
                f'score="{hit["score"]:.3f}">\n{hit.get("content", "")}\n</record>')
    handle = chunk_handle(tc, hit)
    doc = doc_handle(tc, hit["knowledge_id"], hit.get("knowledge_title", ""), hit.get("knowledge_base_id", ""))
    extra = ""
    if hit.get("deprecated"):
        extra += ' deprecated="true"'
    if hit.get("chunk_type") == "faq":
        extra += ' type="faq"' + (' match="exact"' if hit.get("match") == "exact" else "")
    date = (hit.get("knowledge_date") or "")[:10]
    return (f'<result id="{handle}" doc="{doc}" title="{attr(hit.get("knowledge_title"))}" '
            f'source="{attr(hit.get("citation_text") or "")}" '
            f'chunk_index="{hit.get("chunk_index", 0)}" score="{hit["score"]:.3f}" date="{date}"{extra}>\n'
            f'{hit.get("content", "")}\n</result>')


async def search_knowledge(tc: ToolContext, args: dict) -> str:
    queries = list_arg(args, "queries") if isinstance(args.get("queries"), list) else (
        [args["queries"]] if isinstance(args.get("queries"), str) and args["queries"].strip() else [])
    if not queries and str(args.get("query") or "").strip():
        queries = [str(args["query"])]
    queries = [q.strip() for q in queries if q.strip()][:5]
    if not queries:
        raise ToolError("give 1 to 5 queries")
    mode = args.get("mode") or "hybrid"
    if mode not in ("hybrid", "keyword", "semantic"):
        raise ToolError("mode must be hybrid, keyword or semantic")
    hits = await run_searches(tc, queries, tc.kb_ids(list_arg(args, "knowledge_base_ids")),
                              tc.knowledge_ids(list_arg(args, "document_ids")), mode,
                              int_arg(args, "top_k", 8, 1, 20))
    if not hits:
        return ("No results. Rephrase the query, try mode=keyword for exact terms, or grep_knowledge for "
                "identifiers and error codes.")
    return tc.clip("\n".join(format_result(tc, hit) for hit in hits))


# --- grep_knowledge -----------------------------------------------------------------------------------
async def grep_knowledge(tc: ToolContext, args: dict) -> str:
    pattern = str(args.get("pattern") or "")
    regex = args.get("regex", True)
    try:
        found = await grep(tc.auth, tc.services.knowledge_bases, pattern,
                           kb_refs=tc.kb_ids(list_arg(args, "knowledge_base_ids")),
                           knowledge_ids=tc.knowledge_ids(list_arg(args, "document_ids")),
                           tag_ids=list(tc.scope.tag_ids), regex=bool(regex) if regex is not None else True,
                           case_sensitive=bool(args.get("case_sensitive")), limit=int_arg(args, "limit", 20, 1, 50))
    except Invalid as exc:
        raise ToolError(str(exc))
    if not found["matches"]:
        return f'No match for {pattern!r} (mode="{found["mode"]}"). Try search_knowledge for related wording.'
    out, current = [f'<grep pattern="{attr(pattern)}" mode="{found["mode"]}">'], None
    for m in found["matches"]:
        if m["knowledge_id"] != current:
            if current is not None:
                out.append("</doc>")
            current = m["knowledge_id"]
            doc = doc_handle(tc, m["knowledge_id"], m["knowledge_title"], m["knowledge_base_id"])
            out.append(f'<doc id="{doc}" title="{attr(m["knowledge_title"])}">')
        text = "\n...\n".join(line["text"] for line in m["lines"])
        handle = chunk_handle(tc, {"id": m["chunk_id"], "knowledge_id": m["knowledge_id"],
                                   "knowledge_title": m["knowledge_title"], "knowledge_base_id": m["knowledge_base_id"],
                                   "content": text, "score": None, "chunk_index": m["chunk_index"],
                                   "chunk_type": m["chunk_type"], "deprecated": m["deprecated"]})
        out.append(f'<match id="{handle}" chunk_index="{m["chunk_index"]}">\n{text}\n</match>')
    out.append("</doc>")
    if found["limit_reached"]:
        out.append("[limit reached: narrow the pattern or the documents]")
    out.append("</grep>")
    return tc.clip("\n".join(out))


# --- list_documents -----------------------------------------------------------------------------------
def _status(item: Knowledge) -> str:
    now = utcnow()
    if item.deprecated_at is not None:
        return "deprecated"
    live = (item.enable_status == "enabled" and item.parse_status == "completed"
            and (item.valid_from is None or item.valid_from <= now)
            and (item.valid_until is None or item.valid_until > now)
            and (item.expires_at is None or item.expires_at > now))
    return "live" if live else "not-live"


async def list_documents(tc: ToolContext, args: dict) -> str:
    from agenticwf.kb.runtime.services.tags import item_tag_clause, resolve_tags

    kb_ids = tc.kb_ids(list_arg(args, "knowledge_base_ids"))
    page = int_arg(args, "page", 1, 1, 10_000)
    async with async_session_scope(commit=False) as session:
        query = sa.select(Knowledge).where(Knowledge.deleted_at.is_(None), Knowledge.type != "faq",
                                           access.items_clause(Knowledge, tc.auth, kb_ids))
        if tc.scope.knowledge_ids:
            query = query.where(Knowledge.id.in_(tc.scope.knowledge_ids))
        keyword = str(args.get("keyword") or "").strip()
        if keyword:
            query = query.where(sa.or_(Knowledge.title.ilike(f"%{keyword}%"), Knowledge.file_name.ilike(f"%{keyword}%")))
        folder = str(args.get("folder") or "").strip()
        if folder:
            query = query.where(Knowledge.folder_path.like(folder.rstrip("/") + "%"))
        for refs in (list(tc.scope.tag_ids), list_arg(args, "tag_ids")):
            if refs:
                ids, untagged = await resolve_tags(session, kb_ids, refs)
                query = query.where(item_tag_clause(ids, untagged))
        total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
        items = (await session.execute(query.order_by(Knowledge.updated_at.desc()).offset((page - 1) * PAGE)
                                       .limit(PAGE))).scalars().all()
        counts = dict((await session.execute(sa.select(Chunk.knowledge_id, sa.func.count()).where(
            Chunk.knowledge_id.in_([i.id for i in items]), Chunk.chunk_type == "text").group_by(
            Chunk.knowledge_id))).all()) if items else {}
    if not items:
        return "No documents." if page == 1 else f"No documents on page {page} (total {total})."
    kb_handles = {kb_id: tc.handles.issue("b", kb_id, {"name": tc.scope.kbs[kb_id].name}) for kb_id in kb_ids}
    out = [f'<documents total="{total}" page="{page}">']
    for item in items:
        handle = doc_handle(tc, item.id, item.title, item.knowledge_base_id)
        out.append(f'<document id="{handle}" kb="{kb_handles.get(item.knowledge_base_id, "")}" '
                   f'title="{attr(item.title)}" type="{attr(item.file_type or item.type)}" '
                   f'folder="{attr(item.folder_path)}" chunks="{counts.get(item.id, 0)}" version="{item.version}" '
                   f'updated="{item.updated_at.date().isoformat() if item.updated_at else ""}" '
                   f'status="{_status(item)}" parse_status="{item.parse_status}">'
                   f'{attr((item.description or "")[:300])}</document>')
    out.append("</documents>")
    if page * PAGE < total:
        out.append(f"[more documents: call again with page={page + 1}]")
    return tc.clip("\n".join(out))


# --- read_document ------------------------------------------------------------------------------------
def segments(chunks: list[Chunk]) -> tuple[str, list[tuple[int, int, Chunk]]]:
    """The document rebuilt from its chunks without the overlap: (text, [(start, end, chunk)])."""
    parts, spans, position, prev_end = [], [], 0, None
    for chunk in chunks:
        text = chunk.content
        if prev_end is not None and chunk.start_at < prev_end and chunk.end_at > chunk.start_at:
            text = text[min(len(text), prev_end - chunk.start_at):]
        elif prev_end is not None:
            text = "\n" + text
        if not text:
            continue
        parts.append(text)
        spans.append((position, position + len(text), chunk))
        position += len(text)
        prev_end = max(prev_end or 0, chunk.end_at)
    return "".join(parts), spans


def passages(tc: ToolContext, item: Knowledge, text: str, spans, ranges: list[tuple[int, int]]) -> list[str]:
    out = []
    for start, end in ranges:
        for s, e, chunk in spans:
            if e <= start or s >= end:
                continue
            piece = text[max(s, start):min(e, end)].strip("\n")
            if not piece:
                continue
            handle = chunk_handle(tc, {"id": chunk.id, "knowledge_id": item.id, "knowledge_title": item.title,
                                       "knowledge_base_id": item.knowledge_base_id, "content": chunk.content,
                                       "score": None, "chunk_index": chunk.chunk_index, "chunk_type": chunk.chunk_type})
            out.append(f'<passage id="{handle}">\n{piece}\n</passage>')
    return out


async def read_document(tc: ToolContext, args: dict) -> str:
    handle = tc.handles.get(str(args.get("id") or ""), ("d", "c"))
    knowledge_id = handle.key if handle.prefix == "d" else handle.payload.get("knowledge_id")
    async with async_session_scope(commit=False) as session:
        item = await session.get(Knowledge, knowledge_id)
        if (item is None or item.deleted_at is not None or item.knowledge_base_id not in tc.scope.kb_ids
                or not access.can_read_item(tc.auth, item)
                or (tc.scope.knowledge_ids and item.id not in tc.scope.knowledge_ids)):
            raise ToolError("document not found in this search's scope")
        session.expunge(item)
        chunks = list((await session.execute(sa.select(Chunk).where(
            Chunk.knowledge_id == item.id, Chunk.is_enabled, Chunk.chunk_type.in_(READABLE_TYPES),
            Chunk.parent_chunk_id == "").order_by(Chunk.chunk_index))).scalars())
        for chunk in chunks:
            session.expunge(chunk)
    if not chunks:
        raise ToolError("this document has no readable text yet")
    text, spans = segments(chunks)
    limit = int_arg(args, "limit", READ_LIMIT, 200, READ_LIMIT)
    doc = doc_handle(tc, item.id, item.title, item.knowledge_base_id)
    header = (f'<document id="{doc}" title="{attr(item.title)}" version="{item.version}" '
              f'content_length="{len(text)}"')
    query = str(args.get("query") or "")
    if query:
        try:
            matcher = re.compile(query if args.get("regex") else re.escape(query), re.IGNORECASE)
        except re.error:
            matcher = re.compile(re.escape(query), re.IGNORECASE)
        windows: list[list[int]] = []
        for match in list(matcher.finditer(text))[:MAX_WINDOWS]:
            start, end = max(0, match.start() - WINDOW), min(len(text), match.end() + WINDOW)
            if windows and start <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], end)
            else:
                windows.append([start, end])
        if not windows:
            return f'No match for {query!r} in this document. Try search_knowledge.'
        chosen, used = [], 0
        for start, end in windows:
            if used >= limit:
                break
            end = min(end, start + (limit - used))
            chosen.append((start, end))
            used += end - start
        body = passages(tc, item, text, spans, chosen)
        return tc.clip(f'{header} query="{attr(query)}" windows="{len(chosen)}" returned_chars="{used}">\n'
                       + "\n".join(body) + "\n</document>")
    if handle.prefix == "c":
        context = int_arg(args, "context", 2, 0, 10)
        index = next((i for i, (_, _, c) in enumerate(spans) if c.id == handle.key), None)
        if index is None:          # a chunk that is not part of the readable text (summary, image, faq)
            content = handle.payload.get("content", "")
            return tc.clip(f'{header} offset="0" returned_chars="{len(content)}">\n'
                           f'<passage id="{handle.handle}">\n{content}\n</passage>\n</document>')
        lo, hi = max(0, index - context), min(len(spans) - 1, index + context)
        while spans[hi][1] - spans[lo][0] > limit and (lo < index or hi > index):
            if index - lo >= hi - index:          # drop the farther neighbour first
                lo += 1
            else:
                hi -= 1
        start = spans[lo][0]
        end = min(spans[hi][1], start + limit)
        body = passages(tc, item, text, spans, [(start, end)])
        return tc.clip(f'{header} offset="{start}" returned_chars="{end - start}" '
                       f'next_offset="{end if end < len(text) else ""}">\n' + "\n".join(body) + "\n</document>")
    offset = int_arg(args, "offset", 0, 0, max(0, len(text)))
    end = min(len(text), offset + limit)
    # end at a chunk boundary when one falls inside the window
    boundary = max((e for s, e, _ in spans if offset < e <= end), default=end)
    end = boundary if boundary > offset else end
    body = passages(tc, item, text, spans, [(offset, end)])
    return tc.clip(f'{header} offset="{offset}" returned_chars="{end - offset}" '
                   f'next_offset="{end if end < len(text) else ""}">\n' + "\n".join(body) + "\n</document>")


KB_ARGS = {"knowledge_base_ids": {"type": "array", "items": {"type": "string"},
                                  "description": "bN handles; default: every knowledge base in scope"},
           "document_ids": {"type": "array", "items": {"type": "string"}, "description": "dN handles"}}

TOOLS = [
    Tool("search_knowledge",
         "Hybrid (semantic + keyword) search over the knowledge bases. Returns passages as <result id=\"cN\" "
         "doc=\"dN\" …>. Give several phrasings in `queries`; use mode=keyword for exact terms.",
         {"type": "object", "properties": {
             "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
             "mode": {"type": "string", "enum": ["hybrid", "keyword", "semantic"]},
             **KB_ARGS, "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["queries"]},
         search_knowledge),
    Tool("grep_knowledge",
         "Exact text search (regex or literal) over passage text, for identifiers, error codes, versions or "
         "every place something is mentioned. Returns matching lines grouped by document.",
         {"type": "object", "properties": {
             "pattern": {"type": "string", "maxLength": 500}, "regex": {"type": "boolean"},
             "case_sensitive": {"type": "boolean"}, **KB_ARGS,
             "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["pattern"]},
         grep_knowledge),
    Tool("list_documents",
         "List documents (newest first, 30 per page) with title, type, folder, version and status.",
         {"type": "object", "properties": {
             "knowledge_base_ids": KB_ARGS["knowledge_base_ids"], "keyword": {"type": "string"},
             "folder": {"type": "string"}, "tag_ids": {"type": "array", "items": {"type": "string"}},
             "page": {"type": "integer", "minimum": 1}}},
         list_documents),
    Tool("read_document",
         "Read a document (dN, from offset) or the passage around a chunk (cN, with `context` neighbours). "
         "With `query`, returns only the regions that match. Every passage carries a citable cN.",
         {"type": "object", "properties": {
             "id": {"type": "string", "description": "dN or cN"}, "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 200, "maximum": READ_LIMIT}, "query": {"type": "string"},
             "regex": {"type": "boolean"}, "context": {"type": "integer", "minimum": 0, "maximum": 10}},
          "required": ["id"]},
         read_document),
]

