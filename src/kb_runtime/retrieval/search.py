"""
Hybrid search (docs/retrieval.md, Hybrid search).

    resolve KBs → embed the query once per embedding model → vector (pgvector HNSW) and keyword
    (pg_search BM25) with the same SQL filters → weighted RRF → load rows, deprecated down-rank,
    FAQ negative questions → optional rerank (degradation, composite, MMR) → FAQ priority → top_k

The filters run inside both queries: readable KBs, the item's own visibility (owner / all /
groups), enabled, validity window, deprecated (excluded unless the KB down-ranks), knowledge
ids. So hidden items never cost results after the top-k cut.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.kb import Chunk, Knowledge, KnowledgeBase, utcnow
from kb_runtime import access
from kb_runtime.citation import locate
from kb_runtime.faq.rules import apply_priority, drop_negatives
from kb_runtime.models.clients import ModelError
from kb_runtime.retrieval.fusion import rrf
from kb_runtime.retrieval.rerank import rerank
from kb_runtime.services.knowledge_bases import KnowledgeBaseService

LOG = get_logger(__name__)

DEFAULTS = {"embedding_top_k": 30, "vector_threshold": 0.2, "keyword_threshold": 0.3, "rrf_k": 60,
            "vector_weight": 0.7, "keyword_weight": 0.3, "rerank_top_k": 30, "rerank_threshold": 0.3,
            "enable_rerank": True}
_WORD = re.compile(r"\w", re.UNICODE)


def retrieval_defaults() -> dict:
    out = dict(DEFAULTS)
    for key, value in DEFAULTS.items():
        if config_has_option("retrieval", key):
            raw = config_get("retrieval", key)
            out[key] = raw.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, bool) else type(value)(raw)
    return out


@dataclass
class SearchOptions:
    vector_threshold: float | None = None
    keyword_threshold: float | None = None
    top_k: int | None = None
    knowledge_ids: list[str] = field(default_factory=list)
    tag_ids: list[str] = field(default_factory=list)
    disable_vector: bool = False
    disable_keywords: bool = False
    enable_rerank: bool | None = None


_FILTERS = """
  knowledge_base_id = ANY(:kbs) AND is_enabled
  AND (owner_subject = :subject OR visibility = 'all' OR (visibility = 'groups' AND groups && CAST(:groups AS text[])))
  AND (valid_from IS NULL OR valid_from <= now()) AND (valid_until IS NULL OR valid_until > now())
  AND (NOT is_deprecated OR knowledge_base_id = ANY(:downrank_kbs))
  {knowledge_filter}
"""


def _filters(options: SearchOptions, tags: tuple[list[str], bool] = ([], False)) -> str:
    from kb_runtime.services.tags import SEARCH_TAG_SQL, UNTAGGED_SQL

    extra = "AND knowledge_id = ANY(:knowledge_ids)" if options.knowledge_ids else ""
    tag_ids, untagged = tags
    if tag_ids or untagged:
        extra += "\n  " + SEARCH_TAG_SQL.format(untagged=UNTAGGED_SQL if untagged else "")
    return _FILTERS.format(knowledge_filter=extra)


def recency_weight(kb: KnowledgeBase, date) -> float | None:
    """max(floor, 0.5^(age / half-life)) when the KB sets `recency_half_life_days` (docs/retrieval.md)."""
    cfg = kb.freshness_config or {}
    half_life = float(cfg.get("recency_half_life_days") or 0)
    if half_life <= 0 or date is None:
        return None
    age_days = max(0.0, (utcnow() - date).total_seconds() / 86400)
    return max(float(cfg.get("recency_floor", 0.2)), 0.5 ** (age_days / half_life))


def chunk_citation(item, chunk) -> dict:
    """The source citation of a passage: its document's citation, with section and page."""
    meta = chunk.meta or {}
    page, page_end = meta.get("page"), meta.get("page_end")
    section = chunk.context_header or meta.get("section") or None
    base = dict(item.citation or {}) or {"title": item.title, "text": item.title}
    return {"citation": {**base, **({"section": section} if section else {}), **({"page": page} if page else {}),
                         **({"page_end": page_end} if page_end else {})},
            "citation_text": locate(base, section, page, page_end)}


def record_citation(record, kb) -> dict:
    name = kb.name if kb is not None else record.knowledge_base_id
    text = f"{record.name} ({record.record_type}, v{record.version}), {name}"
    return {"citation": {"title": record.name, "record_type": record.record_type, "knowledge_base": name,
                         "version": str(record.version), "text": text}, "citation_text": text}


def record_hit(record, f, kbs: dict) -> dict:
    """A structured record as a search result (`chunk_type = record`): a compact view of its
    name, type, facets and measures, never the raw JSON (docs/structured-records.md)."""
    score = f.score
    if record.deprecated_at is not None:
        score *= 0.5
    date = record.event_time or record.updated_at
    recency = recency_weight(kbs[record.knowledge_base_id], date) if record.knowledge_base_id in kbs else None
    if recency is not None:
        score *= recency
    lines = [f"{record.name} ({record.record_type})"]
    lines += [f"{k}: {v}" for k, v in (record.facets or {}).items()]
    lines += [f"{k}: {v:g}" for k, v in (record.measures or {}).items()]
    if record.text:
        lines.append(record.text)
    content = "\n".join(lines)
    return {"id": record.id, "content": content, "matched_content": content, "knowledge_id": "",
            "knowledge_title": record.name, "knowledge_description": "", "knowledge_filename": "",
            "knowledge_source": "record", "knowledge_base_id": record.knowledge_base_id, "chunk_index": 0,
            "seq": None, "start_at": 0, "end_at": len(content), "chunk_type": "record", "parent_chunk_id": "",
            "pre_chunk_id": "", "next_chunk_id": "", "sub_chunk_id": [], "context_header": "", "score": score,
            "match_type": f.match_type, "metadata": {"record_type": record.record_type, "external_id": record.external_id},
            "image_info": [], "record_id": record.id, "record_type": record.record_type, "record_name": record.name,
            "knowledge_date": date.isoformat() if date else None, "knowledge_version": record.version,
            "deprecated": record.deprecated_at is not None, **record_citation(record, kbs.get(record.knowledge_base_id)),
            **({"recency_weight": recency} if recency is not None else {})}


def wiki_hit(page, content: str, f) -> dict:
    """A wiki passage as a search result (`chunk_type = wiki_page`, docs/llm-wiki.md)."""
    index = int(f.chunk_id.rsplit(":", 1)[1])
    return {"id": f.chunk_id, "content": content, "matched_content": content, "knowledge_id": "",
            "knowledge_title": page.title, "knowledge_description": page.summary, "knowledge_filename": "",
            "knowledge_source": "wiki", "knowledge_base_id": page.knowledge_base_id, "chunk_index": index, "seq": None,
            "start_at": 0, "end_at": len(content), "chunk_type": "wiki_page", "parent_chunk_id": "", "pre_chunk_id": "",
            "next_chunk_id": "", "sub_chunk_id": [], "context_header": page.category_path, "score": f.score,
            "match_type": f.match_type, "metadata": {"slug": page.slug, "page_type": page.page_type},
            "image_info": [], "wiki_slug": page.slug, "wiki_page_id": page.id,
            "knowledge_date": page.updated_at.isoformat() if page.updated_at else None,
            "knowledge_version": page.version, "deprecated": False,
            "citation": {"title": page.title, "wiki_slug": page.slug, "sources": [r.split("|", 1)[-1] for r in page.source_refs or []],
                         "text": f"Wiki: {page.title}"},
            "citation_text": f"Wiki: {page.title}" + (f" (compiled from: {'; '.join(r.split('|', 1)[-1] for r in (page.source_refs or [])[:5])})"
                                                       if page.source_refs else "")}


class SearchService:
    def __init__(self, models, kbs: KnowledgeBaseService):
        self.models = models
        self.kbs = kbs

    async def _kbs(self, ctx: AuthContext, refs: list[str] | None) -> list[KnowledgeBase]:
        async with async_session_scope(commit=False) as session:
            ids = await self.kbs.readable_ids(session, ctx, refs)
            if not ids:
                return []
            rows = (await session.execute(sa.select(KnowledgeBase).where(KnowledgeBase.id.in_(ids)))).scalars().all()
            for row in rows:
                session.expunge(row)
            return list(rows)

    async def search(self, ctx: AuthContext, query: str, kb_refs: list[str] | None = None,
                     options: SearchOptions | None = None) -> list[dict]:
        options = options or SearchOptions()
        query = (query or "").strip()
        if not query:
            return []
        kbs = await self._kbs(ctx, kb_refs)
        if not kbs:
            return []
        cfg = retrieval_defaults()
        if len(kbs) == 1:
            cfg.update({k: v for k, v in (kbs[0].retrieval_config or {}).items() if k in DEFAULTS})
        vector_threshold = cfg["vector_threshold"] if options.vector_threshold is None else options.vector_threshold
        keyword_threshold = cfg["keyword_threshold"] if options.keyword_threshold is None else options.keyword_threshold
        top_k = max(1, min(int(options.top_k or cfg["rerank_top_k"]), 200))
        candidates = max(int(cfg["embedding_top_k"]), top_k)
        tags = ([], False)
        if options.tag_ids:
            from kb_runtime.services.tags import resolve_tags

            async with async_session_scope(commit=False) as session:
                tags = await resolve_tags(session, [kb.id for kb in kbs], options.tag_ids)
            if not tags[0] and not tags[1]:
                return []
        params = {"kbs": [kb.id for kb in kbs], "subject": ctx.subject, "groups": access.principal_groups(ctx),
                  "tag_ids": tags[0],
                  "downrank_kbs": [kb.id for kb in kbs if (kb.freshness_config or {}).get("deprecated_mode") == "downrank"],
                  "knowledge_ids": list(options.knowledge_ids), "k": candidates}

        vector_hits: list[tuple[str, float]] = []
        keyword_hits: list[tuple[str, float]] = []
        jobs = []
        if not options.disable_vector:
            by_model: dict[str, list[str]] = {}
            for kb in kbs:
                if kb.embedding_model_id and kb.indexing_strategy.get("vector_enabled", True):
                    by_model.setdefault(kb.embedding_model_id, []).append(kb.id)
            for model_id, kb_ids in by_model.items():
                jobs.append(self._vector(query, model_id, {**params, "kbs": kb_ids}, options, vector_threshold, tags))
        keyword_kbs = [kb.id for kb in kbs if kb.indexing_strategy.get("keyword_enabled", True)]
        if not options.disable_keywords and keyword_kbs and _WORD.search(query):
            jobs.append(self._keyword(query, {**params, "kbs": keyword_kbs}, options, keyword_threshold, tags))
        for kind, hits in await asyncio.gather(*jobs):
            (vector_hits if kind == "vector" else keyword_hits).extend(hits)
        vector_hits.sort(key=lambda h: -h[1])
        fused = rrf(vector_hits, keyword_hits, int(cfg["rrf_k"]), float(cfg["vector_weight"]),
                    float(cfg["keyword_weight"]))
        if not fused:
            return []
        items = drop_negatives(query, await self._load(fused[:max(candidates, top_k)], {kb.id: kb for kb in kbs}))
        has_faq = any(item["chunk_type"] == "faq" for item in items)

        use_rerank = cfg["enable_rerank"] if options.enable_rerank is None else options.enable_rerank
        scorer = None
        if use_rerank:
            model_id = next((kb.rerank_model_id for kb in kbs if kb.rerank_model_id), None) or \
                await self.models.default_id("Rerank")
            if model_id:
                try:
                    scorer = (await self.models.reranker(model_id)).rerank
                except ModelError as exc:
                    LOG.warning("rerank model unavailable, keeping retrieval order: %s", exc)
        # FAQ priority orders by a boosted score before the top-k cut; greedy MMR's first top_k
        # picks do not depend on how many it is asked for, so rerank everything, then cut.
        results = await rerank(query, items, scorer, len(items) if has_faq else top_k,
                               float(cfg["rerank_threshold"]), scoped=bool(options.knowledge_ids))
        if has_faq:
            results = apply_priority(results, top_k)
        self._count_retrievals([r["knowledge_id"] for r in results])
        return results

    async def _vector(self, query, model_id, params, options, threshold, tags):
        try:
            embedder = await self.models.embedder(model_id)
            vector = (await embedder.embed([query]))[0]
        except ModelError as exc:
            LOG.warning("query embedding failed (%s); keyword results only", exc)
            return "vector", []
        dim = int(embedder.dimension)
        sql = sa.text(f"""
            SELECT chunk_id, 1 - (embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim}))) AS score
            FROM kb_embeddings
            WHERE dimension = {dim} AND {_filters(options, tags)}
            ORDER BY embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim}))
            LIMIT :k""")
        async with async_session_scope(commit=False) as session:
            await session.execute(sa.text(f"SET LOCAL hnsw.ef_search = {max(40, 2 * params['k'])}"))
            await session.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            rows = (await session.execute(sql, {**params, "q": "[" + ",".join(f"{v:.7g}" for v in vector) + "]"})).all()
        return "vector", [(cid, float(score)) for cid, score in rows if score is not None and score >= threshold]

    async def _keyword(self, query, params, options, threshold, tags):
        sql = sa.text(f"""
            SELECT chunk_id, paradedb.score(id) AS score FROM kb_embeddings
            WHERE id @@@ paradedb.match('content', :query) AND {_filters(options, tags)}
            ORDER BY score DESC LIMIT :k""")
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sql, {**params, "query": query})).all()
        if not rows:
            return "keyword", []
        best = max(float(score) for _, score in rows) or 1.0
        return "keyword", [(cid, float(score) / best) for cid, score in rows if float(score) / best >= threshold]

    async def _load(self, fused, kbs: dict[str, KnowledgeBase]) -> list[dict]:
        ids = [f.chunk_id for f in fused]
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(
                sa.select(Chunk, Knowledge).join(Knowledge, Knowledge.id == Chunk.knowledge_id)
                .where(Chunk.id.in_(ids), Knowledge.deleted_at.is_(None)))).all()
        by_id = {chunk.id: (chunk, item) for chunk, item in rows}
        records = await self._load_records([i for i in ids if i not in by_id and not i.startswith("wiki:")])
        wiki = await self._load_wiki([i for i in ids if i.startswith("wiki:")])
        items = []
        for f in fused:
            if f.chunk_id in records:
                items.append(record_hit(records[f.chunk_id], f, kbs))
                continue
            if f.chunk_id in wiki:
                page, content = wiki[f.chunk_id]
                items.append(wiki_hit(page, content, f))
                continue
            if f.chunk_id not in by_id:
                continue
            chunk, item = by_id[f.chunk_id]
            score = f.score
            deprecated = item.deprecated_at is not None
            if deprecated and (kbs[item.knowledge_base_id].freshness_config or {}).get("deprecated_mode") == "downrank":
                score *= 0.5
            date = item.valid_from or item.processed_at or item.updated_at
            recency = recency_weight(kbs[item.knowledge_base_id], date)
            if recency is not None:
                score *= recency
            items.append({
                "id": chunk.id, "content": chunk.content, "matched_content": chunk.content,
                "knowledge_id": item.id, "knowledge_title": item.title, "knowledge_description": item.description,
                "knowledge_filename": item.file_name, "knowledge_source": item.source if item.type == "url" else item.type,
                "knowledge_base_id": item.knowledge_base_id, "chunk_index": chunk.chunk_index, "seq": chunk.seq_id,
                "start_at": chunk.start_at, "end_at": chunk.end_at, "chunk_type": chunk.chunk_type,
                "parent_chunk_id": chunk.parent_chunk_id, "pre_chunk_id": chunk.pre_chunk_id,
                "next_chunk_id": chunk.next_chunk_id, "sub_chunk_id": [], "context_header": chunk.context_header,
                "score": score, "match_type": f.match_type, "metadata": chunk.meta, "image_info": chunk.image_info,
                "knowledge_date": date.isoformat() if date else None, "knowledge_version": item.version,
                "deprecated": deprecated, **({"recency_weight": recency} if recency is not None else {}),
                **chunk_citation(item, chunk)})
        items.sort(key=lambda i: -i["score"])
        return items

    @staticmethod
    async def _load_records(ids: list[str]) -> dict:
        """Record hits (text roles indexed with source_type 5, chunk_id = record id)."""
        if not ids:
            return {}
        from kb_core.orm.models.ext import Record

        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(Record).where(Record.id.in_(ids),
                                                                  Record.deleted_at.is_(None)))).scalars().all()
        return {r.id: r for r in rows}

    @staticmethod
    async def _load_wiki(ids: list[str]) -> dict:
        """Wiki passages (source_type 4, chunk_id `wiki:<page id>:<n>`): published pages only."""
        if not ids:
            return {}
        from kb_core.orm.models.ext import WikiPage
        from kb_core.orm.models.kb import Embedding

        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(Embedding.chunk_id, Embedding.content).where(
                Embedding.chunk_id.in_(ids)))).all()
            page_ids = {cid.split(":")[1] for cid, _ in rows}
            pages = {p.id: p for p in (await session.execute(sa.select(WikiPage).where(
                WikiPage.id.in_(page_ids), WikiPage.deleted_at.is_(None), WikiPage.status == "published"))).scalars()}
        return {cid: (pages[cid.split(":")[1]], content) for cid, content in rows if cid.split(":")[1] in pages}

    def _count_retrievals(self, knowledge_ids: list[str]) -> None:
        """retrieval_count / last_retrieved_at, outside the request (never in its transaction)."""
        ids = list(dict.fromkeys(k for k in knowledge_ids if k))
        if not ids:
            return

        async def update():
            try:
                async with async_session_scope() as session:
                    await session.execute(sa.update(Knowledge).where(Knowledge.id.in_(ids)).values(
                        retrieval_count=Knowledge.retrieval_count + 1, last_retrieved_at=utcnow()))
            except Exception:
                LOG.debug("retrieval counters not updated", exc_info=True)

        task = asyncio.create_task(update())
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)


_BACKGROUND: set[asyncio.Task] = set()
