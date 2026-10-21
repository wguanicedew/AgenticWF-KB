"""
KnowledgeService: knowledge items (file, URL, manual), their processing and their chunks
(docs/ingestion.md; docs/api.md, Knowledge and Chunks).

Adding an item checks that the caller may contribute to the KB, resolves the item's visibility
(the request's, else the KB's default; always private in a private KB), stores the original and
enqueues `document:process` (or `manual:process`) in the same transaction.

Processing parses (in a thread), chunks, embeds (outside any transaction), then swaps the item's
chunks and index rows in one transaction, so the old content stays searchable until the new is
committed. Every index row copies the item's owner, visibility, groups, enable status and
validity, so search filters on them in SQL.
"""

from __future__ import annotations

import datetime
import json
import bisect
import hashlib
import os
from typing import Any

import anyio
import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope, get_engine
from kb_core.orm.models.indexes import ensure_vector_index
from kb_core.orm.models.kb import (SOURCE_CHUNK, Chunk, Embedding, Knowledge, KnowledgeBase, KnowledgeVersion,
                                   new_id, utcnow)
from kb_runtime import access, citation
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound
from kb_runtime.ingestion import chunker, parsers
from kb_runtime.ingestion.parsers import ParseError
from kb_runtime.models.clients import ModelError
from kb_runtime.security.ssrf import FetchError, fetch
from kb_runtime.services.knowledge_bases import KnowledgeBaseService
from kb_runtime.storage import LocalStorage, owner_scope, safe_name
from kb_runtime.tasks import queue
from kb_runtime.tasks.runner import HandlerSpec, Permanent, TaskContext

LOG = get_logger(__name__)

DOCUMENT_TIMEOUT = 1800
MAX_URL_BYTES = 20 * 1024 * 1024
CONTENT_PAGE = 8000
CONTENT_TYPES = {"text/html": ".html", "application/xhtml+xml": ".html", "application/pdf": ".pdf",
                 "text/plain": ".txt", "text/markdown": ".md", "text/csv": ".csv", "application/json": ".json",
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                 "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx"}
ITEM_FIELDS = ("title", "description", "external_id", "refresh_interval")


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _date(value) -> datetime.datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime.datetime):
        parsed = value
    else:
        try:
            parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise Invalid(f"not an ISO 8601 date: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def knowledge_dict(k: Knowledge, tag_ids: list[str] | None = None, superseded_by: list[str] | None = None) -> dict:
    return {"id": k.id, "knowledge_base_id": k.knowledge_base_id, "type": k.type, "title": k.title,
            "description": k.description, "source": k.source, "channel": k.channel, "parse_status": k.parse_status,
            "summary_status": k.summary_status, "enable_status": k.enable_status, "error_message": k.error_message,
            "file_name": k.file_name, "file_type": k.file_type, "file_size": k.file_size, "file_hash": k.file_hash,
            "storage_size": k.storage_size, "metadata": k.meta, "embedding_model_id": k.embedding_model_id,
            "owner_subject": k.owner_subject, "visibility": k.visibility, "groups": list(k.groups or []),
            "tag_ids": list(tag_ids or []), "superseded_by": list(superseded_by or []),
            "citation": k.citation or {}, "external_id": k.external_id, "version": k.version, "content_hash": k.content_hash,
            "valid_from": _iso(k.valid_from), "valid_until": _iso(k.valid_until), "expires_at": _iso(k.expires_at),
            "deprecated": k.deprecated_at is not None, "refresh_interval": k.refresh_interval,
            "retrieval_count": k.retrieval_count, "last_retrieved_at": _iso(k.last_retrieved_at),
            "processed_at": _iso(k.processed_at), "created_at": _iso(k.created_at), "updated_at": _iso(k.updated_at)}


def chunk_dict(c: Chunk) -> dict:
    return {"id": c.id, "seq_id": c.seq_id, "knowledge_id": c.knowledge_id, "knowledge_base_id": c.knowledge_base_id,
            "content": c.content, "chunk_index": c.chunk_index, "chunk_type": c.chunk_type, "start_at": c.start_at,
            "end_at": c.end_at, "pre_chunk_id": c.pre_chunk_id, "next_chunk_id": c.next_chunk_id,
            "parent_chunk_id": c.parent_chunk_id, "context_header": c.context_header, "is_enabled": c.is_enabled,
            "content_revision": c.content_revision, "metadata": c.meta, "image_info": c.image_info,
            "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at)}


def index_text(title: str, context_header: str, content: str) -> str:
    return (title + "\n" if title else "") + (context_header + "\n\n" if context_header else "") + content.strip()


async def event(session, item: Knowledge, payload: dict) -> None:
    await queue.write_event(session, item.knowledge_base_id, item.owner_subject,
                            {"knowledge_id": item.id, **payload}, item.visibility, list(item.groups or []))


def _valid_until(k: Knowledge):
    dates = [d for d in (k.valid_until, k.expires_at) if d is not None]
    return min(dates) if dates else None


def _location(pages: list[int], headings: list, piece) -> dict:
    """Where a chunk is in its document, for citations: page (PDF, slides) and section."""
    out = {}
    if pages:
        out["page"] = bisect.bisect_right(pages, piece.start)
        last = bisect.bisect_right(pages, max(piece.start, piece.end - 1))
        if last > out["page"]:
            out["page_end"] = last
    section = citation.section_at(headings, piece.start, piece.content) if headings else None
    if section:
        out["section"] = section
    return out


def _max_file_bytes() -> int:
    mb = int(config_get("storage", "max_file_mb")) if config_has_option("storage", "max_file_mb") else 50
    return mb * 1024 * 1024


def _default_strategy() -> str:
    return config_get("ingestion", "strategy") if config_has_option("ingestion", "strategy") else "auto"


class KnowledgeService:
    def __init__(self, models, storage: LocalStorage, kbs: KnowledgeBaseService):
        self.models = models
        self.storage = storage
        self.kbs = kbs
        self.postprocess = None          # EnrichmentService: subtasks after the swap (phase 2)
        self.tags = None                 # TagService
        self.wiki = None                 # WikiService (phase 3): retraction on delete / visibility change

    # -- task handlers ---------------------------------------------------------------------------------
    def handlers(self) -> dict[str, HandlerSpec]:
        spec = HandlerSpec(run=self._process_task, timeout=DOCUMENT_TIMEOUT, on_final_failure=self._process_failed)
        return {"document:process": spec, "manual:process": spec}

    # -- loading with access checks ---------------------------------------------------------------
    async def _load(self, session, ctx: AuthContext, knowledge_id: str, for_update: bool = False):
        """(item, kb) if the principal can read the item, else NotFound."""
        query = sa.select(Knowledge).where(Knowledge.id == knowledge_id, Knowledge.deleted_at.is_(None))
        if for_update:
            query = query.with_for_update()
        item = (await session.execute(query)).scalar_one_or_none()
        if item is None:
            raise NotFound("knowledge not found")
        kb = await session.get(KnowledgeBase, item.knowledge_base_id)
        if kb is None or kb.deleted_at is not None or not access.can_see(ctx, kb) or not access.can_read_item(ctx, item):
            raise NotFound("knowledge not found")
        return item, kb

    async def _managed(self, session, ctx, knowledge_id, for_update=True):
        item, kb = await self._load(session, ctx, knowledge_id, for_update=for_update)
        if not access.can_manage_item(ctx, item, kb):
            raise Forbidden("only the item's owner, the KB owner or an admin changes it")
        return item, kb

    def _share(self, ctx: AuthContext, kb: KnowledgeBase, visibility: str | None, groups) -> tuple[str, list[str]]:
        visibility = "private" if kb.visibility == "private" else (visibility or kb.default_item_visibility)
        return visibility, access.check_share(ctx, visibility, groups or [])

    async def _contribute(self, session, ctx, kb_ref) -> KnowledgeBase:
        kb = await self.kbs.load(session, ctx, kb_ref, for_update=True)
        if not access.can_contribute(ctx, kb):
            raise Forbidden("you may not add knowledge to this knowledge base")
        if kb.type != "document":
            raise Invalid(f"a {kb.type} knowledge base does not take documents")
        return kb

    async def _enqueue(self, session, item: Knowledge, task_type: str = "document:process") -> None:
        await queue.enqueue(session, task_type, {"knowledge_id": item.id, "knowledge_base_id": item.knowledge_base_id},
                            owner_subject=item.owner_subject, owner_org=item.owner_org,
                            knowledge_base_id=item.knowledge_base_id, unique_key=f"process:{item.id}", max_retry=3)
        await event(session, item, {"type": "knowledge_status", "parse_status": item.parse_status})

    async def _tags(self, session, ctx, item: Knowledge, tag_refs) -> list[str]:
        if tag_refs is None or self.tags is None:
            return (await self.tags.item_tags(session, [item.id])).get(item.id, []) if self.tags else []
        if isinstance(tag_refs, str):
            tag_refs = [t.strip() for t in tag_refs.split(",") if t.strip()]
        return await self.tags.set_item_tags(session, ctx, item, list(tag_refs))

    # -- source citations (docs/ingestion.md, Source citations) -----------------------------------
    @staticmethod
    def _supplied_citation(fields: dict) -> dict | None:
        """The citation a caller gave at injection: a `citation` object (or JSON string) and the
        flat fields `citation_text`, `authors`, `published`, `source_url`, `doi`, `publisher`,
        `version`. None when nothing was given."""
        raw = fields.get("citation")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except ValueError:
                raise Invalid("citation must be JSON")
        flat = {k: fields[k] for k in ("citation_text", "authors", "published", "source_url", "doi", "publisher",
                                       "version") if fields.get(k) not in (None, "")}
        if not raw and not flat:
            return None
        try:
            return citation.normalise_supplied({**(raw or {}), **flat})
        except ValueError as exc:
            raise Invalid(str(exc))

    @staticmethod
    def _apply_citation(item: Knowledge, supplied: dict | None = None, parsed: dict | None = None) -> None:
        """Resolve the item's citation: supplied > the document's own metadata > defaults."""
        meta = dict(item.meta or {})
        if supplied is not None:
            meta["citation_supplied"] = supplied
        if parsed is not None:
            meta["parsed_metadata"] = {k: v for k, v in parsed.items() if k in citation.FIELDS}
        defaults = {"title": item.title or os.path.splitext(item.file_name or "")[0]}
        if item.type == "url":
            defaults["url"] = item.source
            defaults["accessed"] = (item.last_refreshed_at or item.processed_at or utcnow()).date().isoformat()
        item.citation = citation.build(meta.get("citation_supplied"), meta.get("parsed_metadata"), defaults)
        item.meta = meta

    def _prefix(self, item: Knowledge) -> str:
        return f"{owner_scope(item.owner_subject)}/{item.knowledge_base_id}/{item.id}"

    # -- adding items ---------------------------------------------------------------------------------
    def _freshness(self, item: Knowledge, fields: dict) -> None:
        for name in ("valid_from", "valid_until", "expires_at"):
            if name in fields:
                setattr(item, name, _date(fields[name]))
        if "deprecated" in fields:
            item.deprecated_at = utcnow() if str(fields["deprecated"]).lower() in ("1", "true", "yes") else None

    async def create_file(self, ctx: AuthContext, kb_ref: str, file_name: str, data: bytes,
                          fields: dict | None = None) -> tuple[str, dict]:
        """Returns (outcome, item): outcome is created, replaced or unchanged."""
        fields = fields or {}
        name = safe_name(fields.get("fileName") or file_name)
        if not parsers.supported(name):
            raise Invalid(f"unsupported file type: {parsers.extension(name) or name}")
        if len(data) > _max_file_bytes():
            raise Invalid(f"the file is larger than {_max_file_bytes() // (1024 * 1024)} MB")
        if not data:
            raise Invalid("the file is empty")
        digest = hashlib.sha256(data).hexdigest()
        mode = fields.get("mode") or "create"
        if mode not in ("create", "replace"):
            raise Invalid("mode must be create or replace")
        async with async_session_scope() as session:
            kb = await self._contribute(session, ctx, kb_ref)
            visibility, groups = self._share(ctx, kb, fields.get("visibility"), fields.get("groups"))
            item = None
            if mode == "replace":
                match = [Knowledge.external_id == fields["external_id"]] if fields.get("external_id") else \
                    [Knowledge.file_name == name, Knowledge.folder_path == (fields.get("folder_path") or "")]
                candidates = (await session.execute(sa.select(Knowledge).where(
                    Knowledge.knowledge_base_id == kb.id, Knowledge.deleted_at.is_(None), *match).with_for_update()
                )).scalars().all()
                item = next((c for c in candidates if access.can_read_item(ctx, c)), None)
                if item is not None and not access.can_manage_item(ctx, item, kb):
                    raise Forbidden("only the item's owner, the KB owner or an admin replaces it")
                if item is not None and item.file_hash == digest:
                    return "unchanged", knowledge_dict(item)
            if item is None:
                duplicates = (await session.execute(sa.select(Knowledge).where(
                    Knowledge.knowledge_base_id == kb.id, Knowledge.file_hash == digest,
                    Knowledge.deleted_at.is_(None)))).scalars().all()
                readable = next((d for d in duplicates if access.can_read_item(ctx, d)), None)
                if readable is not None:     # unreadable duplicates are never compared (no leak)
                    raise Conflict("this file is already in the knowledge base", knowledge_dict(readable))
            if kb.storage_used + len(data) > kb.storage_quota:
                raise Invalid("the knowledge base's storage quota is full")
            outcome = "replaced" if item is not None else "created"
            if item is None:
                item = Knowledge(id=new_id(), knowledge_base_id=kb.id, owner_subject=ctx.subject,
                                 owner_org=ctx.organization_id, type="file", channel=ctx.auth_method == "session"
                                 and "web" or "api", visibility=visibility, groups=groups,
                                 external_id=fields.get("external_id") or None,
                                 folder_path=fields.get("folder_path") or "")
                session.add(item)
            replaced = None
            if outcome == "replaced":
                # the version being replaced; the swap transaction records it (docs/ingestion.md, Versions)
                replaced = {"file_name": item.file_name, "file_hash": item.file_hash, "file_size": item.file_size,
                            "stored_path": "", "replaced_by": ctx.subject}
                if (kb.freshness_config or {}).get("keep_previous_versions") and item.file_path:
                    kept = f"{self._prefix(item)}/versions/{item.version}/{item.file_name}"
                    old = await anyio.to_thread.run_sync(self.storage.read, item.file_path)
                    await anyio.to_thread.run_sync(self.storage.save, kept, old)
                    replaced["stored_path"] = kept
                elif item.file_path and item.file_name != name:
                    self.storage.delete(item.file_path)
                kb.storage_used = max(0, kb.storage_used - item.file_size)
            item.title = item.title or os.path.splitext(name)[0]
            item.file_name, item.file_type = name, parsers.extension(name).lstrip(".")
            item.file_size, item.file_hash, item.source = len(data), digest, name
            previous = item.meta or {}
            item.meta = {**(fields.get("metadata") or previous), **({"_replaced": replaced} if replaced else {})}
            supplied = self._supplied_citation(fields)
            if supplied is None and previous.get("citation_supplied"):     # a replacement keeps its citation
                supplied = previous["citation_supplied"]
            self._apply_citation(item, supplied or {}, {})
            item.parse_status, item.error_message = "pending", ""
            self._freshness(item, fields)
            item.file_path = f"{self._prefix(item)}/{name}"
            await anyio.to_thread.run_sync(self.storage.save, item.file_path, data)
            kb.storage_used += len(data)
            await session.flush()
            tags = await self._tags(session, ctx, item, fields.get("tag_ids"))
            await self._enqueue(session, item)
            return outcome, knowledge_dict(item, tags)

    async def create_url(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        url = (body.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise Invalid("url must be an http or https URL")
        async with async_session_scope() as session:
            kb = await self._contribute(session, ctx, kb_ref)
            visibility, groups = self._share(ctx, kb, body.get("visibility"), body.get("groups"))
            duplicates = (await session.execute(sa.select(Knowledge).where(
                Knowledge.knowledge_base_id == kb.id, Knowledge.source == url, Knowledge.type == "url",
                Knowledge.deleted_at.is_(None)))).scalars().all()
            readable = next((d for d in duplicates if access.can_read_item(ctx, d)), None)
            if readable is not None:
                raise Conflict("this URL is already in the knowledge base", knowledge_dict(readable))
            item = Knowledge(id=new_id(), knowledge_base_id=kb.id, owner_subject=ctx.subject,
                             owner_org=ctx.organization_id, type="url", source=url, title=body.get("title") or url,
                             visibility=visibility, groups=groups, external_id=body.get("external_id") or None,
                             refresh_interval=int(body.get("refresh_interval") or 0), channel="api")
            self._freshness(item, body)
            self._apply_citation(item, self._supplied_citation(body) or {}, {})
            session.add(item)
            await session.flush()
            tags = await self._tags(session, ctx, item, body.get("tag_ids"))
            await self._enqueue(session, item)
            return knowledge_dict(item, tags)

    async def create_manual(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        title = (body.get("title") or "").strip()
        content = body.get("content") or ""
        status = body.get("status") or "publish"
        if not title or not content.strip():
            raise Invalid("title and content are required")
        if status not in ("draft", "publish"):
            raise Invalid("status must be draft or publish")
        async with async_session_scope() as session:
            kb = await self._contribute(session, ctx, kb_ref)
            visibility, groups = self._share(ctx, kb, body.get("visibility"), body.get("groups"))
            item = Knowledge(id=new_id(), knowledge_base_id=kb.id, owner_subject=ctx.subject,
                             owner_org=ctx.organization_id, type="manual", title=title, source="manual",
                             visibility=visibility, groups=groups, channel="api",
                             meta={"content": content, "format": "markdown", "status": status, "version": 1},
                             parse_status="pending" if status == "publish" else "draft")
            self._apply_citation(item, self._supplied_citation(body) or {}, {})
            session.add(item)
            await session.flush()
            tags = await self._tags(session, ctx, item, body.get("tag_ids"))
            if status == "publish":
                await self._enqueue(session, item, "manual:process")
            return knowledge_dict(item, tags)

    async def update_manual(self, ctx: AuthContext, knowledge_id: str, body: dict) -> dict:
        async with async_session_scope() as session:
            item, _ = await self._managed(session, ctx, knowledge_id)
            if item.type != "manual":
                raise Invalid("only manual knowledge is edited this way")
            meta = dict(item.meta or {})
            if "content" in body:
                meta["content"] = body["content"]
                meta["version"] = int(meta.get("version", 1)) + 1
            if body.get("status"):
                meta["status"] = body["status"]
            if body.get("title"):
                item.title = body["title"]
            item.meta = meta
            if meta.get("status") == "publish":
                item.parse_status = "pending"
                await self._enqueue(session, item, "manual:process")
            await session.flush()
            return knowledge_dict(item)

    # -- processing ----------------------------------------------------------------------------------
    async def _set_status(self, knowledge_id: str, status: str, error: str = "") -> None:
        async with async_session_scope() as session:
            item = await session.get(Knowledge, knowledge_id, with_for_update=True)
            if item is None or item.deleted_at is not None:
                return
            item.parse_status, item.error_message = status, error[:4000]
            await event(session, item, {"type": "knowledge_status", "parse_status": status})

    async def _process_failed(self, ctx: TaskContext, error: str) -> None:
        if ctx.payload.get("refresh"):
            # a failed refresh changes nothing: the last good version keeps being served
            async with async_session_scope() as session:
                await session.execute(sa.update(Knowledge).where(Knowledge.id == ctx.payload["knowledge_id"])
                                      .values(last_refreshed_at=utcnow()))
            LOG.warning("refresh of %s failed: %s", ctx.payload["knowledge_id"], error)
            return
        await self._set_status(ctx.payload["knowledge_id"], "failed", error)

    async def _markdown(self, item: Knowledge) -> tuple[str, list, dict]:
        if item.type == "manual":
            return parsers.normalise((item.meta or {}).get("content", "")), [], {}
        if item.type == "url":
            try:
                page = await fetch(item.source, max_bytes=MAX_URL_BYTES)
            except FetchError as exc:
                raise Permanent(str(exc)) from exc
            if page.status >= 400:
                raise RuntimeError(f"{item.source}: HTTP {page.status}")    # retried
            ext = CONTENT_TYPES.get(page.content_type) or parsers.extension(page.url) or ".html"
            result = await anyio.to_thread.run_sync(parsers.parse, page.body, "page" + ext)
            return result.markdown, result.images, result.metadata
        data = await anyio.to_thread.run_sync(self.storage.read, item.file_path)
        result = await anyio.to_thread.run_sync(parsers.parse, data, item.file_name)
        return result.markdown, result.images, result.metadata

    async def _process_task(self, ctx: TaskContext) -> None:
        knowledge_id = ctx.payload["knowledge_id"]
        refresh = bool(ctx.payload.get("refresh"))
        async with async_session_scope() as session:
            item = await session.get(Knowledge, knowledge_id)
            if item is None or item.deleted_at is not None or item.parse_status in ("cancelled", "deleting", "draft"):
                return
            kb = await session.get(KnowledgeBase, item.knowledge_base_id)
            if kb is None or kb.deleted_at is not None:
                return
            session.expunge(item)
            session.expunge(kb)
        if not refresh:
            await self._set_status(knowledge_id, "processing")
        try:
            markdown, images, metadata = await self._markdown(item)
        except ParseError as exc:
            raise Permanent(str(exc)) from exc
        if not markdown.strip():
            raise Permanent("no text could be extracted")
        content_hash = hashlib.sha256(markdown.encode()).hexdigest()
        if refresh and content_hash == item.content_hash:
            # unchanged page: one fetch, no re-embedding
            async with async_session_scope() as session:
                await session.execute(sa.update(Knowledge).where(Knowledge.id == knowledge_id)
                                      .values(last_refreshed_at=utcnow()))
            return
        if refresh:
            await self._set_status(knowledge_id, "processing")
        prefix = self._prefix(item)
        parsed_path = f"{prefix}/parsed.{content_hash[:16]}.md"
        await anyio.to_thread.run_sync(self.storage.save, parsed_path, markdown.encode())
        image_names = []
        for image_name, image_bytes in images:
            image_names.append(safe_name(image_name))
            await anyio.to_thread.run_sync(self.storage.save, f"{prefix}/images/{image_names[-1]}", image_bytes)
        title = item.title
        if item.type == "url" and metadata.get("title") and title in ("", item.source):
            title = metadata["title"]

        pieces = await anyio.to_thread.run_sync(lambda: chunker.chunk(markdown, kb.chunking_config,
                                                                      _default_strategy()))
        pages = list(metadata.pop("pages", None) or [])          # page / slide start offsets
        headings = citation.heading_index(markdown)              # sections, whatever the chunking strategy
        chunk_ids = [new_id() for _ in pieces]
        indexed = [i for i, p in enumerate(pieces) if p.chunk_type == "text"]
        texts = [index_text(title, pieces[i].context_header, pieces[i].content) for i in indexed]

        vectors: list[list[float]] | None = None
        dimension = 0
        if kb.indexing_strategy.get("vector_enabled", True) and kb.embedding_model_id and texts:
            try:
                embedder = await self.models.embedder(kb.embedding_model_id)
                dimension = embedder.dimension
                await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), dimension)
                vectors = await embedder.embed(texts)
            except ModelError as exc:
                raise RuntimeError(f"embedding failed: {exc}") from exc     # retried

        async with async_session_scope() as session:
            current = await session.get(Knowledge, knowledge_id, with_for_update=True)
            if current is None or current.deleted_at is not None or current.parse_status in ("cancelled", "deleting"):
                return
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == knowledge_id))
            await session.execute(sa.delete(Chunk).where(Chunk.knowledge_id == knowledge_id))
            enabled = current.enable_status == "enabled"
            chunk_rows = []
            for i, piece in enumerate(pieces):
                row = Chunk(
                    id=chunk_ids[i], knowledge_base_id=kb.id, knowledge_id=knowledge_id, content=piece.content,
                    content_hash=hashlib.sha256(piece.content.encode()).hexdigest(), chunk_index=i,
                    chunk_type=piece.chunk_type, start_at=piece.start, end_at=piece.end,
                    pre_chunk_id=chunk_ids[i - 1] if i > 0 else "",
                    next_chunk_id=chunk_ids[i + 1] if i + 1 < len(pieces) else "",
                    parent_chunk_id=chunk_ids[piece.parent] if piece.parent is not None else "",
                    context_header=piece.context_header, status=2,
                    meta=_location(pages, headings, piece))
                session.add(row)
                chunk_rows.append(row)
            await session.flush()
            for n, i in enumerate(indexed):
                session.add(Embedding(
                    source_id=chunk_ids[i], source_type=SOURCE_CHUNK, chunk_id=chunk_ids[i], knowledge_id=knowledge_id,
                    knowledge_base_id=kb.id, owner_subject=current.owner_subject, visibility=current.visibility,
                    groups=list(current.groups or []), content=texts[n], dimension=dimension if vectors else 0,
                    embedding=vectors[n] if vectors else None, is_enabled=enabled, valid_from=current.valid_from,
                    valid_until=_valid_until(current), is_deprecated=current.deprecated_at is not None))
            meta = dict(current.meta or {})
            replaced = meta.pop("_replaced", None)
            if current.processed_at is not None and current.content_hash and current.content_hash != content_hash:
                replaced = replaced or {}
                session.add(KnowledgeVersion(
                    knowledge_id=knowledge_id, version=current.version,
                    file_name=replaced.get("file_name", current.file_name),
                    file_hash=replaced.get("file_hash", current.file_hash),
                    file_size=replaced.get("file_size", current.file_size), content_hash=current.content_hash,
                    stored_path=replaced.get("stored_path", ""),
                    replaced_by=replaced.get("replaced_by") or ("refresh" if refresh else current.type)))
                current.version += 1
            current.meta = meta
            if refresh:
                current.last_refreshed_at = utcnow()
            current.title = title
            self._apply_citation(current, parsed=metadata)
            current.content_hash, current.parsed_path = content_hash, parsed_path
            current.storage_size = len(markdown.encode())
            current.embedding_model_id = kb.embedding_model_id if vectors else None
            current.processed_at = utcnow()
            current.error_message = ""
            subtasks = await self.postprocess.plan(session, current, kb, chunk_rows, image_names) \
                if self.postprocess is not None else 0
            current.parse_status = "finalizing" if subtasks else "completed"
            current.pending_subtasks_count = subtasks
            await event(session, current, {"type": "knowledge_status", "parse_status": current.parse_status,
                                           "chunks": len(pieces)})
        LOG.info("processed %s: %d chunks, %d indexed, %d subtasks", knowledge_id, len(pieces), len(indexed), subtasks)

    # -- reading ---------------------------------------------------------------------------------------
    async def list(self, ctx: AuthContext, kb_ref: str, page: int = 1, page_size: int = 20,
                   filters: dict | None = None) -> tuple[list[dict], int]:
        filters = filters or {}
        async with async_session_scope(commit=False) as session:
            kb = await self.kbs.load(session, ctx, kb_ref)
            query = sa.select(Knowledge).where(Knowledge.deleted_at.is_(None),
                                               access.items_clause(Knowledge, ctx, [kb.id]))
            if filters.get("keyword"):
                like = f"%{filters['keyword']}%"
                query = query.where(sa.or_(Knowledge.title.ilike(like), Knowledge.file_name.ilike(like)))
            if filters.get("file_type"):
                query = query.where(Knowledge.file_type == filters["file_type"])
            if filters.get("parse_status"):
                query = query.where(Knowledge.parse_status == filters["parse_status"])
            if filters.get("tag_ids"):
                from kb_runtime.services.tags import item_tag_clause, resolve_tags

                tag_ids, untagged = await resolve_tags(session, [kb.id], filters["tag_ids"])
                query = query.where(item_tag_clause(tag_ids, untagged))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = list((await session.execute(query.order_by(Knowledge.created_at.desc())
                                               .offset((page - 1) * page_size).limit(page_size))).scalars())
            tags = await self.tags.item_tags(session, [k.id for k in rows]) if self.tags else {}
            return [knowledge_dict(k, tags.get(k.id)) for k in rows], total

    async def get(self, ctx: AuthContext, knowledge_id: str) -> dict:
        from kb_core.orm.models.kb import KnowledgeConflict

        async with async_session_scope(commit=False) as session:
            item, kb = await self._load(session, ctx, knowledge_id)
            tags = await self.tags.item_tags(session, [item.id]) if self.tags else {}
            newer = list((await session.execute(sa.select(KnowledgeConflict.knowledge_id).where(
                KnowledgeConflict.other_knowledge_id == item.id, KnowledgeConflict.status == "open"))).scalars())
            if newer:
                readable = set((await session.execute(sa.select(Knowledge.id).where(
                    Knowledge.id.in_(newer), Knowledge.deleted_at.is_(None),
                    access.items_clause(Knowledge, ctx, [kb.id])))).scalars())
                newer = [n for n in newer if n in readable]
            return knowledge_dict(item, tags.get(item.id), newer)

    async def versions(self, ctx: AuthContext, knowledge_id: str) -> dict:
        async with async_session_scope(commit=False) as session:
            item, _ = await self._load(session, ctx, knowledge_id)
            rows = (await session.execute(sa.select(KnowledgeVersion).where(
                KnowledgeVersion.knowledge_id == item.id).order_by(KnowledgeVersion.version.desc()))).scalars().all()
            return {"knowledge_id": item.id, "current_version": item.version,
                    "versions": [{"version": v.version, "file_name": v.file_name, "file_hash": v.file_hash,
                                  "file_size": v.file_size, "content_hash": v.content_hash,
                                  "file_kept": bool(v.stored_path), "replaced_by": v.replaced_by,
                                  "replaced_at": _iso(v.replaced_at)} for v in rows]}

    async def download_version(self, ctx: AuthContext, knowledge_id: str, version: int) -> tuple[str, bytes]:
        async with async_session_scope(commit=False) as session:
            item, _ = await self._load(session, ctx, knowledge_id)
            row = (await session.execute(sa.select(KnowledgeVersion).where(
                KnowledgeVersion.knowledge_id == item.id, KnowledgeVersion.version == version))).scalar_one_or_none()
        if row is None or not row.stored_path:
            raise NotFound("this version's file was not kept")
        return row.file_name, await anyio.to_thread.run_sync(self.storage.read, row.stored_path)

    async def batch(self, ctx: AuthContext, ids: list[str]) -> list[dict]:
        out = []
        for knowledge_id in ids[:100]:
            try:
                out.append(await self.get(ctx, knowledge_id))
            except NotFound:
                continue
        return out

    async def content(self, ctx: AuthContext, knowledge_id: str, offset: int = 0, limit: int = CONTENT_PAGE) -> dict:
        async with async_session_scope(commit=False) as session:
            item, _ = await self._load(session, ctx, knowledge_id)
        if not item.parsed_path:
            raise NotFound("this knowledge has not been processed yet")
        text = await anyio.to_thread.run_sync(self.storage.read_text, item.parsed_path)
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 100_000))
        part = text[offset:offset + limit]
        end = offset + len(part)
        return {"knowledge_id": item.id, "title": item.title, "citation": item.citation or {}, "offset": offset,
                "returned_chars": len(part),
                "content_length": len(text), "next_offset": end if end < len(text) else None, "content": part}

    async def download(self, ctx: AuthContext, knowledge_id: str) -> tuple[str, bytes]:
        async with async_session_scope(commit=False) as session:
            item, _ = await self._load(session, ctx, knowledge_id)
        if item.type != "file" or not item.file_path:
            raise NotFound("this knowledge has no original file")
        return item.file_name, await anyio.to_thread.run_sync(self.storage.read, item.file_path)

    # -- changing ------------------------------------------------------------------------------------
    async def update(self, ctx: AuthContext, knowledge_id: str, body: dict) -> dict:
        async with async_session_scope() as session:
            item, kb = await self._managed(session, ctx, knowledge_id)
            for field in ITEM_FIELDS:
                if field in body and body[field] is not None:
                    setattr(item, field, body[field])
            self._freshness(item, body)
            if "citation" in body:          # null resets it to what the document itself says
                supplied = self._supplied_citation({"citation": body["citation"]}) if body["citation"] else {}
                self._apply_citation(item, supplied or {})
            if body.get("enable_status"):
                if body["enable_status"] not in ("enabled", "disabled"):
                    raise Invalid("enable_status must be enabled or disabled")
                item.enable_status = body["enable_status"]
            if "visibility" in body or "groups" in body:
                before = item.visibility
                item.visibility, item.groups = self._share(ctx, kb, body.get("visibility") or item.visibility,
                                                           body.get("groups", item.groups))
                if self.wiki is not None and before != item.visibility and item.parse_status == "completed":
                    # only items every reader of the KB may read feed its wiki (docs/llm-wiki.md)
                    from kb_runtime.wiki.service import feeds_wiki

                    if not feeds_wiki(item, kb):
                        await self.wiki.queue_retract(session, item)
                    elif self.wiki.enabled(kb):
                        from kb_core.orm.models.ext import WikiPendingOp

                        session.add(WikiPendingOp(knowledge_base_id=kb.id, knowledge_id=item.id, op="ingest",
                                                  payload={"content_hash": item.content_hash, "rebuild": True}))
                        await self.wiki.trigger(session, kb)
            item.updated_at = utcnow()
            tags = await self._tags(session, ctx, item, body.get("tag_ids"))
            # the copies on the index rows, so the next search applies the change
            await session.execute(sa.update(Embedding).where(Embedding.knowledge_id == item.id).values(
                visibility=item.visibility, groups=list(item.groups or []), is_enabled=item.enable_status == "enabled",
                valid_from=item.valid_from, valid_until=_valid_until(item),
                is_deprecated=item.deprecated_at is not None))
            await session.flush()
            return knowledge_dict(item, tags)

    async def set_tags(self, ctx: AuthContext, updates: dict[str, list[str]]) -> dict:
        """PUT /knowledge/tags: replace each item's tag set."""
        out = {}
        async with async_session_scope() as session:
            for knowledge_id, tag_refs in updates.items():
                item, _ = await self._managed(session, ctx, knowledge_id)
                out[knowledge_id] = await self._tags(session, ctx, item, list(tag_refs or []))
        return out

    async def delete(self, ctx: AuthContext, knowledge_id: str) -> None:
        async with async_session_scope() as session:
            item, kb = await self._managed(session, ctx, knowledge_id)
            if self.wiki is not None:
                await self.wiki.queue_retract(session, item)
            item.parse_status = "deleting"
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == item.id))
            await session.execute(sa.delete(Chunk).where(Chunk.knowledge_id == item.id))
            kb_row = await session.get(KnowledgeBase, kb.id, with_for_update=True)
            kb_row.storage_used = max(0, kb_row.storage_used - item.file_size)
            item.deleted_at = utcnow()
            prefix = self._prefix(item)
            await event(session, item, {"type": "knowledge_deleted"})
        self.storage.delete_prefix(prefix)

    async def reparse(self, ctx: AuthContext, knowledge_id: str) -> dict:
        async with async_session_scope() as session:
            item, _ = await self._managed(session, ctx, knowledge_id)
            if item.type == "faq":
                raise Invalid("FAQ entries are indexed when written; change faq_config to re-index them")
            if item.type == "manual" and (item.meta or {}).get("status") != "publish":
                raise Invalid("a draft is published with PUT /knowledge/manual/{id}")
            item.parse_status, item.error_message = "pending", ""
            await self._enqueue(session, item, "manual:process" if item.type == "manual" else "document:process")
            return knowledge_dict(item)

    async def regenerate_summary(self, ctx: AuthContext, knowledge_id: str) -> dict:
        """Write the item's summary again (the `summary:generation` subtask)."""
        async with async_session_scope() as session:
            item, kb = await self._managed(session, ctx, knowledge_id)
            if item.parse_status not in ("completed", "finalizing") or not item.content_hash:
                raise Invalid("the item is not processed yet")
            if self.postprocess is None or not await self.postprocess.chat_model_id(kb):
                raise Invalid("no chat model for summaries (the KB's summary_model_id or [models] default_chat)")
            task_id = await queue.enqueue(session, "summary:generation",
                                          {"knowledge_id": item.id, "content_hash": item.content_hash},
                                          owner_subject=item.owner_subject, owner_org=item.owner_org,
                                          knowledge_base_id=kb.id, queue="postprocess",
                                          unique_key=f"summary:{item.id}", max_retry=2)
            item.summary_status = "pending"
            return {"knowledge_id": item.id, "queued": task_id is not None}

    async def reindex(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        """Re-embed every item (optionally with another embedding model): each item is processed
        again and swapped, so it stays searchable throughout."""
        model_id = await self.models.usable(ctx, body["embedding_model_id"], "Embedding") \
            if body.get("embedding_model_id") else None
        async with async_session_scope() as session:
            kb = await self.kbs.load(session, ctx, kb_ref, for_update=True)
            if not access.can_manage(ctx, kb):
                raise Forbidden("only the owner reindexes a knowledge base")
            if model_id:
                kb.embedding_model_id = model_id
                kb.indexing_strategy = {**kb.indexing_strategy, "vector_enabled": True}
            items = (await session.execute(sa.select(Knowledge).where(
                Knowledge.knowledge_base_id == kb.id, Knowledge.deleted_at.is_(None),
                Knowledge.parse_status.notin_(("draft", "deleting"))))).scalars().all()
            for item in items:
                item.parse_status = "pending"
                await self._enqueue(session, item, "manual:process" if item.type == "manual" else "document:process")
            return {"knowledge_base_id": kb.id, "embedding_model_id": kb.embedding_model_id, "queued": len(items)}

    async def cancel(self, ctx: AuthContext, knowledge_id: str) -> dict:
        async with async_session_scope() as session:
            item, _ = await self._managed(session, ctx, knowledge_id)
            if item.parse_status in ("pending", "processing", "finalizing"):
                item.parse_status = "cancelled"
            return knowledge_dict(item)

    # -- chunks ----------------------------------------------------------------------------------------
    async def chunks(self, ctx: AuthContext, knowledge_id: str, page: int = 1, page_size: int = 20):
        async with async_session_scope(commit=False) as session:
            await self._load(session, ctx, knowledge_id)
            query = sa.select(Chunk).where(Chunk.knowledge_id == knowledge_id)
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(Chunk.chunk_index).offset((page - 1) * page_size)
                                          .limit(page_size))).scalars()
            return [chunk_dict(c) for c in rows], total

    async def chunk(self, ctx: AuthContext, chunk_id: str) -> dict:
        async with async_session_scope(commit=False) as session:
            row = await session.get(Chunk, chunk_id)
            if row is None:
                raise NotFound("chunk not found")
            await self._load(session, ctx, row.knowledge_id)
            return chunk_dict(row)

    async def update_chunk(self, ctx: AuthContext, knowledge_id: str, chunk_id: str, body: dict) -> dict:
        vector = None
        async with async_session_scope(commit=False) as session:
            item, kb = await self._managed(session, ctx, knowledge_id, for_update=False)
            row = await session.get(Chunk, chunk_id)
            if row is None or row.knowledge_id != knowledge_id:
                raise NotFound("chunk not found")
            new_content = body.get("content")
            text = index_text(item.title, row.context_header, new_content) if new_content else None
            embedding_model = item.embedding_model_id
        if text is not None and embedding_model:
            try:
                vector = (await (await self.models.embedder(embedding_model)).embed([text]))[0]
            except ModelError as exc:
                raise Invalid(f"re-embedding the chunk failed: {exc}") from exc
        async with async_session_scope() as session:
            row = await session.get(Chunk, chunk_id, with_for_update=True)
            values: dict[str, Any] = {}
            if text is not None:
                row.content, row.content_revision = body["content"], row.content_revision + 1
                row.content_hash = hashlib.sha256(row.content.encode()).hexdigest()
                values["content"] = text
                if vector is not None:
                    values["embedding"] = vector
            if "is_enabled" in body:
                row.is_enabled = bool(body["is_enabled"])
                values["is_enabled"] = row.is_enabled and item.enable_status == "enabled"
            row.updated_at = utcnow()
            if values:
                await session.execute(sa.update(Embedding).where(Embedding.chunk_id == chunk_id).values(**values))
            return chunk_dict(row)

    async def delete_chunk(self, ctx: AuthContext, knowledge_id: str, chunk_id: str | None = None) -> None:
        async with async_session_scope() as session:
            await self._managed(session, ctx, knowledge_id)
            where = [Chunk.knowledge_id == knowledge_id] + ([Chunk.id == chunk_id] if chunk_id else [])
            ewhere = [Embedding.knowledge_id == knowledge_id] + ([Embedding.chunk_id == chunk_id] if chunk_id else [])
            await session.execute(sa.delete(Embedding).where(*ewhere))
            await session.execute(sa.delete(Chunk).where(*where))
