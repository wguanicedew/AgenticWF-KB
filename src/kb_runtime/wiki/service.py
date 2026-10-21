"""
WikiService: the pages of a KB's LLM wiki and everything around them except the LLM passes
(docs/llm-wiki.md): pages, revisions, folders, issues, lint, search, graph, relations,
communities, the search index of pages, and finalize (no LLM calls).

The wiki pipeline (kb_runtime.wiki.pipeline) writes pages through `write_page` too, so every
write snapshots a revision, re-indexes the page and keeps the retention caps.

Access: pages follow their KB. Reads need read access to the KB; writes need write access
(`contribute`); a page in an unreadable KB is "not found".
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

import anyio
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope, get_engine
from kb_core.orm.models.ext import (WikiCommunity, WikiFolder, WikiPage, WikiPageIssue, WikiPageRevision,
                                    WikiPendingOp, WikiRelation)
from kb_core.orm.models.indexes import ensure_vector_index
from kb_core.orm.models.kb import SOURCE_WIKI_PAGE, Embedding, Knowledge, KnowledgeBase, utcnow
from kb_runtime import access, features
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound
from kb_runtime.tasks import queue
from kb_runtime.wiki import text as wt

LOG = get_logger(__name__)

DEFAULTS = {"synthesis_model": "", "extraction_granularity": "standard", "ingest_batch_size": 5,
            "ingest_map_parallel": 10, "ingest_reduce_parallel": 10, "ingest_max_inflight": 4, "language": "auto",
            "revision_soft_cap": 50, "revision_hard_cap": 200}
ISSUE_TYPES = ("mixed_entities", "contradictory_facts", "out_of_date", "dead_link", "other")
PASSAGE_CHARS = 1200
SRC = re.compile(r"\[src:(\d+)\]")


def wiki_defaults() -> dict:
    out = dict(DEFAULTS)
    for key, value in DEFAULTS.items():
        if config_has_option("wiki", key):
            raw = config_get("wiki", key).strip()
            out[key] = type(value)(raw) if not isinstance(value, str) else raw
    return out


def page_dict(page: WikiPage, full: bool = True) -> dict:
    data = {"id": page.id, "knowledge_base_id": page.knowledge_base_id, "slug": page.slug, "title": page.title,
            "summary": page.summary, "page_type": page.page_type, "status": page.status, "aliases": page.aliases,
            "folder_id": page.folder_id, "category_path": page.category_path, "version": page.version,
            "source_refs": page.source_refs, "in_links": page.in_links, "out_links": page.out_links,
            "last_edit_source": page.last_edit_source, "last_editor": page.last_editor_id,
            "updated_at": page.updated_at.isoformat() if page.updated_at else None}
    if full:
        data.update(content=page.content, chunk_refs=page.chunk_refs, source_ids=page.source_ids)
    return data


def passages_of(page: WikiPage) -> list[str]:
    """The page as index passages: title + summary first, then the body by paragraphs."""
    head = f"{page.title}\n{page.summary}".strip()
    body = SRC.sub("", page.content or "")
    parts, current = [], head
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > PASSAGE_CHARS and current:
            parts.append(current)
            current = f"{page.title}\n{para}"
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts or [head]


def wt_seq(ref: str) -> int:
    """The chunk seq of a `chunk_refs` entry `<knowledge_id>|<chunk_id>|<seq>`."""
    parts = str(ref).split("|")
    return int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1


def feeds_wiki(item, kb) -> bool:
    """Wiki pages are readable by everyone who can read the KB, so only items those readers may
    all read feed it: items visible to `all`, or the owner's items in a private KB (whose only
    reader is the owner)."""
    if item.visibility == "all":
        return True
    return kb.visibility == "private" and item.owner_subject == kb.owner_subject


class WikiService:
    def __init__(self, models, kbs):
        self.models = models
        self.kbs = kbs
        self.enrichment = None             # set by bootstrap: finishing a document's wiki subtask

    # == settings and access ==============================================================================
    @staticmethod
    def config(kb: KnowledgeBase) -> dict:
        return {**wiki_defaults(), **(kb.wiki_config or {})}

    @staticmethod
    def enabled(kb: KnowledgeBase) -> bool:
        return features.enabled("wiki") and bool((kb.indexing_strategy or {}).get("wiki_enabled"))

    async def _kb(self, session, ctx: AuthContext, kb_ref: str, write: bool = False,
                  for_update: bool = False) -> KnowledgeBase:
        if not features.enabled("wiki"):
            raise NotFound("the wiki is switched off in this deployment")
        kb = await self.kbs.load(session, ctx, kb_ref, for_update=for_update)
        if write and not access.can_contribute(ctx, kb):
            raise Forbidden("you may not change this wiki")
        return kb

    async def model_id(self, kb: KnowledgeBase) -> str | None:
        cfg = self.config(kb)
        return cfg.get("synthesis_model_id") or cfg.get("synthesis_model") or await self.models.default_id("KnowledgeQA")

    # == hooks from ingestion ===============================================================================
    async def queue_ingest(self, session, item: Knowledge, kb: KnowledgeBase) -> bool:
        """Add a processed document to the work list (one enrichment subtask). Only items
        visible to the whole KB feed the wiki."""
        if not self.enabled(kb) or not feeds_wiki(item, kb) or not await self.model_id(kb):
            return False
        session.add(WikiPendingOp(knowledge_base_id=kb.id, knowledge_id=item.id, op="ingest",
                                  payload={"content_hash": item.content_hash}))
        await self.trigger(session, kb)
        return True

    async def queue_retract(self, session, item: Knowledge) -> None:
        """A deleted, re-parsed or no-longer-public document: remove its citations."""
        cited = (await session.execute(sa.select(WikiPage.id).where(
            WikiPage.knowledge_base_id == item.knowledge_base_id,
            WikiPage.source_ids.contains([item.id])).limit(1))).first()
        if not cited:
            return
        session.add(WikiPendingOp(knowledge_base_id=item.knowledge_base_id, knowledge_id=item.id, op="retract",
                                  payload={}))
        kb = await session.get(KnowledgeBase, item.knowledge_base_id)
        await self.trigger(session, kb)

    @staticmethod
    async def trigger(session, kb: KnowledgeBase) -> None:
        await queue.enqueue(session, "wiki:ingest", {"knowledge_base_id": kb.id}, owner_subject=kb.owner_subject,
                            owner_org=kb.owner_org, knowledge_base_id=kb.id, queue="wiki",
                            unique_key=f"wiki-ingest-{kb.id}", max_retry=3)

    @staticmethod
    async def trigger_finalize(session, kb: KnowledgeBase) -> None:
        await queue.enqueue(session, "wiki:finalize", {"knowledge_base_id": kb.id}, owner_subject=kb.owner_subject,
                            owner_org=kb.owner_org, knowledge_base_id=kb.id, queue="wiki",
                            unique_key=f"wiki-finalize-{kb.id}", max_retry=2)

    # == pages ================================================================================================
    @staticmethod
    async def _page(session, kb_id: str, slug: str, for_update: bool = False) -> WikiPage | None:
        query = sa.select(WikiPage).where(WikiPage.knowledge_base_id == kb_id, WikiPage.slug == slug,
                                          WikiPage.deleted_at.is_(None))
        if for_update:
            query = query.with_for_update()
        return (await session.execute(query)).scalar_one_or_none()

    async def _existing(self, session, kb_id: str, slug: str, for_update: bool = False) -> WikiPage:
        page = await self._page(session, kb_id, slug, for_update)
        if page is None:
            raise NotFound(f"page {slug} not found")
        return page

    async def list_pages(self, ctx, kb_ref, page_type=None, folder_id=None, status="published", page=1,
                         page_size=50) -> tuple[list[dict], int]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            query = sa.select(WikiPage).where(WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None))
            if status:
                query = query.where(WikiPage.status == status)
            if page_type:
                query = query.where(WikiPage.page_type == page_type)
            if folder_id is not None:
                query = query.where(WikiPage.folder_id == folder_id)
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(WikiPage.category_path, WikiPage.sort_order, WikiPage.title)
                                          .offset((page - 1) * page_size).limit(page_size))).scalars().all()
            return [page_dict(p, full=False) for p in rows], total

    async def get_page(self, ctx, kb_ref, slug) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            return page_dict(await self._existing(session, kb.id, slug))

    async def write_page(self, session, kb: KnowledgeBase, page: WikiPage | None, *, slug: str, title: str,
                         content: str, summary: str = "", page_type: str = "entity", aliases=None,
                         status: str = "published", source: str = "pipeline", editor: str = "",
                         source_refs=None, chunk_refs=None, folder_id: str | None = None,
                         vectors: tuple | None = None) -> WikiPage:
        """Create or overwrite a page: snapshot, write, version, retention, re-index."""
        if page is None:
            page = WikiPage(knowledge_base_id=kb.id, slug=slug, version=1, page_type=page_type)
            session.add(page)
        else:
            await self._snapshot(session, page)
            page.version += 1
        page.slug, page.title, page.content, page.summary = slug, title[:512], content, summary
        page.page_type, page.status = page_type, status
        if aliases is not None:
            page.aliases = sorted({a for a in aliases if a and a != title})[:50]
        if source_refs is not None:
            page.source_refs = list(dict.fromkeys(source_refs))
            page.source_ids = list(dict.fromkeys(r.split("|", 1)[0] for r in page.source_refs))
        if chunk_refs is not None:
            page.chunk_refs = list(dict.fromkeys(chunk_refs))
        if folder_id is not None:
            page.folder_id = folder_id
            page.category_path = await self._folder_path(session, folder_id)
        page.out_links = wt.links(content)
        page.last_edit_source, page.last_editor_id = source, editor
        page.updated_at = utcnow()
        await session.flush()
        await self._prune_revisions(session, kb, page)
        await self.index_page(session, kb, page, vectors)
        return page

    @staticmethod
    async def _snapshot(session, page: WikiPage) -> None:
        stmt = insert(WikiPageRevision).values(
            id=str(uuid.uuid4()), page_id=page.id, version=page.version, title=page.title,
            summary=page.summary, content=page.content, page_type=page.page_type, status=page.status,
            aliases=page.aliases, edit_source=page.last_edit_source, editor_id=page.last_editor_id,
            created_at=utcnow()).on_conflict_do_nothing(index_elements=["page_id", "version"])
        await session.execute(stmt)

    async def _prune_revisions(self, session, kb, page) -> None:
        cfg = self.config(kb)
        soft, hard = int(cfg["revision_soft_cap"]), int(cfg["revision_hard_cap"])
        rows = (await session.execute(sa.select(WikiPageRevision.id, WikiPageRevision.edit_source).where(
            WikiPageRevision.page_id == page.id).order_by(WikiPageRevision.version.desc()))).all()
        drop = []
        if len(rows) > soft:
            # beyond the soft cap only pipeline snapshots go, oldest first
            excess = len(rows) - soft
            for rid, src in reversed(rows):
                if excess <= 0:
                    break
                if src == "pipeline":
                    drop.append(rid)
                    excess -= 1
        remaining = [r for r in rows if r[0] not in drop]
        if len(remaining) > hard:
            drop += [rid for rid, _ in remaining[hard:]]
        if drop:
            await session.execute(sa.delete(WikiPageRevision).where(WikiPageRevision.id.in_(drop)))

    async def index_page(self, session, kb, page: WikiPage, vectors: tuple | None = None) -> None:
        """Published pages in hybrid search: source_type 4, chunk_id `wiki:<page id>:<n>`."""
        await session.execute(sa.delete(Embedding).where(Embedding.source_type == SOURCE_WIKI_PAGE,
                                                         Embedding.knowledge_id == f"wiki:{page.id}"))
        if page.status != "published" or page.deleted_at is not None:
            return
        texts = passages_of(page)
        if vectors is None:
            vectors = await self.embed(kb, texts)
        vecs, dim = vectors
        for i, text in enumerate(texts):
            session.add(Embedding(source_id=f"{page.id}:{i}", source_type=SOURCE_WIKI_PAGE,
                                  chunk_id=f"wiki:{page.id}:{i}", knowledge_id=f"wiki:{page.id}",
                                  knowledge_base_id=kb.id, owner_subject=kb.owner_subject, visibility="all", groups=[],
                                  content=text, dimension=dim if vecs else 0,
                                  embedding=vecs[i] if vecs and i < len(vecs) else None, is_enabled=True))

    async def ensure_index(self, kb_id: str) -> None:
        """Create the KB's HNSW index (if missing) before a write transaction touches
        kb_embeddings: CREATE INDEX would otherwise wait on that transaction."""
        async with async_session_scope(commit=False) as session:
            kb = await session.get(KnowledgeBase, kb_id)
        if kb is not None and kb.embedding_model_id and (kb.indexing_strategy or {}).get("vector_enabled", True):
            embedder = await self.models.embedder(kb.embedding_model_id)
            await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), embedder.dimension)

    async def _prepare(self, ctx, kb_ref) -> None:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            kb_id = kb.id
        await self.ensure_index(kb_id)

    async def embed(self, kb, texts: list[str]) -> tuple[list | None, int]:
        if not texts or not kb.embedding_model_id or not (kb.indexing_strategy or {}).get("vector_enabled", True):
            return None, 0
        embedder = await self.models.embedder(kb.embedding_model_id)
        await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), embedder.dimension)
        return await embedder.embed(texts), embedder.dimension

    async def create_page(self, ctx, kb_ref, body: dict) -> dict:
        await self._prepare(ctx, kb_ref)
        page_type = body.get("page_type") or "synthesis"
        if page_type not in wt.API_TYPES:
            raise Invalid(f"page_type must be one of {', '.join(wt.API_TYPES)} (index and community pages are generated)")
        title = (body.get("title") or "").strip()
        if not body.get("content") or not (title or body.get("slug")):
            raise Invalid("a page needs content and a title or slug")
        try:
            slug = wt.normalise_slug(body["slug"], page_type) if body.get("slug") else wt.page_slug(page_type, title)
        except ValueError as exc:
            raise Invalid(str(exc))
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            if await self._page(session, kb.id, slug):
                raise Conflict(f"page {slug} exists already")
            folder_id = await self._ensure_folder(session, kb.id, body["folder"]) if body.get("folder") else ""
            page = await self.write_page(session, kb, None, slug=slug, title=title or slug.rpartition("/")[2],
                                         content=body["content"], summary=body.get("summary") or "", page_type=page_type,
                                         aliases=body.get("aliases") or [], status=body.get("status") or "published",
                                         source="user", editor=ctx.subject, folder_id=folder_id)
            await self.trigger_finalize(session, kb)
            return page_dict(page)

    async def update_page(self, ctx, kb_ref, slug, body: dict) -> dict:
        await self._prepare(ctx, kb_ref)
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            page = await self._existing(session, kb.id, slug, for_update=True)
            if body.get("status") and body["status"] not in ("draft", "published", "archived"):
                raise Invalid("status must be draft, published or archived")
            new_slug = slug
            if body.get("slug") and body["slug"] != slug:
                new_slug = wt.normalise_slug(body["slug"], page.page_type)
                if await self._page(session, kb.id, new_slug):
                    raise Conflict(f"page {new_slug} exists already")
            page = await self.write_page(
                session, kb, page, slug=new_slug, title=body.get("title") or page.title,
                content=body["content"] if body.get("content") is not None else page.content,
                summary=body["summary"] if body.get("summary") is not None else page.summary, page_type=page.page_type,
                aliases=body.get("aliases") if body.get("aliases") is not None else page.aliases,
                status=body.get("status") or page.status, source="user", editor=ctx.subject)
            if new_slug != slug:
                await self._rewrite_backlinks(session, kb, {slug: new_slug}, ctx.subject)
            await self.trigger_finalize(session, kb)
            return page_dict(page)

    async def _rewrite_backlinks(self, session, kb, mapping: dict, editor: str) -> None:
        rows = (await session.execute(sa.select(WikiPage).where(
            WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None)))).scalars().all()
        for page in rows:
            if set(wt.links(page.content)) & set(mapping):
                await self.write_page(session, kb, page, slug=page.slug, title=page.title,
                                      content=wt.replace_links(page.content, mapping), summary=page.summary,
                                      page_type=page.page_type, status=page.status, source="user", editor=editor)

    async def delete_page(self, ctx, kb_ref, slug) -> None:
        await self._prepare(ctx, kb_ref)
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            page = await self._existing(session, kb.id, slug, for_update=True)
            if page.page_type == "index":
                raise Invalid("the index page is rebuilt by finalize")
            await self._snapshot(session, page)
            page.deleted_at = utcnow()
            await session.execute(sa.delete(Embedding).where(Embedding.source_type == SOURCE_WIKI_PAGE,
                                                             Embedding.knowledge_id == f"wiki:{page.id}"))
            await session.flush()
            await self._rewrite_backlinks(session, kb, {slug: None}, ctx.subject)
            await self.trigger_finalize(session, kb)

    async def move_page(self, ctx, kb_ref, slug, folder_id: str) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            page = await self._existing(session, kb.id, slug, for_update=True)
            if folder_id:
                folder = await session.get(WikiFolder, folder_id)
                if folder is None or folder.knowledge_base_id != kb.id:
                    raise NotFound("folder not found")
            page.folder_id = folder_id or ""
            page.category_path = await self._folder_path(session, page.folder_id)
            return page_dict(page, full=False)

    # -- revisions ----------------------------------------------------------------------------------------
    async def revisions(self, ctx, kb_ref, slug, version: int | None = None):
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            page = await self._existing(session, kb.id, slug)
            if version:
                rev = (await session.execute(sa.select(WikiPageRevision).where(
                    WikiPageRevision.page_id == page.id, WikiPageRevision.version == version))).scalar_one_or_none()
                if rev is None:
                    raise NotFound(f"revision {version} not found")
                return {"slug": slug, "version": rev.version, "title": rev.title, "summary": rev.summary,
                        "content": rev.content, "page_type": rev.page_type, "status": rev.status,
                        "aliases": rev.aliases, "edit_source": rev.edit_source, "editor": rev.editor_id,
                        "created_at": rev.created_at.isoformat()}
            rows = (await session.execute(sa.select(WikiPageRevision).where(WikiPageRevision.page_id == page.id)
                                          .order_by(WikiPageRevision.version.desc()))).scalars().all()
            return {"slug": slug, "current_version": page.version,
                    "revisions": [{"version": r.version, "title": r.title, "edit_source": r.edit_source,
                                   "editor": r.editor_id, "created_at": r.created_at.isoformat()} for r in rows]}

    async def revert(self, ctx, kb_ref, slug, version: int) -> dict:
        await self._prepare(ctx, kb_ref)
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            page = await self._existing(session, kb.id, slug, for_update=True)
            if int(version) == page.version:
                raise Invalid("this is the current version")
            rev = (await session.execute(sa.select(WikiPageRevision).where(
                WikiPageRevision.page_id == page.id, WikiPageRevision.version == int(version)))).scalar_one_or_none()
            if rev is None:
                raise NotFound(f"revision {version} not found")
            page = await self.write_page(session, kb, page, slug=page.slug, title=rev.title, content=rev.content,
                                         summary=rev.summary, page_type=rev.page_type, aliases=rev.aliases,
                                         status=rev.status, source="revert", editor=ctx.subject)
            await self.trigger_finalize(session, kb)
            return page_dict(page)

    # -- folders ---------------------------------------------------------------------------------------------
    @staticmethod
    async def _folder_path(session, folder_id: str) -> str:
        if not folder_id:
            return ""
        folder = await session.get(WikiFolder, folder_id)
        return folder.path if folder else ""

    async def _ensure_folder(self, session, kb_id: str, path: str, generated: bool = False) -> str:
        """The folder id for `a/b`, creating missing levels."""
        parent, built = "", ""
        for name in [p.strip() for p in str(path).split("/") if p.strip()][:3]:
            built = f"{built}/{name}" if built else name
            # concurrent page writes may create the same folder: insert-or-ignore, then read
            await session.execute(insert(WikiFolder).values(
                id=str(uuid.uuid4()), knowledge_base_id=kb_id, parent_id=parent, name=name[:255],
                path=built, sort_order=0, generated=generated, created_at=utcnow(), updated_at=utcnow())
                .on_conflict_do_nothing(index_elements=["knowledge_base_id", "path"]))
            folder = (await session.execute(sa.select(WikiFolder).where(
                WikiFolder.knowledge_base_id == kb_id, WikiFolder.path == built))).scalar_one()
            parent = folder.id
        return parent

    async def folders(self, ctx, kb_ref) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            rows = (await session.execute(sa.select(WikiFolder).where(WikiFolder.knowledge_base_id == kb.id)
                                          .order_by(WikiFolder.path))).scalars().all()
            return [{"id": f.id, "parent_id": f.parent_id, "name": f.name, "path": f.path, "sort_order": f.sort_order,
                     "generated": f.generated} for f in rows]

    async def create_folder(self, ctx, kb_ref, body) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            parent = ""
            if body.get("parent_id"):
                folder = await session.get(WikiFolder, body["parent_id"])
                if folder is None or folder.knowledge_base_id != kb.id:
                    raise NotFound("parent folder not found")
                parent = folder.path
            name = (body.get("name") or "").strip().replace("/", "-")
            if not name:
                raise Invalid("a folder needs a name")
            folder_id = await self._ensure_folder(session, kb.id, f"{parent}/{name}" if parent else name)
            folder = await session.get(WikiFolder, folder_id)
            return {"id": folder.id, "parent_id": folder.parent_id, "name": folder.name, "path": folder.path}

    async def update_folder(self, ctx, kb_ref, folder_id, body) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            folder = await session.get(WikiFolder, folder_id)
            if folder is None or folder.knowledge_base_id != kb.id:
                raise NotFound("folder not found")
            if body.get("name"):
                old = folder.path
                folder.name = body["name"].strip().replace("/", "-")
                folder.path = f"{old.rpartition('/')[0]}/{folder.name}".lstrip("/")
                for child in (await session.execute(sa.select(WikiFolder).where(
                        WikiFolder.knowledge_base_id == kb.id, WikiFolder.path.like(old + "/%")))).scalars():
                    child.path = folder.path + child.path[len(old):]
                await session.execute(sa.update(WikiPage).where(WikiPage.knowledge_base_id == kb.id,
                                                                WikiPage.category_path.like(old + "%")).values(
                    category_path=sa.func.concat(folder.path, sa.func.substr(WikiPage.category_path, len(old) + 1))))
            if "sort_order" in body:
                folder.sort_order = int(body["sort_order"] or 0)
            folder.generated = False
            return {"id": folder.id, "parent_id": folder.parent_id, "name": folder.name, "path": folder.path}

    async def delete_folder(self, ctx, kb_ref, folder_id) -> None:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            folder = await session.get(WikiFolder, folder_id)
            if folder is None or folder.knowledge_base_id != kb.id:
                raise NotFound("folder not found")
            used = (await session.execute(sa.select(sa.func.count()).select_from(WikiPage).where(
                WikiPage.folder_id == folder_id, WikiPage.deleted_at.is_(None)))).scalar_one()
            children = (await session.execute(sa.select(sa.func.count()).select_from(WikiFolder).where(
                WikiFolder.parent_id == folder_id))).scalar_one()
            if used or children:
                raise Invalid("the folder is not empty")
            await session.delete(folder)

    # -- issues, lint, stats, graph ------------------------------------------------------------------------
    async def issues(self, ctx, kb_ref, status: str | None = "pending") -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            query = sa.select(WikiPageIssue).where(WikiPageIssue.knowledge_base_id == kb.id)
            if status:
                query = query.where(WikiPageIssue.status == status)
            rows = (await session.execute(query.order_by(WikiPageIssue.created_at.desc()))).scalars().all()
            return [{"id": i.id, "slug": i.page_slug, "issue_type": i.issue_type, "description": i.description,
                     "status": i.status, "reporter": i.reporter, "created_at": i.created_at.isoformat()} for i in rows]

    async def flag_issue(self, ctx, kb_ref, body) -> dict:
        if body.get("issue_type") not in ISSUE_TYPES:
            raise Invalid(f"issue_type must be one of {', '.join(ISSUE_TYPES)}")
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            await self._existing(session, kb.id, body.get("slug") or "")
            issue = WikiPageIssue(knowledge_base_id=kb.id, page_slug=body["slug"], issue_type=body["issue_type"],
                                  description=str(body.get("description") or "")[:4000], reporter=ctx.subject)
            session.add(issue)
            await session.flush()
            return {"id": issue.id, "slug": issue.page_slug, "issue_type": issue.issue_type, "status": issue.status}

    async def set_issue_status(self, ctx, kb_ref, issue_id, status) -> dict:
        if status not in ("pending", "ignored", "resolved"):
            raise Invalid("status must be pending, ignored or resolved")
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            issue = await session.get(WikiPageIssue, issue_id)
            if issue is None or issue.knowledge_base_id != kb.id:
                raise NotFound("issue not found")
            issue.status, issue.updated_at = status, utcnow()
            return {"id": issue.id, "status": issue.status}

    async def quality(self, ctx, kb_ref) -> dict:
        """Deterministic quality metrics of a wiki (docs/llm-wiki.md, Quality): grounding,
        citation coverage and validity, links, orphans, relations, document coverage."""
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            pages = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                WikiPage.status == "published"))).scalars().all()
            seqs = {int(n) for p in pages for n in SRC.findall(p.content or "")}
            from kb_core.orm.models.kb import Chunk

            live_seqs = set((await session.execute(sa.select(Chunk.seq_id).where(
                Chunk.seq_id.in_(seqs or {-1})))).scalars())
            relations = (await session.execute(sa.select(WikiRelation).where(
                WikiRelation.knowledge_base_id == kb.id))).scalars().all()
            documents = [i for i in (await session.execute(sa.select(Knowledge).where(
                Knowledge.knowledge_base_id == kb.id, Knowledge.deleted_at.is_(None),
                Knowledge.parse_status == "completed", Knowledge.type != "faq"))).scalars().all() if feeds_wiki(i, kb)]
            pending = (await session.execute(sa.select(sa.func.count()).select_from(WikiPendingOp).where(
                WikiPendingOp.knowledge_base_id == kb.id))).scalar_one()
            issues = (await session.execute(sa.select(sa.func.count()).select_from(WikiPageIssue).where(
                WikiPageIssue.knowledge_base_id == kb.id, WikiPageIssue.status == "pending"))).scalar_one()
        known = {p.slug for p in pages}
        topical = [p for p in pages if p.page_type in ("entity", "concept")]
        generated = [p for p in pages if p.page_type in ("entity", "concept", "summary")]
        paragraphs = cited = 0
        markers = valid = 0
        for page in generated:
            refs = {wt_seq(r) for r in page.chunk_refs or []}
            for para in re.split(r"\n\s*\n", page.content or ""):
                text = para.strip()
                if len(text) < 40 or text.startswith("#"):
                    continue
                paragraphs += 1
                cited += 1 if SRC.search(text) else 0
            for n in SRC.findall(page.content or ""):
                markers += 1
                valid += 1 if int(n) in refs and int(n) in live_seqs else 0
        dead = sum(1 for p in pages for t in wt.links(p.content) if t not in known and not t.startswith("community/"))
        covered = {sid for p in pages for sid in (p.source_ids or [])}

        def share(a, b):
            return round(a / b, 4) if b else None

        metrics = {
            "pages": len(pages), "by_type": {t: sum(1 for p in pages if p.page_type == t)
                                             for t in sorted({p.page_type for p in pages})},
            "grounded_pages": share(sum(1 for p in topical if p.source_ids), len(topical)),
            "cited_paragraphs": share(cited, paragraphs),
            "valid_citations": share(valid, markers),
            "dead_links": dead,
            "orphan_pages": share(sum(1 for p in topical if not p.in_links), len(topical)),
            "relations": len(relations), "relations_cited": share(sum(1 for r in relations if r.chunk_refs), len(relations)),
            "documents": len(documents),
            "document_coverage": share(sum(1 for d in documents if d.id in covered), len(documents)),
            "sources_per_page": share(sum(len(p.source_ids or []) for p in topical), len(topical)),
            "pending_ops": pending, "open_issues": issues,
        }
        metrics["checks"] = {
            "every_page_has_a_source": metrics["grounded_pages"] in (None, 1.0),
            "no_dead_links": dead == 0,
            "citations_resolve": metrics["valid_citations"] in (None, 1.0),
            "every_relation_cites_a_chunk": metrics["relations_cited"] in (None, 1.0),
            "settled": pending == 0,
        }
        return metrics

    async def lint(self, ctx, kb_ref) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            pages = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                WikiPage.status == "published"))).scalars().all()
        known = {p.slug for p in pages}
        findings = []
        for p in pages:
            if p.page_type in ("entity", "concept") and not p.source_ids:
                findings.append({"slug": p.slug, "check": "no_sources"})
            for target in wt.links(p.content):
                if target not in known and not target.startswith("community/"):
                    findings.append({"slug": p.slug, "check": "dead_link", "target": target})
            if p.page_type in ("entity", "concept") and not p.in_links:
                findings.append({"slug": p.slug, "check": "orphan"})
            if not p.summary:
                findings.append({"slug": p.slug, "check": "no_summary"})
        return {"pages": len(pages), "findings": findings}

    async def stats(self, ctx, kb_ref) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            by_type = dict((await session.execute(sa.select(WikiPage.page_type, sa.func.count()).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                WikiPage.status == "published").group_by(WikiPage.page_type))).all())
            pending = (await session.execute(sa.select(sa.func.count()).select_from(WikiPendingOp).where(
                WikiPendingOp.knowledge_base_id == kb.id))).scalar_one()
            relations = (await session.execute(sa.select(sa.func.count()).select_from(WikiRelation).where(
                WikiRelation.knowledge_base_id == kb.id))).scalar_one()
            issues = (await session.execute(sa.select(sa.func.count()).select_from(WikiPageIssue).where(
                WikiPageIssue.knowledge_base_id == kb.id, WikiPageIssue.status == "pending"))).scalar_one()
            return {"enabled": self.enabled(kb), "pages": sum(by_type.values()), "by_type": by_type,
                    "pending_ops": pending, "relations": relations, "open_issues": issues}

    async def graph(self, ctx, kb_ref, slug: str | None = None) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            pages = (await session.execute(sa.select(WikiPage.slug, WikiPage.title, WikiPage.page_type,
                                                     WikiPage.out_links).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                WikiPage.status == "published"))).all()
            relations = (await session.execute(sa.select(WikiRelation).where(
                WikiRelation.knowledge_base_id == kb.id))).scalars().all()
        nodes = {s: {"slug": s, "title": t, "page_type": k} for s, t, k, _ in pages if k != "index"}
        edges = [{"source": s, "target": t, "kind": "link"} for s, _, _, out in pages for t in out or []
                 if s in nodes and t in nodes]
        edges += [{"source": r.source_slug, "target": r.target_slug, "kind": "relation", "relation": r.relation,
                   "weight": r.weight} for r in relations if r.source_slug in nodes and r.target_slug in nodes]
        if slug:
            if slug not in nodes:
                raise NotFound(f"page {slug} not found")
            edges = [e for e in edges if slug in (e["source"], e["target"])]
            keep = {slug} | {e["source"] for e in edges} | {e["target"] for e in edges}
            nodes = {k: v for k, v in nodes.items() if k in keep}
        return {"nodes": list(nodes.values()), "edges": edges}

    async def index(self, ctx, kb_ref) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            page = await self._page(session, kb.id, "index")
            return page_dict(page) if page else {"slug": "index", "content": "", "title": "Index"}

    async def relations(self, ctx, kb_ref, slug: str | None = None) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            query = sa.select(WikiRelation).where(WikiRelation.knowledge_base_id == kb.id)
            if slug:
                query = query.where(sa.or_(WikiRelation.source_slug == slug, WikiRelation.target_slug == slug))
            rows = (await session.execute(query.order_by(WikiRelation.weight.desc()))).scalars().all()
            return [{"source": r.source_slug, "relation": r.relation, "target": r.target_slug,
                     "description": r.description, "weight": r.weight, "chunk_refs": r.chunk_refs} for r in rows]

    async def communities(self, ctx, kb_ref) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            rows = (await session.execute(sa.select(WikiCommunity).where(WikiCommunity.knowledge_base_id == kb.id)
                                          .order_by(WikiCommunity.level, WikiCommunity.page_slug))).scalars().all()
            return [{"id": c.id, "level": c.level, "parent_id": c.parent_id, "members": c.member_slugs,
                     "page_slug": c.page_slug} for c in rows]

    # -- search ------------------------------------------------------------------------------------------------
    async def search_pages(self, session, kb_ids: list[str], query: str, regex: bool = True, limit: int = 10) -> list:
        """Title, slug, aliases, summary and content as a case-insensitive POSIX regex (ILIKE when
        not a regex, or when it is invalid); title hits first, then summary hits, then in-links."""
        if not query.strip() or not kb_ids:
            return []
        await session.execute(sa.text("SET LOCAL statement_timeout = '5s'"))
        params = {"kbs": kb_ids, "q": query, "lim": max(1, min(int(limit), 30)),
                  "like": "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"}

        def sql(op):
            m = (lambda col: f"{col} ~* :q") if op == "regex" else (lambda col: f"{col} ILIKE :like")
            return sa.text(
                f"SELECT id, (CASE WHEN {m('title')} THEN 0 WHEN {m('summary')} THEN 1 ELSE 2 END) AS rank "
                f"FROM kb_wiki_pages WHERE knowledge_base_id = ANY(:kbs) AND deleted_at IS NULL "
                f"AND status = 'published' AND page_type <> 'index' "
                f"AND ({m('title')} OR {m('slug')} OR {m('aliases::text')} OR {m('summary')} OR {m('content')}) "
                f"ORDER BY rank, jsonb_array_length(in_links) DESC, title LIMIT :lim")

        rows = None
        if regex:
            try:
                async with session.begin_nested():
                    rows = (await session.execute(sql("regex"), params)).all()
            except sa.exc.DBAPIError:
                rows = None
        if rows is None:
            rows = (await session.execute(sql("like"), params)).all()
        ids = [r[0] for r in rows]
        pages = {p.id: p for p in (await session.execute(sa.select(WikiPage).where(WikiPage.id.in_(ids)))).scalars()} \
            if ids else {}
        return [pages[i] for i in ids if i in pages]

    async def search(self, ctx, kb_ref, q: str, regex: bool = True, limit: int = 10) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            pages = await self.search_pages(session, [kb.id], q, regex, limit)
            return [{"slug": p.slug, "title": p.title, "page_type": p.page_type, "summary": p.summary} for p in pages]

    # == finalize (no LLM calls) ======================================================================================
    async def finalize(self, kb_id: str) -> dict:
        await self.ensure_index(kb_id)
        report = {"dead_links": 0, "cross_links": 0, "folders_pruned": 0, "communities_changed": 0}
        async with async_session_scope() as session:
            kb = await session.get(KnowledgeBase, kb_id, with_for_update=True)
            if kb is None or kb.deleted_at is not None:
                return report
            pages = list((await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None)).with_for_update())).scalars())
            live = [p for p in pages if p.status == "published"]
            known = {p.slug for p in live}
            targets = [(p.slug, [p.title, *(p.aliases or [])]) for p in live
                       if p.page_type in ("entity", "concept", "synthesis", "comparison")]
            for page in live:
                if page.page_type in ("index", "community"):
                    continue
                content, dead = wt.strip_dead_links(page.content, known)
                for target in dead:
                    session.add(WikiPageIssue(knowledge_base_id=kb.id, page_slug=page.slug, issue_type="dead_link",
                                              description=f"link to missing page {target} removed"))
                report["dead_links"] += len(dead)
                linked = wt.inject_cross_links(content, page.slug, targets)
                report["cross_links"] += len(wt.links(linked)) - len(wt.links(content))
                if linked != page.content:
                    await self.write_page(session, kb, page, slug=page.slug, title=page.title, content=linked,
                                          summary=page.summary, page_type=page.page_type, status=page.status,
                                          source="pipeline")
            # in / out links
            incoming: dict[str, list[str]] = {}
            for page in live:
                page.out_links = [t for t in wt.links(page.content) if t in known]
                for target in page.out_links:
                    incoming.setdefault(target, []).append(page.slug)
            for page in live:
                page.in_links = sorted(set(incoming.get(page.slug, [])))
            # the index page
            await self._write_index(session, kb, live)
            # empty generated folders
            folders = list((await session.execute(sa.select(WikiFolder).where(
                WikiFolder.knowledge_base_id == kb.id))).scalars())
            used = {p.folder_id for p in pages if p.deleted_at is None}
            parents = {f.parent_id for f in folders}
            for folder in sorted(folders, key=lambda f: -f.path.count("/")):
                if folder.generated and folder.id not in used and folder.id not in parents:
                    await session.delete(folder)
                    report["folders_pruned"] += 1
            report["communities_changed"] = await self._detect_communities(session, kb, live)
        return report

    async def _write_index(self, session, kb, live: list[WikiPage]) -> None:
        groups: dict[str, list[WikiPage]] = {}
        for page in live:
            if page.page_type in ("index",):
                continue
            groups.setdefault(page.category_path or ("Communities" if page.page_type == "community" else
                                                     "Documents" if page.page_type == "summary" else "Pages"),
                              []).append(page)
        lines = ["# Index", ""]
        for path in sorted(groups):
            lines.append(f"## {path}")
            for page in sorted(groups[path], key=lambda p: p.title.lower()):
                lines.append(f"- [[{page.slug}|{page.title}]]" + (f": {page.summary}" if page.summary else ""))
            lines.append("")
        content = "\n".join(lines).strip() + "\n"
        index = await self._page(session, kb.id, "index", for_update=True)
        if index is None or index.content != content:
            await self.write_page(session, kb, index, slug="index", title="Index", content=content,
                                  summary="All pages of this wiki by folder.", page_type="index", source="pipeline")

    async def _detect_communities(self, session, kb, live: list[WikiPage]) -> int:
        """Louvain over page links + relations at two resolutions; changed communities (new
        membership or member versions) are queued for `wiki:communities`."""
        import networkx as nx

        members = [p for p in live if p.page_type in ("entity", "concept")]
        if len(members) < 3:
            return 0
        version = {p.slug: p.version for p in members}
        graph = nx.Graph()
        graph.add_nodes_from(version)
        for page in members:
            for target in page.out_links or []:
                if target in version and target != page.slug:
                    graph.add_edge(page.slug, target, weight=graph.get_edge_data(page.slug, target, {}).get("weight", 0) + 1)
        for rel in (await session.execute(sa.select(WikiRelation).where(WikiRelation.knowledge_base_id == kb.id))).scalars():
            if rel.source_slug in version and rel.target_slug in version and rel.source_slug != rel.target_slug:
                w = graph.get_edge_data(rel.source_slug, rel.target_slug, {}).get("weight", 0)
                graph.add_edge(rel.source_slug, rel.target_slug, weight=w + float(rel.weight or 1))
        if graph.number_of_edges() == 0:
            return 0
        existing = {(c.level, c.fingerprint): c for c in (await session.execute(sa.select(WikiCommunity).where(
            WikiCommunity.knowledge_base_id == kb.id))).scalars()}
        keep, changed = set(), 0
        for level, resolution in ((0, 1.0), (1, 0.5)):
            parts = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=7)
            for part in parts:
                if len(part) < 2:
                    continue
                slugs = sorted(part)
                fingerprint = hashlib.sha256(json.dumps([[s, version[s]] for s in slugs]).encode()).hexdigest()
                key = (level, fingerprint)
                keep.add(key)
                if key in existing:
                    continue
                session.add(WikiCommunity(knowledge_base_id=kb.id, level=level, member_slugs=slugs,
                                          fingerprint=fingerprint, page_slug=""))
                changed += 1
        for key, community in existing.items():
            if key not in keep:
                if community.page_slug:
                    page = await self._page(session, kb.id, community.page_slug, for_update=True)
                    if page is not None:
                        page.status = "archived"
                        await self.index_page(session, kb, page)
                await session.delete(community)
        if changed:
            await queue.enqueue(session, "wiki:communities", {"knowledge_base_id": kb.id},
                                owner_subject=kb.owner_subject, owner_org=kb.owner_org, knowledge_base_id=kb.id,
                                queue="wiki", unique_key=f"wiki-communities-{kb.id}", max_retry=2)
        return changed

    async def rebuild(self, ctx, kb_ref) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            if not self.enabled(kb):
                raise Invalid("the wiki is not enabled on this knowledge base (indexing_strategy.wiki_enabled)")
            items = [i for i in (await session.execute(sa.select(Knowledge).where(
                Knowledge.knowledge_base_id == kb.id, Knowledge.deleted_at.is_(None),
                Knowledge.parse_status == "completed", Knowledge.type != "faq"))).scalars().all() if feeds_wiki(i, kb)]
            for item in items:
                session.add(WikiPendingOp(knowledge_base_id=kb.id, knowledge_id=item.id, op="ingest",
                                          payload={"content_hash": item.content_hash, "rebuild": True}))
            if items:
                await self.trigger(session, kb)
            return {"queued": len(items)}

    async def rebuild_links(self, ctx, kb_ref) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref, write=True)
            kb_id = kb.id
        return await self.finalize(kb_id)
