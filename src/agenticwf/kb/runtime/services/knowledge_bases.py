"""
KnowledgeBaseService: create, list, read, change and delete knowledge bases
(docs/data-model.md, Ownership and visibility; docs/api.md, Knowledge bases).

A KB is addressed by id or by `<owner>:<slug>` (e.g. `local:wguan:personal`, `system:eic-docs`);
the slug is the part after the last colon. Every user has a private `personal` KB, created on
their first request. A KB the principal cannot see is "not found".
"""

from __future__ import annotations

import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.utils import slugify
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import (Chunk, Embedding, Knowledge, KnowledgeBase, default_chunking_config,
                                   default_indexing_strategy, utcnow)
from agenticwf.kb.runtime import access, features
from agenticwf.kb.runtime.errors import Conflict, Forbidden, Invalid, NotFound
from agenticwf.kb.runtime.storage import LocalStorage, owner_scope

KB_TYPES = ("document", "faq", "records")
PERSONAL = "personal"
UPDATABLE = ("name", "description", "chunking_config", "indexing_strategy", "retrieval_config", "freshness_config",
             "question_generation_config", "vlm_config", "image_processing_config", "is_pinned", "records_config",
             "wiki_config", "memory_config")


def document_defaults(owner: str) -> dict:
    """A new document KB's wiki and memory switches, from `[knowledge_bases]`; a system KB feeds
    no one's memory."""
    return {"indexing_strategy": {**default_indexing_strategy(), "wiki_enabled": features.kb_default("wiki_enabled")},
            "memory_config": {"enabled": features.kb_default("memory_enabled") and owner != access.SYSTEM_OWNER}}


def kb_dict(kb: KnowledgeBase, knowledge_count: int | None = None) -> dict:
    data = {"id": kb.id, "slug": kb.slug, "ref": f"{kb.owner_subject}:{kb.slug}", "name": kb.name,
            "description": kb.description, "type": kb.type, "owner_subject": kb.owner_subject,
            "visibility": kb.visibility, "groups": list(kb.groups or []),
            "default_item_visibility": kb.default_item_visibility, "contribute": kb.contribute,
            "chunking_config": kb.chunking_config, "indexing_strategy": kb.indexing_strategy,
            "retrieval_config": kb.retrieval_config, "freshness_config": kb.freshness_config,
            "question_generation_config": kb.question_generation_config, "vlm_config": kb.vlm_config,
            "image_processing_config": kb.image_processing_config, "records_config": kb.records_config,
            "wiki_config": kb.wiki_config, "memory_config": kb.memory_config,
            "embedding_model_id": kb.embedding_model_id, "summary_model_id": kb.summary_model_id,
            "rerank_model_id": kb.rerank_model_id, "is_pinned": kb.is_pinned, "is_temporary": kb.is_temporary,
            "storage_used": kb.storage_used, "storage_quota": kb.storage_quota,
            "created_at": kb.created_at.isoformat() if kb.created_at else None,
            "updated_at": kb.updated_at.isoformat() if kb.updated_at else None}
    if kb.type == "faq":
        data["faq_config"] = (kb.indexing_strategy or {}).get("faq")
    if knowledge_count is not None:
        data["knowledge_count"] = knowledge_count
    return data


def split_ref(ref: str) -> tuple[str | None, str]:
    """(owner, slug) of `<owner>:<slug>`, or (None, ref) for an id."""
    owner, sep, slug = ref.rpartition(":")
    return (owner, slug) if sep and owner else (None, ref)


async def purge_kb_rows(session, kb_id: str) -> None:
    """Delete every row of a KB: each table with `knowledge_base_id` (children first; rows that
    hang off them go by ON DELETE CASCADE), then the KB itself."""
    from agenticwf.kb.core.orm.base.session import BASE

    for table in reversed(BASE.metadata.sorted_tables):
        if "knowledge_base_id" in table.c and table.name != KnowledgeBase.__tablename__:
            await session.execute(sa.delete(table).where(table.c.knowledge_base_id == kb_id))
    await session.execute(sa.delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))


class KnowledgeBaseService:
    def __init__(self, models, storage: LocalStorage):
        self.models = models
        self.storage = storage
        self.memory = None              # MemoryService: deleting a KB forgets the memory its documents fed

    # -- lookups -------------------------------------------------------------------------------------
    @staticmethod
    async def load(session, ctx: AuthContext, ref: str, for_update: bool = False) -> KnowledgeBase:
        """The KB `ref` (id or `<owner>:<slug>`) if the principal can see it, else NotFound."""
        owner, slug = split_ref(ref)
        query = sa.select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
        if owner is None:
            query = query.where(KnowledgeBase.id == slug)
        else:
            query = query.where(KnowledgeBase.owner_subject == owner, KnowledgeBase.slug == slug)
        if for_update:
            query = query.with_for_update()
        kb = (await session.execute(query)).scalar_one_or_none()
        if kb is None or not access.can_see(ctx, kb):
            raise NotFound("knowledge base not found")
        return kb

    @staticmethod
    async def readable_ids(session, ctx: AuthContext, refs: list[str] | None = None) -> list[str]:
        """Ids of the KBs the principal can see, narrowed to `refs` when given (each must be
        visible, else NotFound)."""
        if refs:
            ids = []
            for ref in dict.fromkeys(refs):
                ids.append((await KnowledgeBaseService.load(session, ctx, ref)).id)
            return ids
        query = sa.select(KnowledgeBase.id).where(KnowledgeBase.deleted_at.is_(None),
                                                  access.visible_clause(KnowledgeBase, ctx))
        return list((await session.execute(query)).scalars())

    async def ensure_personal(self, ctx: AuthContext) -> str | None:
        """The principal's personal KB id, created on first use (not for machine clients)."""
        if ctx.is_machine:
            return None
        async with async_session_scope() as session:
            kb = (await session.execute(sa.select(KnowledgeBase).where(
                KnowledgeBase.owner_subject == ctx.subject, KnowledgeBase.slug == PERSONAL,
                KnowledgeBase.deleted_at.is_(None)))).scalar_one_or_none()
            if kb is not None:
                return kb.id
            kb = KnowledgeBase(owner_subject=ctx.subject, owner_org=ctx.organization_id, visibility="private",
                               default_item_visibility="private", slug=PERSONAL, name="Personal",
                               description="Your private knowledge base",
                               embedding_model_id=await self.models.default_id("Embedding"),
                               rerank_model_id=await self.models.default_id("Rerank"),
                               **document_defaults(ctx.subject))
            if kb.embedding_model_id is None:
                kb.indexing_strategy = {**kb.indexing_strategy, "vector_enabled": False}
            session.add(kb)
            try:
                await session.flush()
            except sa.exc.IntegrityError:     # created by a concurrent request
                await session.rollback()
                return None
            return kb.id

    # -- the API -------------------------------------------------------------------------------------
    async def list(self, ctx: AuthContext, page: int = 1, page_size: int = 20, shared_only: bool = False,
                   kb_type: str | None = None) -> tuple[list[dict], int]:
        async with async_session_scope(commit=False) as session:
            query = sa.select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None),
                                                   access.visible_clause(KnowledgeBase, ctx))
            if shared_only:
                query = query.where(KnowledgeBase.owner_subject != ctx.subject)
            if kb_type:
                query = query.where(KnowledgeBase.type == kb_type)
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = list((await session.execute(query.order_by(KnowledgeBase.is_pinned.desc(), KnowledgeBase.created_at)
                                               .offset((page - 1) * page_size).limit(page_size))).scalars())
            counts = await self._knowledge_counts(session, ctx, [kb.id for kb in rows])
            return [kb_dict(kb, counts.get(kb.id, 0)) for kb in rows], total

    @staticmethod
    async def _knowledge_counts(session, ctx, kb_ids) -> dict[str, int]:
        if not kb_ids:
            return {}
        rows = await session.execute(sa.select(Knowledge.knowledge_base_id, sa.func.count()).where(
            Knowledge.deleted_at.is_(None), access.items_clause(Knowledge, ctx, kb_ids)).group_by(
            Knowledge.knowledge_base_id))
        return dict(rows.all())

    async def get(self, ctx: AuthContext, ref: str) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self.load(session, ctx, ref)
            counts = await self._knowledge_counts(session, ctx, [kb.id])
            return kb_dict(kb, counts.get(kb.id, 0))

    async def _check_models(self, ctx: AuthContext, body: dict) -> dict:
        out = {}
        for field, mtype in (("embedding_model_id", "Embedding"), ("rerank_model_id", "Rerank"),
                             ("summary_model_id", "KnowledgeQA")):
            if body.get(field):
                out[field] = await self.models.usable(ctx, body[field], mtype)
        return out

    async def create(self, ctx: AuthContext, body: dict) -> dict:
        name = (body.get("name") or "").strip()
        if not name:
            raise Invalid("name is required")
        kb_type = body.get("type") or "document"
        if kb_type not in KB_TYPES:
            raise Invalid(f"type must be one of {', '.join(KB_TYPES)} in this version")
        if kb_type != "document" and not features.enabled(kb_type):
            raise Invalid(f"{kb_type} knowledge bases are switched off in this deployment ([features] {kb_type})")
        visibility = body.get("visibility") or "private"
        groups = access.check_share(ctx, visibility, body.get("groups") or [])
        if visibility == "all" and not access.is_admin(ctx):
            raise Forbidden("only an admin shares a knowledge base with everyone")
        owner = access.SYSTEM_OWNER if body.get("system") else ctx.subject
        if owner == access.SYSTEM_OWNER and not access.is_admin(ctx):
            raise Forbidden("only an admin creates system knowledge bases")
        item_visibility = body.get("default_item_visibility") or ("private" if visibility == "private" else "all")
        if item_visibility not in access.VISIBILITIES:
            raise Invalid("default_item_visibility must be private, groups or all")
        contribute = body.get("contribute") or "owner"
        if contribute not in ("owner", "readers"):
            raise Invalid("contribute must be owner or readers")
        models = await self._check_models(ctx, body)
        models.setdefault("embedding_model_id", await self.models.default_id("Embedding"))
        models.setdefault("rerank_model_id", await self.models.default_id("Rerank"))
        defaults = document_defaults(owner) if kb_type == "document" else \
            {"indexing_strategy": default_indexing_strategy(), "memory_config": {}}
        strategy = {**defaults["indexing_strategy"], **(body.get("indexing_strategy") or {})}
        if kb_type == "faq":
            from agenticwf.kb.runtime.faq.rules import check_faq_config

            strategy["faq"] = check_faq_config(body.get("faq_config") or strategy.get("faq"))
        if not models["embedding_model_id"]:
            strategy["vector_enabled"] = False
        slug = slugify(body.get("slug") or name)
        async with async_session_scope() as session:
            taken = set((await session.execute(sa.select(KnowledgeBase.slug).where(
                KnowledgeBase.owner_subject == owner, KnowledgeBase.deleted_at.is_(None),
                KnowledgeBase.slug.like(f"{slug}%")))).scalars())
            if body.get("slug") and slug in taken:
                raise Conflict(f"you already have a knowledge base {slug!r}")
            base, n = slug, 2
            while slug in taken:
                slug, n = f"{base}-{n}", n + 1
            kb = KnowledgeBase(owner_subject=owner, owner_org=ctx.organization_id, visibility=visibility, groups=groups,
                               default_item_visibility=item_visibility, contribute=contribute, slug=slug, name=name,
                               description=body.get("description") or "", type=kb_type,
                               chunking_config={**default_chunking_config(), **(body.get("chunking_config") or {})},
                               indexing_strategy=strategy, retrieval_config=body.get("retrieval_config") or {},
                               freshness_config=body.get("freshness_config") or {},
                               records_config=body.get("records_config") or {}, wiki_config=body.get("wiki_config") or {},
                               memory_config={**defaults["memory_config"], **(body.get("memory_config") or {})},
                               **models)
            session.add(kb)
            await session.flush()
            return kb_dict(kb, 0)

    async def update(self, ctx: AuthContext, ref: str, body: dict) -> dict:
        models = await self._check_models(ctx, body)
        async with async_session_scope() as session:
            kb = await self.load(session, ctx, ref, for_update=True)
            if not access.can_manage(ctx, kb):
                raise Forbidden("only the owner changes a knowledge base")
            if "embedding_model_id" in models and models["embedding_model_id"] != kb.embedding_model_id:
                has_items = (await session.execute(sa.select(Knowledge.id).where(
                    Knowledge.knowledge_base_id == kb.id, Knowledge.deleted_at.is_(None)).limit(1))).first()
                if has_items:
                    raise Conflict("the embedding model is fixed once the KB has knowledge; use /reindex")
            for field, value in models.items():
                setattr(kb, field, value)
            for field in UPDATABLE:
                if field in body and body[field] is not None:
                    value = body[field]
                    if field == "chunking_config":
                        value = {**default_chunking_config(), **kb.chunking_config, **value}
                    elif field == "indexing_strategy":
                        value = {**kb.indexing_strategy, **value}
                    setattr(kb, field, value)
            if body.get("faq_config") is not None:
                if kb.type != "faq":
                    raise Invalid("faq_config applies to FAQ knowledge bases")
                from agenticwf.kb.runtime.faq.rules import check_faq_config

                kb.indexing_strategy = {**kb.indexing_strategy, "faq": check_faq_config(body["faq_config"])}
            if "visibility" in body or "groups" in body:
                visibility = body.get("visibility") or kb.visibility
                if visibility == "all" and not access.is_admin(ctx):
                    raise Forbidden("only an admin shares a knowledge base with everyone")
                kb.groups = access.check_share(ctx, visibility, body.get("groups", kb.groups) or [])
                kb.visibility = visibility
            for field, allowed in (("default_item_visibility", access.VISIBILITIES), ("contribute", ("owner", "readers"))):
                if body.get(field):
                    if body[field] not in allowed:
                        raise Invalid(f"{field} must be one of {', '.join(allowed)}")
                    setattr(kb, field, body[field])
            kb.updated_at = utcnow()
            await session.flush()
            return kb_dict(kb)

    async def delete(self, ctx: AuthContext, ref: str, purge: bool = False) -> None:
        """Soft delete (purged by housekeeping after `purge_after_days`), or with `purge` (or
        `[maintenance] purge_on_delete`) every row of the KB at once."""
        from agenticwf.kb.runtime.services.knowledge import purge_on_delete

        async with async_session_scope() as session:
            kb = await self.load(session, ctx, ref, for_update=True)
            if not access.can_manage(ctx, kb):
                raise Forbidden("only the owner deletes a knowledge base")
            now = utcnow()
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_base_id == kb.id))
            await session.execute(sa.delete(Chunk).where(Chunk.knowledge_base_id == kb.id))
            if self.memory is not None:
                ids = list((await session.execute(sa.select(Knowledge.id).where(
                    Knowledge.knowledge_base_id == kb.id))).scalars())
                await self.memory.forget_documents(session, kb.owner_subject, ids)
            await session.execute(sa.update(Knowledge).where(Knowledge.knowledge_base_id == kb.id,
                                                             Knowledge.deleted_at.is_(None))
                                  .values(deleted_at=now, parse_status="deleted", description="", meta={},
                                          citation={}, profile={}))
            kb.deleted_at, kb.storage_used = now, 0
            owner = kb.owner_subject
            kb_id = kb.id
            if purge or purge_on_delete():
                await purge_kb_rows(session, kb_id)
        self.storage.delete_prefix(f"{owner_scope(owner)}/{kb_id}")
