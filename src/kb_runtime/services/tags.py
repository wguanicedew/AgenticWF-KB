"""
Tags (docs/api.md, Tags): per-KB tags, many per document through kb_knowledge_tag_relations;
one per FAQ entry (kb_chunks.tag_id). Tag filters match through the relation table, so
re-tagging needs no re-indexing.
"""

from __future__ import annotations

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.kb import Chunk, Embedding, Knowledge, KnowledgeBase, KnowledgeTag, KnowledgeTagRelation, utcnow
from kb_runtime import access
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound
from kb_runtime.services.knowledge_bases import KnowledgeBaseService

UNTAGGED = "__untagged__"


def tag_dict(tag: KnowledgeTag, knowledge_count: int = 0, chunk_count: int = 0) -> dict:
    return {"id": tag.id, "seq_id": tag.seq_id, "knowledge_base_id": tag.knowledge_base_id, "name": tag.name,
            "color": tag.color, "sort_order": tag.sort_order, "knowledge_count": knowledge_count,
            "chunk_count": chunk_count, "created_at": tag.created_at.isoformat() if tag.created_at else None}


async def resolve_tags(session, kb_ids: list[str], tag_refs: list[str]) -> tuple[list[str], bool]:
    """Tag ids (ids or integer seq_ids) within `kb_ids`; the bool says `__untagged__` was asked."""
    untagged = UNTAGGED in tag_refs
    refs = [t for t in tag_refs if t and t != UNTAGGED]
    if not refs:
        return [], untagged
    seqs = [int(t) for t in refs if str(t).isdigit()]
    ids = [t for t in refs if not str(t).isdigit()]
    rows = (await session.execute(sa.select(KnowledgeTag.id).where(
        KnowledgeTag.knowledge_base_id.in_(kb_ids),
        sa.or_(KnowledgeTag.id.in_(ids), KnowledgeTag.seq_id.in_(seqs))))).scalars().all()
    return list(rows), untagged


def item_tag_clause(tag_ids: list[str], untagged: bool):
    """WHERE clause on kb_knowledges: the item has one of `tag_ids`, or (`untagged`) no tag."""
    clauses = []
    if tag_ids:
        clauses.append(sa.exists().where(KnowledgeTagRelation.knowledge_id == Knowledge.id,
                                         KnowledgeTagRelation.tag_id.in_(tag_ids)))
    if untagged:
        clauses.append(~sa.exists().where(KnowledgeTagRelation.knowledge_id == Knowledge.id))
    return sa.or_(*clauses) if clauses else sa.false()


SEARCH_TAG_SQL = """AND (tag_id = ANY(:tag_ids)
     OR knowledge_id IN (SELECT knowledge_id FROM kb_knowledge_tag_relations WHERE tag_id = ANY(:tag_ids))
     {untagged})"""
UNTAGGED_SQL = ("OR (kb_embeddings.tag_id = '' AND NOT EXISTS (SELECT 1 FROM kb_knowledge_tag_relations r "
                "WHERE r.knowledge_id = kb_embeddings.knowledge_id))")


class TagService:
    def __init__(self, kbs: KnowledgeBaseService):
        self.kbs = kbs

    async def _kb_for_write(self, session, ctx, kb_ref) -> KnowledgeBase:
        kb = await self.kbs.load(session, ctx, kb_ref)
        if not access.can_contribute(ctx, kb):
            raise Forbidden("you may not change this knowledge base's tags")
        return kb

    async def list(self, ctx: AuthContext, kb_ref: str, keyword: str | None = None, page: int = 1,
                   page_size: int = 50) -> tuple[list[dict], int]:
        async with async_session_scope(commit=False) as session:
            kb = await self.kbs.load(session, ctx, kb_ref)
            query = sa.select(KnowledgeTag).where(KnowledgeTag.knowledge_base_id == kb.id)
            if keyword:
                query = query.where(KnowledgeTag.name.ilike(f"%{keyword}%"))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            tags = (await session.execute(query.order_by(KnowledgeTag.sort_order, KnowledgeTag.created_at)
                                          .offset((page - 1) * page_size).limit(page_size))).scalars().all()
            ids = [t.id for t in tags]
            k_counts = dict((await session.execute(sa.select(KnowledgeTagRelation.tag_id, sa.func.count()).join(
                Knowledge, Knowledge.id == KnowledgeTagRelation.knowledge_id).where(
                KnowledgeTagRelation.tag_id.in_(ids), Knowledge.deleted_at.is_(None),
                access.items_clause(Knowledge, ctx, [kb.id])).group_by(KnowledgeTagRelation.tag_id))).all()) if ids else {}
            c_counts = dict((await session.execute(sa.select(Chunk.tag_id, sa.func.count()).where(
                Chunk.tag_id.in_(ids)).group_by(Chunk.tag_id))).all()) if ids else {}
            return [tag_dict(t, k_counts.get(t.id, 0), c_counts.get(t.id, 0)) for t in tags], total

    async def create(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        name = (body.get("name") or "").strip()
        if not name or len(name) > 128:
            raise Invalid("a tag needs a name of at most 128 characters")
        async with async_session_scope() as session:
            kb = await self._kb_for_write(session, ctx, kb_ref)
            existing = (await session.execute(sa.select(KnowledgeTag).where(
                KnowledgeTag.knowledge_base_id == kb.id, KnowledgeTag.name == name))).scalar_one_or_none()
            if existing is not None:
                raise Conflict("this tag exists already", tag_dict(existing))
            tag = KnowledgeTag(knowledge_base_id=kb.id, name=name, color=body.get("color") or "",
                               sort_order=int(body.get("sort_order") or 0))
            session.add(tag)
            await session.flush()
            return tag_dict(tag)

    async def update(self, ctx: AuthContext, kb_ref: str, tag_id: str, body: dict) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb_for_write(session, ctx, kb_ref)
            tag = await session.get(KnowledgeTag, tag_id)
            if tag is None or tag.knowledge_base_id != kb.id:
                raise NotFound("tag not found")
            if body.get("name"):
                tag.name = body["name"].strip()
            if "color" in body:
                tag.color = body["color"] or ""
            if "sort_order" in body:
                tag.sort_order = int(body["sort_order"] or 0)
            tag.updated_at = utcnow()
            try:
                await session.flush()
            except sa.exc.IntegrityError:
                raise Conflict("another tag has this name")
            return tag_dict(tag)

    async def delete(self, ctx: AuthContext, kb_ref: str, tag_id: str, force: bool = False) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb_for_write(session, ctx, kb_ref)
            tag = await session.get(KnowledgeTag, tag_id)
            if tag is None or tag.knowledge_base_id != kb.id:
                raise NotFound("tag not found")
            used = (await session.execute(sa.select(sa.func.count()).select_from(KnowledgeTagRelation).where(
                KnowledgeTagRelation.tag_id == tag_id))).scalar_one()
            if used and not force:
                raise Invalid(f"{used} document(s) use this tag; delete with force=true to detach it")
            await session.execute(sa.update(Chunk).where(Chunk.tag_id == tag_id).values(tag_id=""))
            await session.execute(sa.update(Embedding).where(Embedding.tag_id == tag_id).values(tag_id=""))
            await session.execute(sa.delete(KnowledgeTagRelation).where(KnowledgeTagRelation.tag_id == tag_id))
            await session.delete(tag)
            return {"id": tag_id, "deleted": True, "detached": used}

    async def set_item_tags(self, session, ctx: AuthContext, item: Knowledge, tag_refs: list[str]) -> list[str]:
        """Replace an item's tag set (inside the caller's transaction); `[]` clears it."""
        tag_ids, _ = await resolve_tags(session, [item.knowledge_base_id], tag_refs)
        if len(tag_ids) != len([t for t in tag_refs if t]):
            raise Invalid("unknown tag for this knowledge base")
        await session.execute(sa.delete(KnowledgeTagRelation).where(KnowledgeTagRelation.knowledge_id == item.id))
        for tag_id in tag_ids:
            session.add(KnowledgeTagRelation(knowledge_id=item.id, tag_id=tag_id))
        return tag_ids

    async def item_tags(self, session, knowledge_ids: list[str]) -> dict[str, list[str]]:
        if not knowledge_ids:
            return {}
        rows = (await session.execute(sa.select(KnowledgeTagRelation.knowledge_id, KnowledgeTagRelation.tag_id).where(
            KnowledgeTagRelation.knowledge_id.in_(knowledge_ids)))).all()
        out: dict[str, list[str]] = {}
        for knowledge_id, tag_id in rows:
            out.setdefault(knowledge_id, []).append(tag_id)
        return out
