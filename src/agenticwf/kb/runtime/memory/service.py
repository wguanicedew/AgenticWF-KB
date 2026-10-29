"""
Temporal memory (docs/data-model.md, Temporal memory; docs/api.md, Temporal memory): episodes,
entities and bi-temporal facts on PostgreSQL. Memory is not inside a KB: every row belongs to
one owner (`owner_subject`) and is visible to that owner only; `group_id` namespaces it.

    add_episode            stored `pending`; `memory:process` extracts it in the background
    memory:process         LLM extraction → entity resolution (names, aliases, embeddings, then an
                           LLM check) → facts; a contradicted current fact gets `invalid_at` (event
                           time) and `invalidated_by`, never deleted
    delete_episode         facts only it supported go, and the invalidations they caused are reverted
    delete_fact / _entity  one fact, or an entity with every fact about it (invalidations reverted)
    redact                 matching text out of episodes, facts and entities (docs/api.md,
                           Temporal memory); `memory:reembed` then embeds the changed rows again
    search                 BM25 + vector + the focal entity's 2-hop neighbourhood, RRF, MMR; current
                           facts, or `as_of` (event time) / `known_at` (system time)

Event time is valid_at / invalid_at (when it was true in the world); system time is created_at /
expired_at (when the KB learned or retired it).
"""

from __future__ import annotations

import datetime
import json
import re

import anyio
import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope, get_engine
from agenticwf.kb.core.orm.models.ext import MemoryEntity, MemoryEpisode, MemoryFact
from agenticwf.kb.core.orm.models.indexes import ensure_vector_index
from agenticwf.kb.core.orm.models.kb import new_id, utcnow
from agenticwf.kb.runtime import features
from agenticwf.kb.runtime.errors import Invalid, NotFound
from agenticwf.kb.runtime.models.clients import ModelError
from agenticwf.kb.runtime.prompts import render
from agenticwf.kb.runtime.redaction import Redactor
from agenticwf.kb.runtime.retrieval.fusion import rrf
from agenticwf.kb.runtime.retrieval.rerank import mmr
from agenticwf.kb.runtime.tasks import queue
from agenticwf.kb.runtime.tasks.runner import HandlerSpec, Permanent, TaskContext

LOG = get_logger(__name__)

SOURCES = ("conversation", "document", "manual")
MAX_CONTENT = 50_000
SAME_ENTITY = 0.92            # embedding similarity above which the LLM is asked "same entity?"
CANDIDATES = 50


def setting(name: str, default: str) -> str:
    return config_get("memory", name).strip() if config_has_option("memory", name) else default


def _time(value, name: str):
    if value in (None, ""):
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    try:
        parsed = datetime.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        raise Invalid(f"{name} is not an ISO-8601 time")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _iso(value):
    return value.isoformat() if value else None


def _key(name: str) -> str:
    return re.sub(r"[\W_]+", " ", str(name or "").casefold()).strip()


def episode_dict(e: MemoryEpisode) -> dict:
    return {"id": e.id, "group_id": e.group_id, "name": e.name, "source": e.source, "source_ref": e.source_ref,
            "content": e.content, "reference_time": _iso(e.reference_time), "status": e.status, "error": e.error,
            "created_at": _iso(e.created_at), "processed_at": _iso(e.processed_at)}


def fact_dict(f: MemoryFact, names: dict | None = None) -> dict:
    names = names or {}
    return {"id": f.id, "group_id": f.group_id, "source_entity_id": f.source_entity_id,
            "source": names.get(f.source_entity_id), "relation": f.relation,
            "target_entity_id": f.target_entity_id, "target": names.get(f.target_entity_id), "fact": f.fact,
            "valid_at": _iso(f.valid_at), "invalid_at": _iso(f.invalid_at), "created_at": _iso(f.created_at),
            "expired_at": _iso(f.expired_at), "invalidated_by": f.invalidated_by or None,
            "episode_ids": f.episode_ids}


def entity_dict(e: MemoryEntity) -> dict:
    return {"id": e.id, "group_id": e.group_id, "name": e.name, "entity_type": e.entity_type, "summary": e.summary,
            "aliases": e.aliases, "created_at": _iso(e.created_at)}


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON")
    return json.loads(match.group(0))


class MemoryService:
    def __init__(self, models):
        self.models = models

    @staticmethod
    def _check():
        if not features.enabled("memory"):
            raise NotFound("memory is switched off in this deployment")

    @staticmethod
    def _group(group_id) -> str:
        group = str(group_id or setting("default_group", "default")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", group):
            raise Invalid("group_id is letters, digits and _.:- (≤ 128)")
        return group

    @staticmethod
    def _mine(model, ctx: AuthContext, group: str | None = None):
        clause = model.owner_subject == ctx.subject
        return sa.and_(clause, model.group_id == group) if group else clause

    # == episodes ============================================================================================
    async def add_episode(self, ctx: AuthContext, body: dict) -> dict:
        self._check()
        content = str(body.get("content") or "").strip()
        if not content:
            raise Invalid("content is required")
        if len(content) > MAX_CONTENT:
            raise Invalid(f"an episode is at most {MAX_CONTENT} characters")
        source = body.get("source") or "manual"
        if source not in SOURCES:
            raise Invalid(f"source must be one of {', '.join(SOURCES)}")
        async with async_session_scope() as session:
            episode = await self.store_episode(session, ctx.subject, ctx.organization_id, content,
                                               group_id=self._group(body.get("group_id")), source=source,
                                               name=str(body.get("name") or "")[:255],
                                               source_ref=str(body.get("source_ref") or "")[:255],
                                               reference_time=_time(body.get("reference_time"), "reference_time"))
            return episode_dict(episode)

    async def store_episode(self, session, owner: str, org: str | None, content: str, group_id: str = "default",
                            source: str = "manual", name: str = "", source_ref: str = "",
                            reference_time=None) -> MemoryEpisode:
        episode = MemoryEpisode(id=new_id(), owner_subject=owner, owner_org=org, group_id=group_id, name=name,
                                source=source, source_ref=source_ref, content=content,
                                reference_time=reference_time or utcnow(), status="pending")
        session.add(episode)
        await session.flush()
        # one owner's episodes are processed in order (entity resolution sees earlier ones)
        await queue.enqueue(session, "memory:process", {"episode_id": episode.id}, owner_subject=owner, owner_org=org,
                            queue="postprocess", max_retry=3)
        return episode

    async def list_episodes(self, ctx, group_id=None, page=1, page_size=20) -> tuple[list[dict], int]:
        self._check()
        async with async_session_scope(commit=False) as session:
            query = sa.select(MemoryEpisode).where(self._mine(MemoryEpisode, ctx, group_id))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(MemoryEpisode.reference_time.desc(), MemoryEpisode.id)
                                          .offset((page - 1) * page_size).limit(page_size))).scalars().all()
            return [episode_dict(e) for e in rows], total

    async def _episode(self, session, ctx, episode_id, for_update=False) -> MemoryEpisode:
        episode = await session.get(MemoryEpisode, episode_id, with_for_update=for_update)
        if episode is None or episode.owner_subject != ctx.subject:
            raise NotFound("episode not found")
        return episode

    async def get_episode(self, ctx, episode_id) -> dict:
        self._check()
        async with async_session_scope(commit=False) as session:
            episode = await self._episode(session, ctx, episode_id)
            facts = (await session.execute(sa.select(MemoryFact).where(
                MemoryFact.owner_subject == ctx.subject, MemoryFact.episode_ids.contains([episode.id])))).scalars().all()
            names = await self._names(session, facts)
            return {**episode_dict(episode), "facts": [fact_dict(f, names) for f in facts]}

    async def delete_episode(self, ctx, episode_id) -> dict:
        """Facts only this episode supported are removed, and the invalidations they caused are
        reverted; facts other episodes support lose this one."""
        self._check()
        async with async_session_scope() as session:
            episode = await self._episode(session, ctx, episode_id, for_update=True)
            removed, reverted = await self._forget_episode(session, episode)
            return {"deleted": True, "facts_removed": removed, "invalidations_reverted": reverted}

    async def _forget_episode(self, session, episode: MemoryEpisode) -> tuple[int, int]:
        """(facts removed, invalidations reverted), inside the caller's transaction."""
        facts = (await session.execute(sa.select(MemoryFact).where(
            MemoryFact.owner_subject == episode.owner_subject, MemoryFact.episode_ids.contains([episode.id]))
            .with_for_update())).scalars().all()
        removed = [f for f in facts if f.episode_ids == [episode.id]]
        for fact in facts:
            if fact not in removed:
                fact.episode_ids = [e for e in fact.episode_ids if e != episode.id]
        reverted = await self._remove_facts(session, episode.owner_subject, removed)
        await session.delete(episode)
        return len(removed), reverted

    async def forget_documents(self, session, owner: str, knowledge_ids: list[str]) -> int:
        """Forget the episodes deleted documents fed into `owner`'s memory (source_ref
        `<knowledge id>@<version>`); returns how many."""
        if not knowledge_ids:
            return 0
        episodes = (await session.execute(sa.select(MemoryEpisode).where(
            MemoryEpisode.owner_subject == owner, MemoryEpisode.source == "document",
            sa.func.split_part(MemoryEpisode.source_ref, "@", 1).in_(knowledge_ids)).with_for_update())).scalars().all()
        for episode in episodes:
            await self._forget_episode(session, episode)
        return len(episodes)

    @staticmethod
    async def _remove_facts(session, owner: str, facts: list[MemoryFact]) -> int:
        """Delete facts and revert the invalidations they caused; returns how many were reverted."""
        ids = [f.id for f in facts]
        if not ids:
            return 0
        result = await session.execute(sa.update(MemoryFact).where(
            MemoryFact.owner_subject == owner, MemoryFact.invalidated_by.in_(ids)).values(
            invalid_at=None, invalidated_by=""))
        await session.execute(sa.delete(MemoryFact).where(MemoryFact.id.in_(ids)))
        return result.rowcount or 0

    async def delete_fact(self, ctx, fact_id) -> dict:
        """Forget one fact. The episodes it came from keep their text: redact them to remove it."""
        self._check()
        async with async_session_scope() as session:
            fact = await session.get(MemoryFact, fact_id, with_for_update=True)
            if fact is None or fact.owner_subject != ctx.subject:
                raise NotFound("fact not found")
            reverted = await self._remove_facts(session, ctx.subject, [fact])
            return {"deleted": True, "invalidations_reverted": reverted, "episode_ids": list(fact.episode_ids)}

    async def delete_entity(self, ctx, entity_id) -> dict:
        """Forget an entity: the entity row and every fact it is the source or target of."""
        self._check()
        async with async_session_scope() as session:
            entity = await session.get(MemoryEntity, entity_id, with_for_update=True)
            if entity is None or entity.owner_subject != ctx.subject or entity.deleted_at is not None:
                raise NotFound("entity not found")
            facts = (await session.execute(sa.select(MemoryFact).where(
                MemoryFact.owner_subject == ctx.subject,
                sa.or_(MemoryFact.source_entity_id == entity.id, MemoryFact.target_entity_id == entity.id))
                .with_for_update())).scalars().all()
            reverted = await self._remove_facts(session, ctx.subject, facts)
            episodes = sorted({e for f in facts for e in f.episode_ids})
            await session.delete(entity)
            return {"deleted": True, "facts_removed": len(facts), "invalidations_reverted": reverted,
                    "episode_ids": episodes}

    # == redaction ===========================================================================================
    async def redact(self, ctx, body: dict) -> dict:
        """Remove text matching `terms` / `patterns` from the caller's memory (one group, or all)."""
        self._check()
        redactor, dry_run = Redactor.from_body(body), bool(body.get("dry_run"))
        group = self._group(body["group_id"]) if body.get("group_id") else None
        async with async_session_scope() as session:
            return {"dry_run": dry_run, **await self.scrub(session, redactor, ctx.subject, group=group,
                                                            dry_run=dry_run)}

    async def scrub(self, session, redactor: Redactor, owner: str, group: str | None = None,
                    source_ref_prefix: str | None = None, dry_run: bool = False) -> dict:
        """Redact the owner's episodes, facts and entities inside the caller's transaction. With
        `source_ref_prefix` only the episodes from that source (a document) and what was learned
        from them. Changed facts and entities are embedded again by `memory:reembed`."""
        episodes_q = sa.select(MemoryEpisode).where(MemoryEpisode.owner_subject == owner)
        if group:
            episodes_q = episodes_q.where(MemoryEpisode.group_id == group)
        if source_ref_prefix:
            episodes_q = episodes_q.where(MemoryEpisode.source_ref.startswith(source_ref_prefix, autoescape=True))
        episodes = (await session.execute(episodes_q.with_for_update())).scalars().all()
        facts_q = sa.select(MemoryFact).where(MemoryFact.owner_subject == owner)
        if group:
            facts_q = facts_q.where(MemoryFact.group_id == group)
        if source_ref_prefix:
            if not episodes:
                return {"episodes": 0, "facts": 0, "entities": 0, "matches": 0}
            facts_q = facts_q.where(sa.or_(*(MemoryFact.episode_ids.contains([e.id]) for e in episodes)))
        facts = (await session.execute(facts_q.with_for_update())).scalars().all()
        entities_q = sa.select(MemoryEntity).where(MemoryEntity.owner_subject == owner,
                                                   MemoryEntity.deleted_at.is_(None))
        if group:
            entities_q = entities_q.where(MemoryEntity.group_id == group)
        if source_ref_prefix:
            touched = {i for f in facts for i in (f.source_entity_id, f.target_entity_id)}
            entities_q = entities_q.where(MemoryEntity.id.in_(touched or {""}))
        entities = (await session.execute(entities_q.with_for_update())).scalars().all()

        counts, matches = {"episodes": 0, "facts": 0, "entities": 0}, 0
        changed_facts, changed_entities = [], []
        for episode in episodes:
            found = redactor.count(episode.content) + redactor.count(episode.name)
            if found:
                counts["episodes"], matches = counts["episodes"] + 1, matches + found
                if not dry_run:
                    episode.content, episode.name = redactor.apply(episode.content)[0], redactor.apply(episode.name)[0]
        for fact in facts:
            found = redactor.count(fact.fact)
            if found:
                counts["facts"], matches = counts["facts"] + 1, matches + found
                if not dry_run:
                    fact.fact = redactor.apply(fact.fact)[0]
                    changed_facts.append(fact.id)
        for entity in entities:
            found = redactor.count(entity.name) + redactor.count(entity.summary) + \
                sum(redactor.count(a) for a in entity.aliases or [])
            if found:
                counts["entities"], matches = counts["entities"] + 1, matches + found
                if not dry_run:
                    entity.name = redactor.apply(entity.name)[0] or "[REDACTED]"
                    entity.summary = redactor.apply(entity.summary)[0]
                    entity.aliases = [a for a in redactor.scrub(list(entity.aliases or [])) if a]
                    changed_entities.append(entity.id)
        if changed_facts or changed_entities:
            # the old vectors were made from the old text: drop them now, embed the new text later
            if changed_facts:
                await session.execute(sa.update(MemoryFact).where(MemoryFact.id.in_(changed_facts))
                                      .values(embedding=None, dimension=0))
            if changed_entities:
                await session.execute(sa.update(MemoryEntity).where(MemoryEntity.id.in_(changed_entities))
                                      .values(embedding=None, dimension=0))
            await queue.enqueue(session, "memory:reembed", {"fact_ids": changed_facts, "entity_ids": changed_entities},
                                owner_subject=owner, queue="postprocess", max_retry=3)
        return {**counts, "matches": matches}

    # == processing ===========================================================================================
    def handlers(self) -> dict[str, HandlerSpec]:
        return {"memory:process": HandlerSpec(self._process_task, 600, self._process_failed),
                "memory:reembed": HandlerSpec(self._reembed_task, 600)}

    async def _reembed_task(self, ctx: TaskContext) -> None:
        embedder = await self._embedder()
        if embedder is None:
            return
        async with async_session_scope(commit=False) as session:
            facts = (await session.execute(sa.select(MemoryFact.id, MemoryFact.fact).where(
                MemoryFact.id.in_(ctx.payload.get("fact_ids") or [""])))).all()
            entities = (await session.execute(sa.select(MemoryEntity.id, MemoryEntity.name, MemoryEntity.summary).where(
                MemoryEntity.id.in_(ctx.payload.get("entity_ids") or [""])))).all()
        fact_vectors = await embedder.embed([f.fact for f in facts]) if facts else []
        entity_vectors = await embedder.embed([f"{e.name}: {e.summary or ''}" for e in entities]) if entities else []
        dim = int(embedder.dimension)
        async with async_session_scope() as session:
            for (fact_id, _), vector in zip(facts, fact_vectors):
                await session.execute(sa.update(MemoryFact).where(MemoryFact.id == fact_id)
                                      .values(embedding=vector, dimension=dim))
            for (entity_id, *_), vector in zip(entities, entity_vectors):
                await session.execute(sa.update(MemoryEntity).where(MemoryEntity.id == entity_id)
                                      .values(embedding=vector, dimension=dim))

    async def _process_failed(self, ctx: TaskContext, error: str) -> None:
        async with async_session_scope() as session:
            episode = await session.get(MemoryEpisode, ctx.payload["episode_id"])
            if episode is not None:
                episode.status, episode.error = "failed", error[:2000]

    async def _chat(self):
        model = setting("model", "") or await self.models.default_id("KnowledgeQA")
        if not model:
            raise Permanent("no chat model for memory extraction ([memory] model)")
        return await self.models.chat(model)

    async def _embedder(self):
        model = await self.models.default_id("Embedding")
        if not model:
            return None
        embedder = await self.models.embedder(model)
        await anyio.to_thread.run_sync(ensure_vector_index_for_memory, embedder.dimension)
        return embedder

    async def _process_task(self, ctx: TaskContext) -> None:
        async with async_session_scope(commit=False) as session:
            episode = await session.get(MemoryEpisode, ctx.payload["episode_id"])
            if episode is None or episode.status == "processed":
                return
            session.expunge(episode)
            known = (await session.execute(sa.select(MemoryEntity.name).where(
                MemoryEntity.owner_subject == episode.owner_subject, MemoryEntity.group_id == episode.group_id,
                MemoryEntity.deleted_at.is_(None)).order_by(MemoryEntity.created_at.desc()).limit(100))).scalars().all()
        chat = await self._chat()
        reply = await chat.complete([
            {"role": "system", "content": render("memory", "extract_system")},
            {"role": "user", "content": render("memory", "extract_user", reference_time=episode.reference_time.isoformat(),
                                               known=", ".join(known) or "(none)", content=episode.content)}],
            max_tokens=2000, temperature=0.0)
        try:
            data = _parse_json(reply.get("content") or "")
        except ValueError:
            raise Permanent("the extraction was not JSON")
        entities = [e for e in data.get("entities") or [] if isinstance(e, dict) and str(e.get("name") or "").strip()]
        facts = [f for f in data.get("facts") or [] if isinstance(f, dict) and f.get("fact")]
        embedder = await self._embedder()
        entity_vectors = await embedder.embed([f"{e['name']}: {e.get('summary') or ''}" for e in entities]) \
            if embedder and entities else [None] * len(entities)
        fact_vectors = await embedder.embed([str(f["fact"]) for f in facts]) if embedder and facts else [None] * len(facts)
        dim = int(embedder.dimension) if embedder else 0
        async with async_session_scope() as session:
            await session.execute(sa.text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                                  {"k": f"memory:{episode.owner_subject}:{episode.group_id}"})
            ids = {}
            for entity, vector in zip(entities, entity_vectors):
                ids[_key(entity["name"])] = await self._resolve(session, chat, episode, entity, vector, dim)
                for alias in entity.get("aliases") or []:
                    ids.setdefault(_key(alias), ids[_key(entity["name"])])
            for fact, vector in zip(facts, fact_vectors):
                source, target = ids.get(_key(fact.get("source"))), ids.get(_key(fact.get("target")))
                if not source or not target:
                    continue
                await self._add_fact(session, chat, episode, source, target, fact, vector, dim)
            row = await session.get(MemoryEpisode, episode.id)
            if row is not None:
                row.status, row.processed_at, row.error = "processed", utcnow(), ""

    async def _resolve(self, session, chat, episode, entity: dict, vector, dim: int) -> str:
        """An existing entity of this owner and group (same name or alias; else similar embedding
        confirmed by the LLM), or a new one."""
        name = str(entity["name"]).strip()[:255]
        keys = {_key(name)} | {_key(a) for a in entity.get("aliases") or []}
        rows = (await session.execute(sa.select(MemoryEntity).where(
            MemoryEntity.owner_subject == episode.owner_subject, MemoryEntity.group_id == episode.group_id,
            MemoryEntity.deleted_at.is_(None)))).scalars().all()
        for row in rows:
            if keys & ({_key(row.name)} | {_key(a) for a in row.aliases or []}):
                row.aliases = sorted({*row.aliases, *(a for a in [name, *(entity.get("aliases") or [])]
                                                    if _key(a) != _key(row.name))})[:50]
                return row.id
        if vector is not None and dim:
            q = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
            near = (await session.execute(sa.text(
                f"SELECT id, name, summary, 1 - (embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim}))) AS s "
                f"FROM kb_memory_entities WHERE owner_subject = :o AND group_id = :g AND deleted_at IS NULL "
                f"AND dimension = {dim} ORDER BY embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim})) LIMIT 3"),
                {"q": q, "o": episode.owner_subject, "g": episode.group_id})).all()
            for cid, cname, csummary, score in near:
                if score is None or score < SAME_ENTITY:
                    continue
                try:
                    reply = await chat.complete([
                        {"role": "system", "content": render("memory", "resolve_system")},
                        {"role": "user", "content": render("memory", "resolve_user", a=f"{name}: {entity.get('summary', '')}",
                                                           b=f"{cname}: {csummary}")}], max_tokens=50, temperature=0.0)
                    if _parse_json(reply.get("content") or "").get("same") is True:
                        row = await session.get(MemoryEntity, cid)
                        row.aliases = sorted({*row.aliases, name})[:50]
                        return cid
                except (ValueError, ModelError):
                    continue
        row = MemoryEntity(id=new_id(), owner_subject=episode.owner_subject, owner_org=episode.owner_org,
                           group_id=episode.group_id, name=name, entity_type=str(entity.get("type") or "thing")[:64],
                           summary=str(entity.get("summary") or "")[:2000],
                           aliases=sorted({str(a)[:255] for a in entity.get("aliases") or [] if a})[:50],
                           dimension=dim if vector is not None else 0, embedding=vector)
        session.add(row)
        await session.flush()
        return row.id

    async def _add_fact(self, session, chat, episode, source: str, target: str, fact: dict, vector, dim: int) -> None:
        text = str(fact["fact"]).strip()[:2000]
        relation = re.sub(r"[^A-Z0-9_]+", "_", str(fact.get("relation") or "RELATES_TO").upper()).strip("_")[:128] \
            or "RELATES_TO"
        valid_at = _safe_time(fact.get("valid_at")) or episode.reference_time
        invalid_at = _safe_time(fact.get("invalid_at"))
        result = await session.execute(sa.select(MemoryFact).where(
            MemoryFact.owner_subject == episode.owner_subject, MemoryFact.group_id == episode.group_id,
            MemoryFact.expired_at.is_(None), MemoryFact.invalid_at.is_(None),
            sa.or_(sa.and_(MemoryFact.source_entity_id == source, MemoryFact.target_entity_id == target),
                   sa.and_(MemoryFact.source_entity_id == source, MemoryFact.relation == relation))))
        current = result.scalars().all()
        for old in current:                        # the same fact again: one more supporting episode
            if _key(old.fact) == _key(text) and old.relation == relation and old.target_entity_id == target:
                if episode.id not in old.episode_ids:
                    old.episode_ids = [*old.episode_ids, episode.id]
                return
        new = MemoryFact(id=new_id(), owner_subject=episode.owner_subject, owner_org=episode.owner_org,
                         group_id=episode.group_id, source_entity_id=source, target_entity_id=target, relation=relation,
                         fact=text, valid_at=valid_at, invalid_at=invalid_at, episode_ids=[episode.id],
                         dimension=dim if vector is not None else 0, embedding=vector, created_at=utcnow())
        session.add(new)
        await session.flush()
        if not current:
            return
        try:
            reply = await chat.complete([
                {"role": "system", "content": render("memory", "contradiction_system")},
                {"role": "user", "content": render("memory", "contradiction_user", fact=text,
                                                   existing="\n".join(f"{f.id}: {f.fact}" for f in current))}],
                max_tokens=300, temperature=0.0)
            contradicted = set(_parse_json(reply.get("content") or "").get("contradicted") or [])
        except (ValueError, ModelError) as exc:
            LOG.warning("contradiction check failed: %s", exc)
            return
        for old in current:
            if old.id in contradicted:
                # event time: the old fact stops being true when the new one starts
                old.invalid_at = max(valid_at, old.valid_at) if old.valid_at else valid_at
                old.invalidated_by = new.id

    # == reading ==============================================================================================
    @staticmethod
    async def _names(session, facts) -> dict:
        ids = {i for f in facts for i in (f.source_entity_id, f.target_entity_id)}
        if not ids:
            return {}
        return dict((await session.execute(sa.select(MemoryEntity.id, MemoryEntity.name).where(
            MemoryEntity.id.in_(ids)))).all())

    @staticmethod
    def _time_filter(as_of=None, known_at=None, include_invalid: bool = False):
        clauses = []
        if known_at is not None:
            clauses += [MemoryFact.created_at <= known_at,
                        sa.or_(MemoryFact.expired_at.is_(None), MemoryFact.expired_at > known_at)]
        else:
            clauses.append(MemoryFact.expired_at.is_(None))
        if as_of is not None:
            clauses += [sa.or_(MemoryFact.valid_at.is_(None), MemoryFact.valid_at <= as_of),
                        sa.or_(MemoryFact.invalid_at.is_(None), MemoryFact.invalid_at > as_of)]
        elif not include_invalid:
            clauses.append(sa.or_(MemoryFact.invalid_at.is_(None), MemoryFact.invalid_at > utcnow()))
        return sa.and_(*clauses)

    async def _entity_id(self, session, ctx, entity, group) -> str | None:
        if not entity:
            return None
        row = await session.get(MemoryEntity, entity)
        if row is not None and row.owner_subject == ctx.subject:
            return row.id
        query = sa.select(MemoryEntity).where(self._mine(MemoryEntity, ctx, group), MemoryEntity.deleted_at.is_(None))
        for row in (await session.execute(query)).scalars():
            if _key(entity) in {_key(row.name), *(_key(a) for a in row.aliases or [])}:
                return row.id
        raise NotFound(f"entity {entity} not found")

    async def facts(self, ctx, group_id=None, q=None, entity=None, as_of=None, known_at=None,
                    include_invalid: bool = False, limit: int = 100) -> list[dict]:
        self._check()
        as_of, known_at = _time(as_of, "as_of"), _time(known_at, "known_at")
        async with async_session_scope(commit=False) as session:
            query = sa.select(MemoryFact).where(self._mine(MemoryFact, ctx, group_id),
                                                self._time_filter(as_of, known_at, include_invalid))
            if q:
                query = query.where(MemoryFact.fact.ilike(f"%{q}%"))
            entity_id = await self._entity_id(session, ctx, entity, group_id)
            if entity_id:
                query = query.where(sa.or_(MemoryFact.source_entity_id == entity_id,
                                           MemoryFact.target_entity_id == entity_id))
            rows = (await session.execute(query.order_by(MemoryFact.valid_at.desc().nullslast(),
                                                         MemoryFact.created_at.desc()).limit(max(1, min(limit, 500))))).scalars().all()
            names = await self._names(session, rows)
            return [fact_dict(f, names) for f in rows]

    async def entities(self, ctx, group_id=None, q=None, page=1, page_size=50) -> tuple[list[dict], int]:
        self._check()
        async with async_session_scope(commit=False) as session:
            query = sa.select(MemoryEntity).where(self._mine(MemoryEntity, ctx, group_id), MemoryEntity.deleted_at.is_(None))
            if q:
                query = query.where(MemoryEntity.name.ilike(f"%{q}%"))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(MemoryEntity.name).offset((page - 1) * page_size)
                                          .limit(page_size))).scalars().all()
            return [entity_dict(e) for e in rows], total

    async def entity(self, ctx, entity_id) -> dict:
        """The entity with its full fact history (current, invalidated and expired)."""
        self._check()
        async with async_session_scope(commit=False) as session:
            row = await session.get(MemoryEntity, entity_id)
            if row is None or row.owner_subject != ctx.subject or row.deleted_at is not None:
                raise NotFound("entity not found")
            facts = (await session.execute(sa.select(MemoryFact).where(
                MemoryFact.owner_subject == ctx.subject,
                sa.or_(MemoryFact.source_entity_id == row.id, MemoryFact.target_entity_id == row.id))
                .order_by(MemoryFact.valid_at.nullsfirst(), MemoryFact.created_at))).scalars().all()
            names = await self._names(session, facts)
            return {**entity_dict(row), "facts": [fact_dict(f, names) for f in facts]}

    async def search(self, ctx, body: dict) -> list[dict]:
        """BM25 + vector + the focal entity's neighbourhood (graph_hops), fused with RRF, then MMR."""
        self._check()
        query = str(body.get("query") or "").strip()
        if not query:
            raise Invalid("query is required")
        group = body.get("group_id")
        limit = max(1, min(int(body.get("limit") or 10), 50))
        as_of, known_at = _time(body.get("as_of"), "as_of"), _time(body.get("known_at"), "known_at")
        time_filter = self._time_filter(as_of, known_at, bool(body.get("include_invalid")))
        mine = self._mine(MemoryFact, ctx, group)
        async with async_session_scope(commit=False) as session:
            keyword = (await session.execute(sa.text(
                "SELECT id, paradedb.score(seq) AS s FROM kb_memory_facts "
                "WHERE seq @@@ paradedb.match('fact', :q) AND owner_subject = :o ORDER BY s DESC LIMIT 50"),
                {"q": query, "o": ctx.subject})).all() if re.search(r"\w", query) else []
            vector_hits = []
            embedder = await self._embedder()
            if embedder is not None:
                vector = (await embedder.embed([query]))[0]
                dim = int(embedder.dimension)
                q = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
                vector_hits = (await session.execute(sa.text(
                    f"SELECT id, 1 - (embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim}))) FROM kb_memory_facts "
                    f"WHERE owner_subject = :o AND dimension = {dim} "
                    f"ORDER BY embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim})) LIMIT 50"),
                    {"q": q, "o": ctx.subject})).all()
            graph = []
            focal = await self._entity_id(session, ctx, body.get("focal_entity"), group) if body.get("focal_entity") else None
            if focal:
                hops = int(setting("graph_hops", "2"))
                frontier, seen = {focal}, {focal}
                for depth in range(hops):
                    rows = (await session.execute(sa.select(MemoryFact.id, MemoryFact.source_entity_id,
                                                            MemoryFact.target_entity_id).where(
                        mine, time_filter, sa.or_(MemoryFact.source_entity_id.in_(frontier),
                                                  MemoryFact.target_entity_id.in_(frontier))))).all()
                    nxt = set()
                    for fid, s, t in rows:
                        graph.append((fid, 1.0 / (depth + 1)))
                        nxt |= {s, t}
                    frontier = nxt - seen
                    seen |= nxt
                    if not frontier:
                        break
            candidates = {fid for fid, _ in [*keyword, *vector_hits, *graph]}
            allowed = set((await session.execute(sa.select(MemoryFact.id).where(
                mine, time_filter, MemoryFact.id.in_(candidates or {"-"})))).scalars())
            best = max((float(s) for _, s in keyword), default=1.0) or 1.0
            kw = [(fid, float(s) / best) for fid, s in keyword if fid in allowed]
            vec = [(fid, float(s)) for fid, s in vector_hits if fid in allowed and s is not None]
            gr = sorted({fid: s for fid, s in graph if fid in allowed}.items(), key=lambda x: -x[1])
            fused = rrf(vec, kw, 60, 0.6, 0.4)
            if gr:
                scores = {f.chunk_id: f.score for f in fused}
                for rank, (fid, _) in enumerate(gr, start=1):
                    scores[fid] = scores.get(fid, 0.0) + 0.5 / (60 + rank) * 61
                order = sorted(scores.items(), key=lambda x: -x[1])
            else:
                order = [(f.chunk_id, f.score) for f in fused]
            ids = [fid for fid, _ in order[:limit * 3]]
            rows = {f.id: f for f in (await session.execute(sa.select(MemoryFact).where(
                MemoryFact.id.in_(ids or ["-"])))).scalars()}
            names = await self._names(session, rows.values())
        items = [{**fact_dict(rows[fid], names), "score": round(score, 4), "content": rows[fid].fact}
                 for fid, score in order[:limit * 3] if fid in rows]
        chosen = mmr(items, limit)
        for item in chosen:
            item.pop("content", None)
        return chosen

    # == documents of a KB with memory_config.enabled ========================================================
    async def feed_document(self, session, item, kb, text: str) -> None:
        cfg = kb.memory_config or {}
        if not features.enabled("memory") or not cfg.get("enabled") or not text.strip():
            return
        await self.store_episode(session, kb.owner_subject, kb.owner_org, text[:MAX_CONTENT],
                                 group_id=self._group(cfg.get("group_id")), source="document",
                                 name=item.title[:255], source_ref=f"{item.id}@{item.version}",
                                 reference_time=item.valid_from or utcnow())


def _safe_time(value):
    try:
        return _time(value, "time")
    except Invalid:
        return None


def ensure_vector_index_for_memory(dimension: int) -> None:
    """HNSW indexes on memory entities and facts for this dimension (as for kb_embeddings)."""
    import zlib

    engine = get_engine()
    dimension = int(dimension)
    for table in ("kb_memory_entities", "kb_memory_facts"):
        name = f"{table}_hnsw_{dimension}"
        with engine.connect() as conn:
            if conn.execute(sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": name}).first():
                continue
        with engine.begin() as conn:
            conn.execute(sa.text("SELECT pg_advisory_xact_lock(:k)"), {"k": zlib.crc32(name.encode())})
            conn.execute(sa.text(
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw ((embedding::halfvec({dimension})) "
                f"halfvec_cosine_ops) WHERE dimension = {dimension}"))
    ensure_vector_index(engine, dimension)
