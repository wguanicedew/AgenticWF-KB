"""
FAQService: entries of FAQ knowledge bases (docs/faq.md).

An FAQ KB has one container `kb_knowledges` row (type `faq`); each entry is a `kb_chunks` row
(`chunk_type = faq`) whose metadata holds the questions and answers, indexed into `kb_embeddings`
like any chunk. Writes are synchronous: the batch is planned under the KB's row lock, embedded,
then written in the same transaction, and the report returned.
"""

from __future__ import annotations

import hashlib
import random

import anyio
import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.orm.base.session import async_session_scope, get_engine
from kb_core.orm.models.indexes import ensure_vector_index
from kb_core.orm.models.kb import Chunk, Embedding, Knowledge, KnowledgeBase, KnowledgeTag, new_id, utcnow
from kb_runtime import access, features
from kb_runtime.errors import Forbidden, Invalid, NotFound
from kb_runtime.faq import io as faq_io
from kb_runtime.faq.rules import (clean_entry, comparable, entry_content, faq_config, index_texts, matched_question,
                                  normalise)
from kb_runtime.retrieval.search import SearchOptions
from kb_runtime.services.knowledge_bases import KnowledgeBaseService
from kb_runtime.services.tags import UNTAGGED, resolve_tags

MAX_BATCH = 5000


def _fields(chunk: Chunk) -> dict:
    meta = chunk.meta or {}
    return {"standard_question": meta.get("standard_question", ""),
            "similar_questions": list(meta.get("similar_questions") or []),
            "negative_questions": list(meta.get("negative_questions") or []),
            "answers": list(meta.get("answers") or []), "answer_strategy": meta.get("answer_strategy") or "all",
            "is_enabled": bool(chunk.is_enabled), "is_recommended": bool(meta.get("is_recommended", True))}


def entry_dict(chunk: Chunk) -> dict:
    meta = chunk.meta or {}
    return {"id": chunk.seq_id, "chunk_id": chunk.id, "knowledge_base_id": chunk.knowledge_base_id,
            **_fields(chunk), "tag_id": chunk.tag_id or None, "version": int(meta.get("version", 1)),
            "source": meta.get("source", "api"),
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else None}


def _planned_dict(plan: dict) -> dict:
    return {"id": plan.get("seq"), "chunk_id": plan["chunk_id"], **plan["fields"], "tag_id": plan["tag_id"] or None,
            "version": plan["version"]}


class FAQService:
    def __init__(self, models, kbs: KnowledgeBaseService, search):
        self.models = models
        self.kbs = kbs
        self.search_service = search

    # -- lookups ----------------------------------------------------------------------------------
    async def _kb(self, session, ctx: AuthContext, kb_ref: str, write: bool = False,
                  for_update: bool = False) -> KnowledgeBase:
        if not features.enabled("faq"):
            raise NotFound("FAQ knowledge bases are switched off in this deployment")
        kb = await self.kbs.load(session, ctx, kb_ref, for_update=for_update)
        if kb.type != "faq":
            raise Invalid("not an FAQ knowledge base")
        if write and not access.can_contribute(ctx, kb):
            raise Forbidden("you may not change this FAQ")
        return kb

    @staticmethod
    async def _container(session, kb: KnowledgeBase, create: bool = False) -> Knowledge | None:
        item = (await session.execute(sa.select(Knowledge).where(
            Knowledge.knowledge_base_id == kb.id, Knowledge.type == "faq",
            Knowledge.deleted_at.is_(None)))).scalars().first()
        if item is None and create:
            item = Knowledge(knowledge_base_id=kb.id, type="faq", title=kb.name, owner_subject=kb.owner_subject,
                             owner_org=kb.owner_org, visibility="all", groups=[], parse_status="completed",
                             enable_status="enabled", processed_at=utcnow(), channel="api",
                             embedding_model_id=kb.embedding_model_id)
            session.add(item)
            await session.flush()
        return item

    @staticmethod
    async def _chunks(session, container: Knowledge | None, for_update: bool = False) -> list[Chunk]:
        if container is None:
            return []
        query = sa.select(Chunk).where(Chunk.knowledge_id == container.id, Chunk.chunk_type == "faq") \
            .order_by(Chunk.chunk_index)
        if for_update:
            query = query.with_for_update()
        return list((await session.execute(query)).scalars())

    @staticmethod
    def _find(chunks: list[Chunk], entry_id) -> Chunk:
        text = str(entry_id)
        for chunk in chunks:
            if chunk.id == text or (text.isdigit() and chunk.seq_id == int(text)):
                return chunk
        raise NotFound("FAQ entry not found")

    async def _tag_ids(self, session, kb: KnowledgeBase, refs: list[str]) -> dict[str, str]:
        """ref (id or seq_id) → tag id, for the refs that exist in this KB."""
        out = {}
        for ref in dict.fromkeys(r for r in refs if r):
            ids, _ = await resolve_tags(session, [kb.id], [str(ref)])
            if ids:
                out[str(ref)] = ids[0]
        return out

    # -- indexing --------------------------------------------------------------------------------
    async def _embed(self, kb: KnowledgeBase, texts: list[str]) -> tuple[list | None, int]:
        if not texts or not kb.embedding_model_id or not kb.indexing_strategy.get("vector_enabled", True):
            return None, 0
        embedder = await self.models.embedder(kb.embedding_model_id)
        await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), embedder.dimension)
        return await embedder.embed(texts), embedder.dimension

    @staticmethod
    def _index_rows(container: Knowledge, chunk_id: str, fields: dict, tag_id: str, cfg: dict) -> list[tuple]:
        return [(source_id, source_type, text, chunk_id, fields["is_enabled"], tag_id)
                for source_id, source_type, text in index_texts(chunk_id, fields, cfg)]

    @staticmethod
    def _embedding(container: Knowledge, row: tuple, vector, dimension: int) -> Embedding:
        source_id, source_type, text, chunk_id, enabled, tag_id = row
        return Embedding(source_id=source_id, source_type=source_type, chunk_id=chunk_id, knowledge_id=container.id,
                         knowledge_base_id=container.knowledge_base_id, owner_subject=container.owner_subject,
                         visibility=container.visibility, groups=list(container.groups or []), tag_id=tag_id or "",
                         content=text, dimension=dimension if vector is not None else 0, embedding=vector,
                         is_enabled=bool(enabled), valid_from=None, valid_until=None, is_deprecated=False)

    async def _write_index(self, session, kb, container, rows: list[tuple], replace_chunk_ids: list[str]) -> None:
        vectors, dimension = await self._embed(kb, [row[2] for row in rows])
        if replace_chunk_ids:
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == container.id,
                                                             Embedding.chunk_id.in_(replace_chunk_ids)))
        for i, row in enumerate(rows):
            session.add(self._embedding(container, row, vectors[i] if vectors else None, dimension))

    # -- batch upsert --------------------------------------------------------------------------------
    async def upsert(self, ctx: AuthContext, kb_ref: str, entries: list[dict], mode: str = "append",
                     dry_run: bool = False, source: str = "api", create_only: bool = False,
                     partial_id=None) -> dict:
        if mode not in ("append", "replace"):
            raise Invalid("mode must be append or replace")
        if not isinstance(entries, list):
            raise Invalid("entries must be a list")
        if len(entries) > MAX_BATCH:
            raise Invalid(f"at most {MAX_BATCH} entries per batch")
        report = {"created": 0, "updated": 0, "unchanged": 0, "deleted": 0, "failed": [], "warnings": [],
                  "entries": [], "dry_run": bool(dry_run)}
        async with async_session_scope(commit=not dry_run) as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            container = await self._container(session, kb)
            chunks = await self._chunks(session, container, for_update=True)
            cfg = faq_config(kb)
            tag_refs = [str(e.get("tag_id")) for e in entries if isinstance(e, dict) and e.get("tag_id")]
            known_tags = await self._tag_ids(session, kb, tag_refs)
            tag_names = dict((await session.execute(sa.select(KnowledgeTag.name, KnowledgeTag.id).where(
                KnowledgeTag.knowledge_base_id == kb.id))).all())

            existing = {c.id: {"chunk_id": c.id, "seq": c.seq_id, "fields": _fields(c), "tag_id": c.tag_id or "",
                               "version": int((c.meta or {}).get("version", 1)), "chunk": c} for c in chunks}
            by_std = {normalise(p["fields"]["standard_question"]): p for p in existing.values()}
            owner: dict[str, str] = {}               # question key → chunk id using it
            if mode == "append":
                for p in existing.values():
                    for q in [p["fields"]["standard_question"], *p["fields"]["similar_questions"]]:
                        owner[normalise(q)] = p["chunk_id"]
            plans: list[dict] = []
            touched: set[str] = set()
            new_tag_names: set[str] = set()
            for index, raw in enumerate(entries):
                try:
                    if not isinstance(raw, dict):
                        raise Invalid("an entry must be an object")
                    match = None
                    if partial_id is not None:
                        match = existing[self._find(chunks, partial_id).id]
                    elif raw.get("id") not in (None, ""):
                        match = existing[self._find(chunks, raw["id"]).id]
                    elif raw.get("standard_question"):
                        match = by_std.get(normalise(str(raw["standard_question"])))
                        if match is not None and create_only:
                            raise Invalid("an entry with this standard question exists already")
                    keep = match is not None and (mode == "append" or partial_id is not None)
                    fields, warnings = clean_entry(raw, match["fields"] if keep else None,
                                                   merge=keep and partial_id is None)
                    if "tag_id" in raw:
                        ref = raw["tag_id"]
                        if ref in (None, "", UNTAGGED):
                            tag_id = ""
                        elif str(ref) in known_tags:
                            tag_id = known_tags[str(ref)]
                        else:
                            raise Invalid("unknown tag for this knowledge base")
                    elif raw.get("tag_name"):
                        name = str(raw["tag_name"]).strip()[:128]
                        tag_id = tag_names.get(name) or f"name:{name}"
                        if name not in tag_names:
                            new_tag_names.add(name)
                    else:
                        tag_id = match["tag_id"] if keep else ""
                    chunk_id = match["chunk_id"] if match else new_id()
                    std_key = normalise(fields["standard_question"])
                    if owner.get(std_key, chunk_id) != chunk_id:
                        raise Invalid("the standard question is already used by another entry")
                    similar = []
                    for question in fields["similar_questions"]:
                        if owner.get(normalise(question), chunk_id) != chunk_id:
                            warnings.append(f"similar question {question!r} is used by another entry and was removed")
                        else:
                            similar.append(question)
                    fields["similar_questions"] = similar
                    if match is not None:
                        for q in [match["fields"]["standard_question"], *match["fields"]["similar_questions"]]:
                            if owner.get(normalise(q)) == chunk_id:
                                owner.pop(normalise(q))
                    for q in [fields["standard_question"], *similar]:
                        owner[normalise(q)] = chunk_id
                    if match is None:
                        outcome = "created"
                    elif comparable(fields, tag_id) == comparable(match["fields"], match["tag_id"]):
                        outcome = "unchanged"
                    else:
                        outcome = "updated"
                    plan = {"index": index, "outcome": outcome, "chunk_id": chunk_id, "fields": fields,
                            "tag_id": tag_id, "seq": match["seq"] if match else None,
                            "version": (match["version"] + (1 if outcome == "updated" else 0)) if match else 1,
                            "chunk": match["chunk"] if match else None}
                    if chunk_id in touched:          # the same entry twice in one batch: the later one wins
                        plans = [p for p in plans if p["chunk_id"] != chunk_id]
                    touched.add(chunk_id)
                    plans.append(plan)
                    if match is not None:
                        by_std.pop(normalise(match["fields"]["standard_question"]), None)
                    if match is None:
                        existing[chunk_id] = {**plan, "chunk": None}
                    else:
                        existing[chunk_id].update(fields=fields, tag_id=tag_id)
                    by_std[std_key] = existing[chunk_id]
                    if warnings:
                        report["warnings"].append({"index": index, "messages": warnings})
                except (Invalid, NotFound) as exc:
                    report["failed"].append({"index": index, "reason": str(exc)})
            deleted = [c for c in chunks if c.id not in touched] if mode == "replace" and partial_id is None else []
            for plan in plans:
                report[plan["outcome"]] += 1
            report["deleted"] = len(deleted)
            if dry_run:
                report["entries"] = [_planned_dict(p) for p in plans]
                return report

            # tags named on import are created
            for name in new_tag_names:
                tag = KnowledgeTag(knowledge_base_id=kb.id, name=name)
                session.add(tag)
                await session.flush()
                tag_names[name] = tag.id
            for plan in plans:
                if plan["tag_id"].startswith("name:"):
                    plan["tag_id"] = tag_names[plan["tag_id"][5:]]
            changed = [p for p in plans if p["outcome"] != "unchanged"]
            if changed or deleted:
                container = container or await self._container(session, kb, create=True)
                next_index = max((c.chunk_index for c in chunks), default=-1) + 1
                rows = []
                for plan in changed:
                    rows += self._index_rows(container, plan["chunk_id"], plan["fields"], plan["tag_id"], cfg)
                    meta = {**{k: v for k, v in plan["fields"].items() if k != "is_enabled"},
                            "version": plan["version"], "source": source}
                    content = entry_content(plan["fields"])
                    chunk = plan["chunk"]
                    if chunk is None:
                        chunk = Chunk(id=plan["chunk_id"], knowledge_base_id=kb.id, knowledge_id=container.id,
                                      chunk_index=next_index, chunk_type="faq")
                        next_index += 1
                        session.add(chunk)
                        plan["chunk"] = chunk
                    chunk.content, chunk.tag_id, chunk.is_enabled, chunk.meta = content, plan["tag_id"], \
                        plan["fields"]["is_enabled"], meta
                    chunk.content_hash = hashlib.sha256(content.encode()).hexdigest()
                    chunk.end_at = len(content)
                    chunk.updated_at = utcnow()
                await session.flush()
                await self._write_index(session, kb, container, rows,
                                        [p["chunk_id"] for p in changed] + [c.id for c in deleted])
                for chunk in deleted:
                    await session.delete(chunk)
                container.updated_at = utcnow()
                await session.flush()
            report["entries"] = [entry_dict(p["chunk"]) for p in plans]
            return report

    # -- single entries -------------------------------------------------------------------------------
    async def create_entry(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        report = await self.upsert(ctx, kb_ref, [{k: v for k, v in body.items() if k != "id"}], create_only=True)
        if report["failed"]:
            raise Invalid(report["failed"][0]["reason"])
        return {**report["entries"][0], "warnings": (report["warnings"] or [{"messages": []}])[0]["messages"]}

    async def update_entry(self, ctx: AuthContext, kb_ref: str, entry_id, body: dict) -> dict:
        report = await self.upsert(ctx, kb_ref, [{k: v for k, v in body.items() if k != "id"}], partial_id=entry_id)
        if report["failed"]:
            reason = report["failed"][0]["reason"]
            raise NotFound(reason) if "not found" in reason else Invalid(reason)
        return {**report["entries"][0], "warnings": (report["warnings"] or [{"messages": []}])[0]["messages"]}

    async def add_similar(self, ctx: AuthContext, kb_ref: str, entry_id, questions: list[str]) -> dict:
        entry = await self.get_entry(ctx, kb_ref, entry_id)
        return await self.update_entry(ctx, kb_ref, entry_id,
                                       {"similar_questions": entry["similar_questions"] + list(questions or [])})

    async def get_entry(self, ctx: AuthContext, kb_ref: str, entry_id) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            return entry_dict(self._find(await self._chunks(session, await self._container(session, kb)), entry_id))

    async def list_entries(self, ctx: AuthContext, kb_ref: str, page: int = 1, page_size: int = 20,
                           keyword: str | None = None, tag_id: str | None = None,
                           is_enabled: bool | None = None) -> tuple[list[dict], int]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            container = await self._container(session, kb)
            if container is None:
                return [], 0
            query = sa.select(Chunk).where(Chunk.knowledge_id == container.id, Chunk.chunk_type == "faq")
            if keyword:
                query = query.where(Chunk.content.ilike(f"%{keyword}%"))
            if tag_id == UNTAGGED:
                query = query.where(Chunk.tag_id == "")
            elif tag_id:
                ids, _ = await resolve_tags(session, [kb.id], [tag_id])
                query = query.where(Chunk.tag_id.in_(ids or ["-"]))
            if is_enabled is not None:
                query = query.where(Chunk.is_enabled.is_(is_enabled))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(Chunk.chunk_index).offset((page - 1) * page_size)
                                          .limit(page_size))).scalars().all()
            return [entry_dict(c) for c in rows], total

    async def _set(self, ctx: AuthContext, kb_ref: str, changes: dict[str, dict]) -> dict:
        """chunk id → {is_enabled?, is_recommended?, tag_id?} without re-embedding."""
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            chunks = {c.id: c for c in await self._chunks(session, await self._container(session, kb), True)}
            updated = 0
            for chunk_id, change in changes.items():
                chunk = chunks[chunk_id]
                values = {}
                if "is_enabled" in change:
                    chunk.is_enabled = bool(change["is_enabled"])
                    values["is_enabled"] = chunk.is_enabled
                if "is_recommended" in change:
                    chunk.meta = {**(chunk.meta or {}), "is_recommended": bool(change["is_recommended"])}
                if "tag_id" in change:
                    ref = change["tag_id"]
                    if ref in (None, "", UNTAGGED):
                        chunk.tag_id = ""
                    else:
                        ids, _ = await resolve_tags(session, [kb.id], [str(ref)])
                        if not ids:
                            raise Invalid("unknown tag for this knowledge base")
                        chunk.tag_id = ids[0]
                    values["tag_id"] = chunk.tag_id
                chunk.updated_at = utcnow()
                if values:
                    await session.execute(sa.update(Embedding).where(Embedding.chunk_id == chunk_id).values(**values))
                updated += 1
            return {"updated": updated}

    async def update_fields(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        allowed = ("is_enabled", "is_recommended", "tag_id")
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            chunks = await self._chunks(session, await self._container(session, kb))
            exclude = {self._find(chunks, e).id for e in body.get("exclude_ids") or [] if e not in (None, "")}
            changes: dict[str, dict] = {}
            for tag_ref, change in (body.get("by_tag") or {}).items():
                if tag_ref == UNTAGGED:
                    selected = [c for c in chunks if not c.tag_id]
                else:
                    ids, _ = await resolve_tags(session, [kb.id], [str(tag_ref)])
                    selected = [c for c in chunks if c.tag_id and c.tag_id in ids]
                for chunk in selected:
                    changes.setdefault(chunk.id, {}).update({k: v for k, v in change.items() if k in allowed})
            for entry_id, change in (body.get("by_id") or {}).items():
                chunk = self._find(chunks, entry_id)
                changes.setdefault(chunk.id, {}).update({k: v for k, v in change.items() if k in allowed})
        return await self._set(ctx, kb_ref, {k: v for k, v in changes.items() if k not in exclude})

    async def update_tags(self, ctx: AuthContext, kb_ref: str, updates: dict) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            chunks = await self._chunks(session, await self._container(session, kb))
            changes = {self._find(chunks, entry_id).id: {"tag_id": tag} for entry_id, tag in (updates or {}).items()}
        return await self._set(ctx, kb_ref, changes)

    async def delete_entries(self, ctx: AuthContext, kb_ref: str, ids: list) -> dict:
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, write=True, for_update=True)
            chunks = await self._chunks(session, await self._container(session, kb), True)
            selected = {self._find(chunks, entry_id).id for entry_id in ids or []}
            if selected:
                await session.execute(sa.delete(Embedding).where(Embedding.chunk_id.in_(selected)))
                await session.execute(sa.delete(Chunk).where(Chunk.id.in_(selected)))
            return {"deleted": len(selected)}

    async def reindex(self, ctx: AuthContext, kb_ref: str) -> dict:
        """Rebuild every entry's index rows (after faq_config changed)."""
        async with async_session_scope() as session:
            kb = await self._kb(session, ctx, kb_ref, for_update=True)
            container = await self._container(session, kb)
            chunks = await self._chunks(session, container, True)
            if not chunks:
                return {"reindexed": 0}
            cfg = faq_config(kb)
            rows = []
            for chunk in chunks:
                rows += self._index_rows(container, chunk.id, _fields(chunk), chunk.tag_id or "", cfg)
            await self._write_index(session, kb, container, rows, [c.id for c in chunks])
            return {"reindexed": len(chunks)}

    # -- search, import, export ------------------------------------------------------------------------------
    async def search(self, ctx: AuthContext, kb_ref: str, body: dict) -> list[dict]:
        query = (body.get("query_text") or body.get("query") or "").strip()
        if not query:
            raise Invalid("query_text is required")
        first = [str(t) for t in body.get("first_priority_tag_ids") or []]
        second = [str(t) for t in body.get("second_priority_tag_ids") or []]
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            first_ids = (await resolve_tags(session, [kb.id], first))[0] if first else []
            kb_id = kb.id
        threshold = body.get("vector_threshold")
        top_k = max(1, min(int(body.get("match_count") or 10), 50))
        options = SearchOptions(vector_threshold=float(threshold) if threshold not in (None, "") else None,
                                top_k=top_k * 2, tag_ids=first + second)
        if (first or second) and not options.tag_ids:
            return []
        hits = await self.search_service.search(ctx, query, [kb_id], options)
        out = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            if hit.get("chunk_type") != "faq" or (body.get("only_recommended") and not meta.get("is_recommended", True)):
                continue
            fields = {"standard_question": meta.get("standard_question", ""),
                      "similar_questions": meta.get("similar_questions") or []}
            out.append({"id": hit["seq"], "chunk_id": hit["id"], "score": hit["score"], "match_type": hit["match_type"],
                        "matched_question": matched_question(query, fields), "standard_question": fields["standard_question"],
                        "similar_questions": fields["similar_questions"], "answers": meta.get("answers") or [],
                        "answer_strategy": meta.get("answer_strategy", "all"), "is_recommended": meta.get("is_recommended", True),
                        **({"match": hit["match"]} if hit.get("match") else {})})
        if first_ids:
            async with async_session_scope(commit=False) as session:
                tags = dict((await session.execute(sa.select(Chunk.id, Chunk.tag_id).where(
                    Chunk.id.in_([o["chunk_id"] for o in out])))).all()) if out else {}
            for o in out:
                o["priority"] = 1 if tags.get(o["chunk_id"]) in first_ids else 2
            out.sort(key=lambda o: (o["priority"], -o["score"]))
        return out[:top_k]

    async def import_file(self, ctx: AuthContext, kb_ref: str, file_name: str, data: bytes, mode: str = "append",
                          dry_run: bool = False) -> dict:
        entries = faq_io.parse(file_name, data)
        return await self.upsert(ctx, kb_ref, entries, mode=mode, dry_run=dry_run, source="import")

    async def export(self, ctx: AuthContext, kb_ref: str) -> tuple[str, bytes]:
        async with async_session_scope(commit=False) as session:
            kb = await self._kb(session, ctx, kb_ref)
            chunks = await self._chunks(session, await self._container(session, kb))
            names = {tag_id: name for tag_id, name in (await session.execute(sa.select(
                KnowledgeTag.id, KnowledgeTag.name).where(KnowledgeTag.knowledge_base_id == kb.id))).all()}
            entries = [{**_fields(c), "tag_id": c.tag_id} for c in chunks]
            return f"{kb.slug}-faq.csv", faq_io.export(entries, names)


def pick_answer(entry: dict) -> list[str]:
    """What a client shows: every answer (`all`) or one at random (`random`)."""
    answers = entry.get("answers") or []
    return answers if entry.get("answer_strategy", "all") == "all" or not answers else [random.choice(answers)]
