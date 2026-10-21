"""
Post-processing after the swap (docs/ingestion.md, Post-processing and enrichment; Multimodal;
Conflicts).

    plan()                   counts and enqueues the subtasks that apply; the item becomes
                             `finalizing` with `pending_subtasks_count = n`
    summary:generation       summary, gist, topics → description / profile, a `summary` chunk
                             indexed with source_type 2 (when the KB has `summary_model_id`)
    question:generation      questions per text chunk, indexed with source_type 1
                             (`question_generation_config.enabled`)
    image:multimodal         VLM OCR and caption per image → `image_ocr` / `image_caption` chunks,
                             source_type 3 (`vlm_config.enabled` and a VLM model)
    knowledge:conflicts      near-duplicate / superseding older items (`conflict_detection`)

Each subtask decrements the counter when it ends, successfully or after its last retry; the last
one sets `completed`. A subtask never fails the document. A subtask that belongs to an older
version of the content (the item was re-processed meanwhile) does nothing.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import uuid
from typing import Any

import anyio
import sqlalchemy as sa

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope, get_engine
from kb_core.orm.models.indexes import ensure_vector_index
from kb_core.orm.models.kb import (SOURCE_IMAGE, SOURCE_QUESTION, SOURCE_SUMMARY, Chunk, Embedding, Knowledge,
                                   KnowledgeBase, KnowledgeConflict, new_id, utcnow)
from kb_runtime.models.clients import ModelError
from kb_runtime.prompts import render
from kb_runtime.tasks import queue
from kb_runtime.tasks.runner import HandlerSpec, TaskContext

LOG = get_logger(__name__)

SUMMARY_CHARS = 8000
QUESTION_BATCH = 20
MAX_CONFLICT_CHUNKS = 300
DEFAULT_MAX_IMAGES = 50


def _json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in the reply")
    return json.loads(match.group(0))


def _default(option: str) -> str | None:
    return config_get("models", option).strip() or None if config_has_option("models", option) else None


class EnrichmentService:
    wiki = None                          # WikiService (phase 3): the wiki ingest is one more subtask
    memory = None                        # MemoryService (phase 3): documents of memory-enabled KBs as episodes

    def __init__(self, models, storage):
        self.models = models
        self.storage = storage

    # -- planning ------------------------------------------------------------------------------------
    async def chat_model_id(self, kb: KnowledgeBase) -> str | None:
        return kb.summary_model_id or await self.models.default_id("KnowledgeQA")

    async def vlm_model_id(self, kb: KnowledgeBase) -> str | None:
        cfg = kb.vlm_config or {}
        if not cfg.get("enabled"):
            return None
        return cfg.get("model_id") or await self.models.default_id("VLLM")

    async def plan(self, session, item: Knowledge, kb: KnowledgeBase, chunks: list[Chunk],
                   image_names: list[str]) -> int:
        jobs: list[tuple[str, dict, str]] = []
        base = {"knowledge_id": item.id, "content_hash": item.content_hash}
        if kb.summary_model_id:
            jobs.append(("summary:generation", base, f"summary:{item.id}"))
        qcfg = kb.question_generation_config or {}
        if qcfg.get("enabled") and await self.chat_model_id(kb) and any(c.chunk_type == "text" for c in chunks):
            jobs.append(("question:generation", base, f"questions:{item.id}"))
        if image_names and await self.vlm_model_id(kb):
            limit = int((kb.vlm_config or {}).get("max_images") or DEFAULT_MAX_IMAGES)
            for name in image_names[:limit]:
                jobs.append(("image:multimodal", {**base, "image": name}, f"image:{item.id}:{name}"))
        if (kb.freshness_config or {}).get("conflict_detection"):
            jobs.append(("knowledge:conflicts", base, f"conflicts:{item.id}"))
        for task_type, payload, key in jobs:
            await queue.enqueue(session, task_type, payload, owner_subject=item.owner_subject,
                                owner_org=item.owner_org, knowledge_base_id=kb.id, queue="postprocess",
                                unique_key=key, max_retry=2)
        wiki = 1 if self.wiki is not None and await self.wiki.queue_ingest(session, item, kb) else 0
        if self.memory is not None and (kb.memory_config or {}).get("enabled"):
            await self.memory.feed_document(session, item, kb, "\n\n".join(
                c.content for c in chunks if c.chunk_type == "text"))
        return len(jobs) + wiki

    def handlers(self) -> dict[str, HandlerSpec]:
        def spec(body, timeout):
            async def run(ctx: TaskContext):
                if await body(ctx) is not False:
                    await self.finish(ctx.payload["knowledge_id"], ctx.payload.get("content_hash"))

            async def failed(ctx: TaskContext, error: str):
                LOG.warning("%s of %s failed: %s", ctx.task["type"], ctx.payload["knowledge_id"], error)
                await self.finish(ctx.payload["knowledge_id"], ctx.payload.get("content_hash"))
            return HandlerSpec(run=run, timeout=timeout, on_final_failure=failed)

        return {"summary:generation": spec(self._summary, 600), "question:generation": spec(self._questions, 1200),
                "image:multimodal": spec(self._image, 600), "knowledge:conflicts": spec(self._conflicts, 600)}

    async def finish(self, knowledge_id: str, content_hash: str | None) -> None:
        from kb_runtime.services.knowledge import event

        async with async_session_scope() as session:
            item = await session.get(Knowledge, knowledge_id, with_for_update=True)
            if item is None or item.deleted_at is not None or (content_hash and item.content_hash != content_hash):
                return
            item.pending_subtasks_count = max(0, item.pending_subtasks_count - 1)
            if item.pending_subtasks_count == 0 and item.parse_status == "finalizing":
                item.parse_status = "completed"
                await event(session, item, {"type": "knowledge_status", "parse_status": "completed"})

    # -- shared --------------------------------------------------------------------------------------
    async def _current(self, ctx: TaskContext):
        """(item, kb) if the item still has the content this subtask was planned for, else None."""
        async with async_session_scope(commit=False) as session:
            item = await session.get(Knowledge, ctx.payload["knowledge_id"])
            if item is None or item.deleted_at is not None or item.content_hash != ctx.payload.get("content_hash"):
                return None
            kb = await session.get(KnowledgeBase, item.knowledge_base_id)
            session.expunge(item)
            session.expunge(kb)
            return item, kb

    async def _vectors(self, kb: KnowledgeBase, texts: list[str]) -> tuple[list | None, int]:
        if not texts or not kb.embedding_model_id or not kb.indexing_strategy.get("vector_enabled", True):
            return None, 0
        embedder = await self.models.embedder(kb.embedding_model_id)
        await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), embedder.dimension)
        return await embedder.embed(texts), embedder.dimension

    @staticmethod
    def _index_row(item: Knowledge, source_id: str, source_type: int, chunk_id: str, content: str,
                   vector, dimension: int) -> Embedding:
        from kb_runtime.services.knowledge import _valid_until

        return Embedding(source_id=source_id, source_type=source_type, chunk_id=chunk_id, knowledge_id=item.id,
                         knowledge_base_id=item.knowledge_base_id, owner_subject=item.owner_subject,
                         visibility=item.visibility, groups=list(item.groups or []), content=content,
                         dimension=dimension if vector is not None else 0, embedding=vector,
                         is_enabled=item.enable_status == "enabled", valid_from=item.valid_from,
                         valid_until=_valid_until(item), is_deprecated=item.deprecated_at is not None)

    async def _still_current(self, session, item: Knowledge) -> bool:
        row = await session.get(Knowledge, item.id, with_for_update=True)
        return row is not None and row.deleted_at is None and row.content_hash == item.content_hash

    # -- summary -------------------------------------------------------------------------------------
    async def _summary(self, ctx: TaskContext):
        found = await self._current(ctx)
        if found is None:
            return False
        item, kb = found
        text = (await anyio.to_thread.run_sync(self.storage.read_text, item.parsed_path))[:SUMMARY_CHARS]
        chat = await self.models.chat(kb.summary_model_id)
        reply = await chat.complete([{"role": "system", "content": render("enrichment", "summary_system")},
                                     {"role": "user", "content": render("enrichment", "summary_user",
                                                                        title=item.title, text=text)}],
                                    max_tokens=1024, temperature=0.3)
        data = _json_object(reply["content"])
        summary, gist = str(data.get("summary") or "").strip(), str(data.get("gist") or "").strip()
        content = "# Summary\n" + (gist + "\n\n" if gist else "") + summary
        vectors, dim = await self._vectors(kb, [item.title + "\n" + content])
        async with async_session_scope() as session:
            if not await self._still_current(session, item):
                return False
            row = await session.get(Knowledge, item.id)
            row.description = summary[:4000]
            row.profile = {"gist": gist, "topics": list(data.get("topics") or [])[:6],
                           "doc_type": data.get("doc_type") or "", "typical_question": data.get("typical_question") or ""}
            row.summary_status = "completed"
            last = (await session.execute(sa.select(sa.func.max(Chunk.chunk_index)).where(
                Chunk.knowledge_id == item.id))).scalar() or 0
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == item.id,
                                                             Embedding.source_type == SOURCE_SUMMARY))
            await session.execute(sa.delete(Chunk).where(Chunk.knowledge_id == item.id, Chunk.chunk_type == "summary"))
            chunk = Chunk(knowledge_base_id=kb.id, knowledge_id=item.id, content=content, chunk_index=last + 1,
                          chunk_type="summary", status=2)
            session.add(chunk)
            await session.flush()
            session.add(self._index_row(row, f"{item.id}:summary", SOURCE_SUMMARY, chunk.id,
                                        item.title + "\n" + content, vectors[0] if vectors else None, dim))

    # -- questions -----------------------------------------------------------------------------------
    async def _questions(self, ctx: TaskContext):
        found = await self._current(ctx)
        if found is None:
            return False
        item, kb = found
        cfg = kb.question_generation_config or {}
        count = max(1, min(int(cfg.get("question_count") or 3), 10))
        chat = await self.models.chat(await self.chat_model_id(kb))
        async with async_session_scope(commit=False) as session:
            chunks = list((await session.execute(sa.select(Chunk).where(
                Chunk.knowledge_id == item.id, Chunk.chunk_type == "text").order_by(Chunk.chunk_index))).scalars())
            for c in chunks:
                session.expunge(c)
        generated: dict[str, list[dict]] = {}
        for start in range(0, len(chunks), QUESTION_BATCH):
            for index in range(start, min(start + QUESTION_BATCH, len(chunks))):
                chunk = chunks[index]
                prompt = render("enrichment", "questions_user", title=item.title, count=count,
                                instructions=cfg.get("custom_instructions") or "",
                                before=chunks[index - 1].content[-500:] if index > 0 else "",
                                after=chunks[index + 1].content[:500] if index + 1 < len(chunks) else "",
                                passage=chunk.content)
                reply = await chat.complete([{"role": "system", "content": render("enrichment", "questions_system")},
                                             {"role": "user", "content": prompt}], max_tokens=512, temperature=0.7)
                lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in reply["content"].splitlines()]
                questions = [q for q in lines if len(q) > 3][:count]
                generated[chunk.id] = [{"id": uuid.uuid4().hex[:8], "question": q,
                                        "content_revision": chunk.content_revision} for q in questions]
        texts, keys = [], []
        for chunk_id, questions in generated.items():
            for q in questions:
                texts.append(item.title + "\n" + q["question"])
                keys.append((chunk_id, q["id"]))
        vectors, dim = await self._vectors(kb, texts)
        async with async_session_scope() as session:
            if not await self._still_current(session, item):
                return False
            row = await session.get(Knowledge, item.id)
            await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == item.id,
                                                             Embedding.source_type == SOURCE_QUESTION))
            for chunk_id, questions in generated.items():
                chunk = await session.get(Chunk, chunk_id)
                if chunk is not None:
                    chunk.meta = {**(chunk.meta or {}), "generated_questions": questions}
            for n, (chunk_id, qid) in enumerate(keys):
                session.add(self._index_row(row, f"{chunk_id}-{qid}", SOURCE_QUESTION, chunk_id, texts[n],
                                            vectors[n] if vectors else None, dim))

    # -- images (VLM) --------------------------------------------------------------------------------
    async def _image(self, ctx: TaskContext):
        found = await self._current(ctx)
        if found is None:
            return False
        item, kb = found
        name = ctx.payload["image"]
        prefix = item.parsed_path.rsplit("/", 1)[0]
        data = await anyio.to_thread.run_sync(self.storage.read, f"{prefix}/images/{name}")
        mime = mimetypes.guess_type(name)[0] or "image/png"
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        vlm = await self.models.chat(await self.vlm_model_id(kb))
        scanned = name.startswith("page-")
        cfg = kb.vlm_config or {}

        async def ask(system: str) -> str:
            reply = await vlm.complete([{"role": "system", "content": system},
                                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
                                       max_tokens=1500, temperature=0.1)
            return reply["content"].strip()

        ocr = await ask(render("enrichment", "ocr_scanned_system" if scanned else "ocr_system"))
        if ocr.upper().startswith("NO_TEXT") or not re.search(r"\w", re.sub(r"[|\-:\s]", "", ocr)):
            ocr = ""
        caption = ""
        if not (scanned and ocr):
            caption = await ask(render("enrichment", "caption_system", language=cfg.get("description_language") or "English",
                                       instructions=cfg.get("custom_instructions") or ""))
        texts = [t for t in (ocr, caption) if t]
        vectors, dim = await self._vectors(kb, [item.title + "\n" + t for t in texts])
        async with async_session_scope() as session:
            if not await self._still_current(session, item):
                return False
            row = await session.get(Knowledge, item.id)
            parent = (await session.execute(sa.select(Chunk).where(
                Chunk.knowledge_id == item.id, Chunk.chunk_type == "text",
                Chunk.content.contains(f"images/{name}")).order_by(Chunk.chunk_index).limit(1))).scalar_one_or_none()
            if parent is None:
                parent = (await session.execute(sa.select(Chunk).where(Chunk.knowledge_id == item.id).order_by(
                    Chunk.chunk_index).limit(1))).scalar_one_or_none()
            last = (await session.execute(sa.select(sa.func.max(Chunk.chunk_index)).where(
                Chunk.knowledge_id == item.id))).scalar() or 0
            n = 0
            for kind, text in (("image_ocr", ocr), ("image_caption", caption)):
                if not text:
                    continue
                chunk = Chunk(knowledge_base_id=kb.id, knowledge_id=item.id, content=text, chunk_index=last + 1 + n,
                              chunk_type=kind, parent_chunk_id=parent.id if parent else "", status=2,
                              image_info=[{"url": f"images/{name}"}])
                session.add(chunk)
                await session.flush()
                session.add(self._index_row(row, chunk.id, SOURCE_IMAGE, chunk.id, item.title + "\n" + text,
                                            vectors[n] if vectors else None, dim))
                n += 1
            if parent is not None:
                info = [i for i in (parent.image_info or []) if i.get("url") != f"images/{name}"]
                info.append({"url": f"images/{name}", "original_url": name, "caption": caption, "ocr_text": ocr})
                parent.image_info = info

    # -- conflicts -----------------------------------------------------------------------------------
    async def _conflicts(self, ctx: TaskContext):
        found = await self._current(ctx)
        if found is None:
            return False
        item, kb = found
        cfg = kb.freshness_config or {}
        threshold = float(cfg.get("conflict_similarity") or 0.92)
        params = {"kb": kb.id, "k": item.id, "limit": MAX_CONFLICT_CHUNKS}
        async with async_session_scope(commit=False) as session:
            dim = (await session.execute(sa.select(Embedding.dimension).where(
                Embedding.knowledge_id == item.id, Embedding.source_type == 0, Embedding.dimension > 0).limit(1)
            )).scalar()
            if dim:
                rows = (await session.execute(sa.text(f"""
                    SELECT e.chunk_id, n.chunk_id, n.knowledge_id, n.score FROM
                      (SELECT chunk_id, embedding FROM kb_embeddings
                       WHERE knowledge_id = :k AND source_type = 0 AND dimension = {dim} LIMIT :limit) e
                    CROSS JOIN LATERAL (
                       SELECT o.chunk_id, o.knowledge_id,
                              1 - (o.embedding::halfvec({dim}) <=> e.embedding::halfvec({dim})) AS score
                       FROM kb_embeddings o
                       WHERE o.knowledge_base_id = :kb AND o.knowledge_id <> :k AND o.source_type = 0
                         AND o.dimension = {dim}
                       ORDER BY o.embedding::halfvec({dim}) <=> e.embedding::halfvec({dim}) LIMIT 3) n"""),
                    params)).all()
            else:
                rows = (await session.execute(sa.text("""
                    SELECT e.id, n.id, n.knowledge_id, n.score FROM
                      (SELECT id, content FROM kb_chunks WHERE knowledge_id = :k AND chunk_type = 'text' LIMIT :limit) e
                    CROSS JOIN LATERAL (
                       SELECT o.id, o.knowledge_id, similarity(o.content, e.content) AS score FROM kb_chunks o
                       WHERE o.knowledge_base_id = :kb AND o.knowledge_id <> :k AND o.chunk_type = 'text'
                         AND o.content % e.content
                       ORDER BY similarity(o.content, e.content) DESC LIMIT 3) n"""), params)).all()
            pairs_by_item: dict[str, list[tuple[str, str, float]]] = {}
            for mine, theirs, other_item, score in rows:
                if score is not None and float(score) >= threshold:
                    pairs_by_item.setdefault(other_item, []).append((mine, theirs, float(score)))
            others = {}
            if pairs_by_item:
                for other in (await session.execute(sa.select(Knowledge).where(
                        Knowledge.id.in_(list(pairs_by_item)), Knowledge.deleted_at.is_(None)))).scalars():
                    total = (await session.execute(sa.select(sa.func.count()).select_from(Chunk).where(
                        Chunk.knowledge_id == other.id, Chunk.chunk_type == "text"))).scalar_one()
                    others[other.id] = (other, total)
        records = []
        for other_id, pairs in pairs_by_item.items():
            if other_id not in others:
                continue
            other, total = others[other_id]
            if other.created_at >= item.created_at:      # only older items are superseded
                continue
            matched = len({mine for mine, _, _ in pairs})
            coverage = len({theirs for _, theirs, _ in pairs}) / max(1, total)
            if matched >= 2 or coverage >= 0.5:
                records.append((other, pairs, matched, coverage))
        judged = {}
        if records and cfg.get("conflict_llm_judge") and await self.chat_model_id(kb):
            chat = await self.models.chat(await self.chat_model_id(kb))
            async with async_session_scope(commit=False) as session:
                for other, pairs, _, _ in records:
                    mine, theirs, _ = max(pairs, key=lambda p: p[2])
                    newer, older = await session.get(Chunk, mine), await session.get(Chunk, theirs)
                    try:
                        reply = await chat.complete(
                            [{"role": "system", "content": render("enrichment", "conflict_judge_system")},
                             {"role": "user", "content": render("enrichment", "conflict_judge_user",
                                                                older=older.content[:2000], newer=newer.content[:2000])}],
                            max_tokens=200, temperature=0.0)
                        judged[other.id] = _json_object(reply["content"])
                    except (ModelError, ValueError) as exc:
                        LOG.warning("conflict judge failed: %s", exc)
        from sqlalchemy.dialects.postgresql import insert

        async with async_session_scope() as session:
            for other, pairs, matched, coverage in records:
                verdict = judged.get(other.id, {})
                label = verdict.get("verdict") if verdict.get("verdict") in (
                    "duplicate", "updates", "contradicts", "unrelated") else ""
                values = dict(id=new_id(), knowledge_base_id=kb.id, knowledge_id=item.id, other_knowledge_id=other.id,
                              similarity=max(p[2] for p in pairs), matched_chunks=matched, other_coverage=coverage,
                              pairs=[{"chunk_id": a, "other_chunk_id": b, "similarity": round(s, 4)}
                                     for a, b, s in pairs[:20]],
                              verdict=label, explanation=str(verdict.get("explanation") or "")[:1000],
                              status="dismissed" if label == "unrelated" else "open",
                              created_at=utcnow(), updated_at=utcnow())
                stmt = insert(KnowledgeConflict).values(**values).on_conflict_do_update(
                    index_elements=["knowledge_id", "other_knowledge_id"],
                    set_={k: values[k] for k in ("similarity", "matched_chunks", "other_coverage", "pairs", "verdict",
                                                 "explanation", "status", "updated_at")})
                await session.execute(stmt)
