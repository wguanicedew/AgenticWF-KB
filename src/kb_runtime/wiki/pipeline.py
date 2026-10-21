"""
The wiki pipeline (docs/llm-wiki.md, Generation pipeline; Graph extensions).

    wiki:ingest        claim ≤ ingest_batch_size pending ops of one KB (SKIP LOCKED) and run them:
                       retract → candidates (per document) → citations (per chunk window, with new
                       slugs and relations) → taxonomy (folders for new slugs) → dedup ("related ≠
                       same") → reduce (one grounded page write per slug) → summary pages
    wiki:finalize      no LLM: index page, dead links, cross-links, in/out links, folders, communities
    wiki:communities   one LLM-written page per changed community
    global_search      map over community (or summary) pages, reduce to one cited answer

Citations: the model cites passages as [cNNN]; stored pages carry `[src:<chunk seq>]` markers and
`chunk_refs` entries `<knowledge_id>|<chunk_id>|<seq>`, so a document's citations can be retracted
after its chunks are gone. Slugs appear in prompts as `ref-N` handles.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.ext import WikiCommunity, WikiPage, WikiPendingOp, WikiRelation
from kb_core.orm.models.kb import Chunk, Knowledge, KnowledgeBase, utcnow
from kb_runtime.errors import Invalid, NotFound
from kb_runtime.models.clients import ModelError
from kb_runtime.prompts import render
from kb_runtime.tasks.runner import HandlerSpec, TaskContext
from kb_runtime.wiki import text as wt
from kb_runtime.wiki.service import SRC, WikiService, feeds_wiki, passages_of

LOG = get_logger(__name__)

WINDOW = 15                        # chunks per citation call
DOC_CHARS = 12000                  # document text in the candidates prompt
PASSAGE_CHARS = 1500
MAX_KNOWN = 300
MAX_CONTINUATIONS = 3
MAX_FAILURES = 3
CLAIM_TIMEOUT = "30 minutes"
PASSAGE_TYPES = ("text", "image_ocr", "image_caption")
_LABEL = re.compile(r"\[(c\d{3,})\]")


def _json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("not an object")
    return value


def chunk_ref(knowledge_id: str, chunk_id: str, seq: int) -> str:
    return f"{knowledge_id}|{chunk_id}|{seq}"


def parse_ref(ref: str) -> tuple[str, str, int]:
    kid, cid, seq = (str(ref).split("|") + ["", "", "0"])[:3]
    return kid, cid, int(seq or 0)


class WikiPipeline:
    def __init__(self, wiki: WikiService, models, kbs):
        self.wiki = wiki
        self.models = models
        self.kbs = kbs

    def handlers(self) -> dict[str, HandlerSpec]:
        return {"wiki:ingest": HandlerSpec(self._ingest_task, 3600),
                "wiki:finalize": HandlerSpec(self._finalize_task, 900),
                "wiki:communities": HandlerSpec(self._communities_task, 1800)}

    # == LLM calls ==========================================================================================
    async def _chat(self, kb):
        model_id = await self.wiki.model_id(kb)
        if not model_id:
            raise ModelError("no synthesis model")
        return await self.models.chat(model_id)

    @staticmethod
    async def _ask(chat, sem, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> dict:
        async with sem:
            return await chat.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       max_tokens=max_tokens, temperature=temperature)

    async def _ask_json(self, chat, sem, system, user, max_tokens=1500) -> dict:
        reply = await self._ask(chat, sem, system, user, max_tokens, 0.0)
        try:
            return _json(reply.get("content") or "")
        except ValueError:
            return {}

    # == the ingest task ======================================================================================
    async def _claim(self, kb_id: str, limit: int) -> list[dict]:
        async with async_session_scope() as session:
            rows = (await session.execute(sa.text(
                "UPDATE kb_wiki_pending_ops SET claimed_at = now() WHERE id IN ("
                "  SELECT id FROM kb_wiki_pending_ops WHERE knowledge_base_id = :kb"
                f"  AND (claimed_at IS NULL OR claimed_at < now() - interval '{CLAIM_TIMEOUT}')"
                "  ORDER BY fail_count, id LIMIT :n FOR UPDATE SKIP LOCKED) "
                "RETURNING id, knowledge_id, op, payload, fail_count"), {"kb": kb_id, "n": limit})).mappings().all()
        return [dict(r) for r in rows]

    async def _done(self, ops: list[dict], finish: bool = True) -> None:
        if not ops:
            return
        async with async_session_scope() as session:
            await session.execute(sa.delete(WikiPendingOp).where(WikiPendingOp.id.in_([o["id"] for o in ops])))
        if finish and self.wiki.enrichment is not None:
            for op in ops:
                if op["op"] == "ingest" and not (op["payload"] or {}).get("rebuild"):
                    await self.wiki.enrichment.finish(op["knowledge_id"], (op["payload"] or {}).get("content_hash"))

    async def _failed(self, op: dict, error: str) -> None:
        LOG.warning("wiki %s of %s failed: %s", op["op"], op["knowledge_id"], error)
        if op["fail_count"] + 1 >= MAX_FAILURES:
            await self._done([op])
            return
        async with async_session_scope() as session:
            await session.execute(sa.update(WikiPendingOp).where(WikiPendingOp.id == op["id"]).values(
                fail_count=WikiPendingOp.fail_count + 1, claimed_at=None))

    async def _ingest_task(self, ctx: TaskContext) -> None:
        await self.run_batch(ctx.payload["knowledge_base_id"])

    async def run_batch(self, kb_id: str) -> dict:
        async with async_session_scope(commit=False) as session:
            kb = await session.get(KnowledgeBase, kb_id)
            if kb is not None:
                session.expunge(kb)
        if kb is None or kb.deleted_at is not None:
            async with async_session_scope() as session:
                await session.execute(sa.delete(WikiPendingOp).where(WikiPendingOp.knowledge_base_id == kb_id))
            return {"dropped": True}
        ops = await self._claim(kb_id, int(self.wiki.config(kb)["ingest_batch_size"]))
        if not ops:
            return {"ops": 0}
        if not self.wiki.enabled(kb) or not await self.wiki.model_id(kb):
            await self._done(ops)                          # released: documents never hang in finalizing
            return {"released": len(ops)}
        await self.wiki.ensure_index(kb.id)
        report = {"ops": len(ops), "pages": 0, "archived": 0}
        try:
            report.update(await self._batch(kb, ops))
        except Exception as exc:          # the whole batch failed: each op is retried on its own count
            LOG.exception("wiki batch for %s failed", kb_id)
            for op in ops:
                await self._failed(op, f"{type(exc).__name__}: {exc}")
            raise
        await self._done(ops)
        async with async_session_scope() as session:
            kb_row = await session.get(KnowledgeBase, kb.id)
            await self.wiki.trigger_finalize(session, kb_row)
            more = (await session.execute(sa.select(WikiPendingOp.id).where(
                WikiPendingOp.knowledge_base_id == kb.id).limit(1))).first()
            if more:
                await self.wiki.trigger(session, kb_row)
        return report

    async def _batch(self, kb, ops: list[dict]) -> dict:
        cfg = self.wiki.config(kb)
        chat = await self._chat(kb)
        sem = asyncio.Semaphore(max(1, int(cfg["ingest_max_inflight"])))
        language = "the language of the source documents" if cfg["language"] == "auto" else cfg["language"]

        # 1. retraction (explicit retract ops, and the old citations of re-ingested documents)
        docs, retract_ids = [], set()
        async with async_session_scope(commit=False) as session:
            for op in ops:
                item = await session.get(Knowledge, op["knowledge_id"])
                if op["op"] == "retract" or item is None or item.deleted_at is not None or not feeds_wiki(item, kb):
                    retract_ids.add(op["knowledge_id"])
                    continue
                if item.content_hash != (op["payload"] or {}).get("content_hash"):
                    continue                                     # superseded; a newer op follows
                chunks = (await session.execute(sa.select(Chunk).where(
                    Chunk.knowledge_id == item.id, Chunk.is_enabled, Chunk.chunk_type.in_(PASSAGE_TYPES))
                    .order_by(Chunk.chunk_index))).scalars().all()
                previous = (await session.execute(sa.select(WikiPage.slug).where(
                    WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                    WikiPage.source_ids.contains([item.id]), WikiPage.page_type != "summary"))).scalars().all()
                docs.append(SimpleNamespace(item=item, chunks=list(chunks), previous=list(previous)))
                retract_ids.add(item.id)
            for doc in docs:
                session.expunge(doc.item)
                for c in doc.chunks:
                    session.expunge(c)
        affected, archived = await self._retract(kb, retract_ids, keep_for_reingest={d.item.id for d in docs})

        # 2. candidates per document (map)
        granularity = cfg["extraction_granularity"] if cfg["extraction_granularity"] in (
            "focused", "standard", "exhaustive") else "standard"
        rule = render("wiki_ingest", f"granularity_{granularity}").strip()
        map_sem = asyncio.Semaphore(max(1, int(cfg["ingest_map_parallel"])))

        async def candidates(doc):
            async with map_sem:
                text = "\n\n".join(c.content for c in doc.chunks)[:DOC_CHARS]
                data = await self._ask_json(chat, sem, render("wiki_ingest", "candidates_system", granularity_rule=rule),
                                            render("wiki_ingest", "candidates_user", title=doc.item.title,
                                                   previous=", ".join(doc.previous) or "(none)", text=text))
                return data.get("candidates") or []

        candidate_lists = await asyncio.gather(*(candidates(d) for d in docs))
        async with async_session_scope(commit=False) as session:
            existing = {p.slug: p for p in (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None)))).scalars()}
            for page in existing.values():
                session.expunge(page)
        by_title = {wt.title_key(p.title): s for s, p in existing.items() if p.page_type in ("entity", "concept")}
        proposals: dict[str, dict] = {}          # slug -> {title, aliases}
        proposed_titles: dict[str, str] = {}     # title key -> slug proposed earlier in this batch

        def register(raw: dict) -> str | None:
            if not isinstance(raw, dict) or not raw.get("slug"):
                return None
            try:
                slug = wt.normalise_slug(raw["slug"], "concept")
            except ValueError:
                return None
            if slug.split("/")[0] not in ("entity", "concept"):
                return None
            title = str(raw.get("title") or slug.rpartition("/")[2].replace("-", " ")).strip()[:300]
            if slug not in existing:
                # cross-type reuse by normalised title: an existing page, or one proposed earlier in
                # this batch (two documents may type the same subject differently)
                key = wt.title_key(title)
                twin = by_title.get(key) or proposed_titles.get(key)
                if twin:
                    slug = twin
                else:
                    proposed_titles.setdefault(key, slug)
            entry = proposals.setdefault(slug, {"title": existing[slug].title if slug in existing else title,
                                                "aliases": set()})
            entry["aliases"] |= {str(a)[:200] for a in raw.get("aliases") or [] if a}
            return slug

        doc_slugs = []
        for doc, cands in zip(docs, candidate_lists):
            slugs = [s for s in (register(c) for c in cands) if s]
            slugs += [s for s in doc.previous if s in existing]
            doc_slugs.append(list(dict.fromkeys(slugs)))

        # 3. citations per document and chunk window
        citations: dict[str, list[tuple]] = {}    # slug -> [(item, chunk)]
        relations: list[dict] = []

        async def cite(doc, slugs):
            handles = wt.Handles(slugs)
            labels = {wt.chunk_label(i): c for i, c in enumerate(doc.chunks)}
            out_c, out_new, out_rel = {}, [], []
            for start in range(0, len(doc.chunks), WINDOW):
                window = doc.chunks[start:start + WINDOW]
                passages = "\n\n".join(f"[{wt.chunk_label(start + i)}] {c.content[:PASSAGE_CHARS]}"
                                       for i, c in enumerate(window))
                pages = "\n".join(f"{handles.ref(s)}: {proposals.get(s, {}).get('title') or existing[s].title}"
                                  for s in slugs if s in proposals or s in existing)
                data = await self._ask_json(chat, sem, render("wiki_ingest", "citations_system"),
                                            render("wiki_ingest", "citations_user", title=doc.item.title,
                                                   pages=pages or "(none yet)", passages=passages), 2000)
                for raw in data.get("new_slugs") or []:          # proposed here: citable from now on
                    slug = register(raw) if isinstance(raw, dict) else None
                    if slug and slug not in slugs:
                        slugs.append(slug)
                        handles.ref(slug)
                    out_new.append(raw)
                for ref, ids in (data.get("citations") or {}).items():
                    slug = handles.slug(ref)
                    if slug not in slugs:
                        try:
                            slug = wt.normalise_slug(slug)
                        except ValueError:
                            continue
                        if slug not in slugs:
                            continue
                    out_c.setdefault(slug, []).extend(labels[i] for i in ids or [] if i in labels)
                for rel in data.get("relations") or []:
                    if isinstance(rel, dict):
                        out_rel.append({**rel, "source": handles.slug(str(rel.get("source", ""))),
                                        "target": handles.slug(str(rel.get("target", ""))),
                                        "chunks": [labels[i] for i in rel.get("chunks") or [] if i in labels]})
            return out_c, out_new, out_rel

        results = await asyncio.gather(*(cite(d, s) for d, s in zip(docs, doc_slugs)))
        for doc, (cmap, new, rels) in zip(docs, results):
            for slug, chunk_list in cmap.items():
                citations.setdefault(slug, []).extend((doc.item, c) for c in chunk_list)
            for rel in rels:
                relations.append({**rel, "item": doc.item})

        # 4. dedup new slugs against existing pages ("related ≠ same")
        new_slugs = [s for s in proposals if s not in existing]
        merges: dict[str, str] = {}
        if new_slugs and existing:
            candidates_existing = [s for s, p in existing.items() if p.page_type in ("entity", "concept")][:MAX_KNOWN]
            if candidates_existing:
                data = await self._ask_json(
                    chat, sem, render("wiki_ingest", "dedup_system"), render(
                        "wiki_ingest", "dedup_user",
                        new="\n".join(f"{s}: {proposals[s]['title']} (aliases: {', '.join(sorted(proposals[s]['aliases']))})"
                                      for s in new_slugs),
                        existing="\n".join(f"{s}: {existing[s].title} (aliases: {', '.join(existing[s].aliases or [])})"
                                           for s in candidates_existing)))
                for new, old in (data.get("merges") or {}).items():
                    if new in proposals and new not in existing and old in existing:
                        merges[new] = old
        for new, old in merges.items():
            citations.setdefault(old, []).extend(citations.pop(new, []))
            proposals.setdefault(old, {"title": existing[old].title, "aliases": set()})["aliases"] |= \
                proposals[new]["aliases"] | {proposals[new]["title"]}
            proposals.pop(new, None)
        for rel in relations:
            rel["source"], rel["target"] = merges.get(rel["source"], rel["source"]), merges.get(rel["target"], rel["target"])

        # 5. taxonomy for the pages about to be created
        creating = [s for s in citations if s not in existing and citations[s]]
        folders = await self._taxonomy(kb, chat, sem, creating, proposals) if creating else {}

        # 6. reduce: one page write per slug
        known = sorted(set(existing) | set(creating))
        reduce_sem = asyncio.Semaphore(max(1, int(cfg["ingest_reduce_parallel"])))
        written = 0

        async def reduce(slug: str):
            nonlocal written
            async with reduce_sem:
                page = existing.get(slug)
                entry = proposals.get(slug, {"title": page.title if page else slug, "aliases": set()})
                if await self._reduce(kb, chat, sem, slug, page, entry, citations.get(slug, []), known, folders,
                                      language):
                    written += 1

        targets = set(s for s in citations if citations[s]) | {s for s in affected if s in existing
                                                               and existing[s].page_type in ("entity", "concept")}
        await asyncio.gather(*(reduce(s) for s in sorted(targets)))
        # summary pages, one per document
        for doc in docs:
            if doc.chunks:
                slug = f"summary/{doc.item.id}"
                page = await self._fresh(kb.id, slug)
                if await self._reduce(kb, chat, sem, slug, page, {"title": doc.item.title, "aliases": set()},
                                      [(doc.item, c) for c in doc.chunks[:20]], known, {slug: "Documents"}, language,
                                      page_type="summary", replace_sources=True):
                    written += 1
        await self._store_relations(kb, relations)
        archived += await self._archive_unsourced(kb)
        return {"pages": written, "archived": archived, "documents": len(docs), "merges": merges}

    async def _archive_unsourced(self, kb) -> int:
        """Generated pages left without any source (a re-ingested document stopped citing them)."""
        count = 0
        async with async_session_scope() as session:
            kb_row = await session.get(KnowledgeBase, kb.id)
            pages = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None), WikiPage.status == "published",
                WikiPage.page_type.in_(("entity", "concept", "summary")),
                WikiPage.last_edit_source == "pipeline",
                sa.func.jsonb_array_length(WikiPage.chunk_refs) == 0).with_for_update())).scalars().all()
            for page in pages:
                await self.wiki.write_page(session, kb_row, page, slug=page.slug, title=page.title,
                                           content=page.content, summary=page.summary, page_type=page.page_type,
                                           status="archived", source="pipeline")
                count += 1
        return count

    async def _fresh(self, kb_id, slug) -> WikiPage | None:
        async with async_session_scope(commit=False) as session:
            page = await WikiService._page(session, kb_id, slug)
            if page is not None:
                session.expunge(page)
            return page

    async def _taxonomy(self, kb, chat, sem, slugs, proposals) -> dict[str, str]:
        async with async_session_scope(commit=False) as session:
            from kb_core.orm.models.ext import WikiFolder

            paths = list((await session.execute(sa.select(WikiFolder.path).where(
                WikiFolder.knowledge_base_id == kb.id).order_by(WikiFolder.path).limit(200))).scalars())
        data = await self._ask_json(chat, sem, render("wiki_ingest", "taxonomy_system"), render(
            "wiki_ingest", "taxonomy_user", folders="\n".join(paths) or "(none)",
            pages="\n".join(f"{s}: {proposals.get(s, {}).get('title', s)}" for s in slugs)))
        out = {}
        for slug in slugs:
            path = str((data.get("folders") or {}).get(slug) or "").strip().strip("/")
            parts = [p.strip()[:80] for p in path.split("/") if p.strip()]
            # small models echo the page back as its folder ("Concept/Batch System"): drop a
            # page-type segment and a last segment that is the page's own title
            if parts and parts[0].lower() in ("concept", "concepts", "entity", "entities", "summary"):
                parts = parts[1:]
            title = proposals.get(slug, {}).get("title") or slug.rpartition("/")[2]
            if parts and wt.title_key(parts[-1]) in (wt.title_key(title), slug.rpartition("/")[2]):
                parts = parts[:-1]
            if not parts:
                parts = ["Entities" if slug.startswith("entity/") else "Concepts"]
            if "/".join(parts) not in paths:
                parts = parts[:2]                        # new paths: at most two levels
            out[slug] = "/".join(parts[:3])
        return out

    async def _reduce(self, kb, chat, sem, slug, page, entry, cited, known, folders, language,
                      page_type: str | None = None, replace_sources: bool = False) -> bool:
        """One grounded page write: existing content + cited passages → SUMMARY + body."""
        page_type = page_type or (page.page_type if page else slug.split("/")[0])
        passages, label_of = [], {}
        seen = set()
        for item, chunk in cited:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            label = wt.chunk_label(len(passages))
            label_of[label] = (item, chunk)
            passages.append(f"[{label}] ({item.title}) {chunk.content[:PASSAGE_CHARS]}")
        if page is not None and not passages:          # re-reduce after retraction: its remaining sources
            remaining = [parse_ref(r) for r in page.chunk_refs or []]
            async with async_session_scope(commit=False) as session:
                rows = (await session.execute(sa.select(Chunk, Knowledge).join(Knowledge, Knowledge.id == Chunk.knowledge_id)
                                              .where(Chunk.id.in_([cid for _, cid, _ in remaining] or ["-"])))).all()
            for chunk, item in rows[:40]:
                label = wt.chunk_label(len(passages))
                label_of[label] = (item, chunk)
                passages.append(f"[{label}] ({item.title}) {chunk.content[:PASSAGE_CHARS]}")
        if not passages:
            return False
        handles = wt.Handles([s for s in known if s != slug][:MAX_KNOWN])
        handles.ref(slug)
        known_lines = "\n".join(f"{handles.ref(s)}: {s.rpartition('/')[2].replace('-', ' ')}"
                                for s in known if s != slug)[:8000]
        existing = (page.content if page else "") or "(new page)"
        messages = [{"role": "system", "content": render("wiki_ingest", "page_system", language=language)},
                    {"role": "user", "content": render(
                        "wiki_ingest", "page_user", title=entry["title"], slug_ref=handles.ref(slug), page_type=page_type,
                        aliases=", ".join(sorted(entry.get("aliases") or [])) or "(none)", existing=existing,
                        known=known_lines or "(none)", passages="\n\n".join(passages))}]
        output = ""
        for attempt in range(MAX_CONTINUATIONS + 1):
            async with sem:
                reply = await chat.complete(messages, max_tokens=3000, temperature=0.2)
            output += reply.get("content") or ""
            if reply.get("finish_reason") != "length":
                break
            messages = messages + [{"role": "assistant", "content": reply.get("content") or ""},
                                   {"role": "user", "content": render("wiki_ingest", "continue_user")}]
        else:
            LOG.warning("wiki page %s still truncated after %d continuations; not written", slug, MAX_CONTINUATIONS)
            return False
        summary, body = wt.split_summary(output)
        if not body.strip():
            return False
        body = _LABEL.sub(lambda m: f"[src:{label_of[m.group(1)][1].seq_id}]" if m.group(1) in label_of else "", body)
        body = wt.strip_self_links(handles.restore(body), slug)
        new_refs = [chunk_ref(item.id, chunk.id, chunk.seq_id) for item, chunk in label_of.values()]
        old_refs = [] if (replace_sources or page is None) else list(page.chunk_refs or [])
        # a marker the model copied wrong or invented cites nothing this page has: drop it
        allowed = {parse_ref(r)[2] for r in old_refs + new_refs}
        body = SRC.sub(lambda m: m.group(0) if int(m.group(1)) in allowed else "", body)
        cited_seqs = {int(n) for n in SRC.findall(body)}
        chunk_refs = [r for r in old_refs + new_refs if parse_ref(r)[2] in cited_seqs or not cited_seqs]
        titles = {item.id: item.title for item, _ in label_of.values()}
        old_sources = [] if (replace_sources or page is None) else list(page.source_refs or [])
        source_ids = {parse_ref(r)[0] for r in chunk_refs}
        source_refs = [r for r in old_sources if r.split("|", 1)[0] in source_ids]
        source_refs += [f"{kid}|{title}" for kid, title in titles.items() if kid in source_ids]
        draft = SimpleNamespace(title=entry["title"], summary=summary, content=body)
        vectors = await self.wiki.embed(kb, passages_of(draft))
        async with async_session_scope() as session:
            await session.execute(sa.text("SELECT pg_advisory_xact_lock(:k)"),
                                  {"k": int(hashlib.sha256(f"{kb.id}:{slug}".encode()).hexdigest()[:15], 16)})
            current = await WikiService._page(session, kb.id, slug, for_update=True)
            folder_id = None
            if current is None and folders.get(slug):
                folder_id = await self.wiki._ensure_folder(session, kb.id, folders[slug], generated=True)
            kb_row = await session.get(KnowledgeBase, kb.id)
            await self.wiki.write_page(session, kb_row, current, slug=slug, title=entry["title"], content=body,
                                       summary=summary, page_type=page_type,
                                       aliases=sorted(set(current.aliases if current else []) | set(entry.get("aliases") or [])),
                                       status="published", source="pipeline", source_refs=source_refs, chunk_refs=chunk_refs,
                                       folder_id=folder_id, vectors=vectors)
        return True

    async def _retract(self, kb, knowledge_ids: set[str], keep_for_reingest: set[str]) -> tuple[set[str], int]:
        """Remove documents' citations: source refs, chunk refs, [src:N] markers and relations.
        Pages left without sources are archived (unless the document is being re-ingested); the
        others are returned to be re-reduced."""
        if not knowledge_ids:
            return set(), 0
        affected, archived = set(), 0
        async with async_session_scope() as session:
            pages = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None),
                sa.or_(*[WikiPage.source_ids.contains([k]) for k in knowledge_ids])).with_for_update())).scalars().all()
            kb_row = await session.get(KnowledgeBase, kb.id)
            for page in pages:
                gone = {parse_ref(r)[2] for r in page.chunk_refs or [] if parse_ref(r)[0] in knowledge_ids}
                content = SRC.sub(lambda m: "" if int(m.group(1)) in gone else m.group(0), page.content or "")
                refs = [r for r in page.chunk_refs or [] if parse_ref(r)[0] not in knowledge_ids]
                sources = [r for r in page.source_refs or [] if r.split("|", 1)[0] not in knowledge_ids]
                only_reingest = not refs and set(page.source_ids or []) <= keep_for_reingest
                status = page.status
                if not refs and not only_reingest:
                    status = "archived"
                    archived += 1
                elif refs:
                    affected.add(page.slug)
                await self.wiki.write_page(session, kb_row, page, slug=page.slug, title=page.title, content=content,
                                           summary=page.summary, page_type=page.page_type, status=status,
                                           source="pipeline", source_refs=sources, chunk_refs=refs)
            rels = (await session.execute(sa.select(WikiRelation).where(
                WikiRelation.knowledge_base_id == kb.id,
                sa.or_(*[WikiRelation.source_ids.contains([k]) for k in knowledge_ids])))).scalars().all()
            for rel in rels:
                rel.source_ids = [k for k in rel.source_ids if k not in knowledge_ids]
                rel.chunk_refs = [r for r in rel.chunk_refs if parse_ref(r)[0] not in knowledge_ids]
                if not rel.source_ids:
                    await session.delete(rel)
        return affected, archived

    async def _store_relations(self, kb, relations: list[dict]) -> None:
        if not relations:
            return
        from sqlalchemy.dialects.postgresql import insert

        async with async_session_scope() as session:
            known = set((await session.execute(sa.select(WikiPage.slug).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None)))).scalars())
            for rel in relations:
                source, target = rel.get("source"), rel.get("target")
                name = str(rel.get("relation") or "").strip()[:128]
                if not name or source not in known or target not in known or source == target or not rel["chunks"]:
                    continue                      # every relationship cites at least one chunk
                refs = [chunk_ref(rel["item"].id, c.id, c.seq_id) for c in rel["chunks"]]
                row = (await session.execute(sa.select(WikiRelation).where(
                    WikiRelation.knowledge_base_id == kb.id, WikiRelation.source_slug == source,
                    WikiRelation.relation == name, WikiRelation.target_slug == target).with_for_update())).scalar_one_or_none()
                if row is None:
                    session.add(WikiRelation(knowledge_base_id=kb.id, source_slug=source, relation=name,
                                             target_slug=target, description=str(rel.get("description") or "")[:2000],
                                             chunk_refs=refs, source_ids=[rel["item"].id], weight=1.0))
                    await session.flush()
                else:
                    row.chunk_refs = list(dict.fromkeys(list(row.chunk_refs) + refs))
                    row.source_ids = list(dict.fromkeys(list(row.source_ids) + [rel["item"].id]))
                    row.weight = float(len(row.source_ids))
                    row.description = row.description or str(rel.get("description") or "")[:2000]

    # == finalize and communities ================================================================================
    async def _finalize_task(self, ctx: TaskContext) -> dict:
        return await self.wiki.finalize(ctx.payload["knowledge_base_id"])

    async def _communities_task(self, ctx: TaskContext) -> int:
        kb_id = ctx.payload["knowledge_base_id"]
        async with async_session_scope(commit=False) as session:
            kb = await session.get(KnowledgeBase, kb_id)
            if kb is None or kb.deleted_at is not None:
                return 0
            session.expunge(kb)
            pending = (await session.execute(sa.select(WikiCommunity).where(
                WikiCommunity.knowledge_base_id == kb_id, WikiCommunity.page_slug == ""))).scalars().all()
            pages = {p.slug: p for p in (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb_id, WikiPage.deleted_at.is_(None)))).scalars()}
            rels = (await session.execute(sa.select(WikiRelation).where(
                WikiRelation.knowledge_base_id == kb_id))).scalars().all()
            for obj in [*pending, *pages.values(), *rels]:
                session.expunge(obj)
        if not pending:
            return 0
        await self.wiki.ensure_index(kb_id)
        chat = await self._chat(kb)
        sem = asyncio.Semaphore(max(1, int(self.wiki.config(kb)["ingest_max_inflight"])))
        language = self.wiki.config(kb)["language"]
        language = "the language of the source documents" if language == "auto" else language
        written = 0
        for community in pending:
            members = [pages[s] for s in community.member_slugs if s in pages]
            if len(members) < 2:
                continue
            handles = wt.Handles([m.slug for m in members])
            member_set = set(community.member_slugs)
            relation_lines = [f"{handles.ref(r.source_slug)} {r.relation} {handles.ref(r.target_slug)}"
                              for r in rels if r.source_slug in member_set and r.target_slug in member_set]
            reply = await self._ask(chat, sem, render("wiki_graph", "community_system", language=language),
                                    render("wiki_graph", "community_user",
                                           members="\n".join(f"{handles.ref(m.slug)}: {m.title}: {m.summary}"
                                                             for m in members),
                                           relations="\n".join(relation_lines) or "(none)"), 2000)
            summary, body = wt.split_summary(reply.get("content") or "")
            if not body:
                continue
            body = handles.restore(body)
            slug = f"community/l{community.level}-{community.fingerprint[:10]}"
            title = (f"Community (level {community.level}): {', '.join(m.title for m in members[:3])}"
                     + (" …" if len(members) > 3 else ""))
            chunk_refs = list(dict.fromkeys(r for m in members for r in (m.chunk_refs or [])))[:200]
            source_refs = list(dict.fromkeys(r for m in members for r in (m.source_refs or [])))
            vectors = await self.wiki.embed(kb, passages_of(SimpleNamespace(title=title, summary=summary, content=body)))
            async with async_session_scope() as session:
                kb_row = await session.get(KnowledgeBase, kb_id)
                row = await session.get(WikiCommunity, community.id, with_for_update=True)
                if row is None:
                    continue
                current = await WikiService._page(session, kb_id, slug, for_update=True)
                folder_id = await self.wiki._ensure_folder(session, kb_id, "Communities", generated=True)
                await self.wiki.write_page(session, kb_row, current, slug=slug, title=title[:500], content=body,
                                           summary=summary, page_type="community", source="pipeline",
                                           source_refs=source_refs, chunk_refs=chunk_refs, folder_id=folder_id,
                                           vectors=vectors)
                row.page_slug = slug
            written += 1
        return written

    # == global search ===========================================================================================
    async def global_search(self, ctx: AuthContext, kb_ref: str, query: str, level: int = 0) -> dict:
        query = (query or "").strip()
        if not query:
            raise Invalid("query is required")
        async with async_session_scope(commit=False) as session:
            kb = await self.wiki._kb(session, ctx, kb_ref)
            session.expunge(kb)
            pages = (await session.execute(sa.select(WikiPage).where(
                WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None), WikiPage.status == "published",
                WikiPage.page_type == "community"))).scalars().all()
            community_level = {c.page_slug: c.level for c in (await session.execute(sa.select(WikiCommunity).where(
                WikiCommunity.knowledge_base_id == kb.id))).scalars()}
            pages = [p for p in pages if community_level.get(p.slug) == int(level)] or [
                p for p in (await session.execute(sa.select(WikiPage).where(
                    WikiPage.knowledge_base_id == kb.id, WikiPage.deleted_at.is_(None), WikiPage.status == "published",
                    WikiPage.page_type == "summary"))).scalars()]
            for p in pages:
                session.expunge(p)
        if not await self.wiki.model_id(kb):
            raise Invalid("global search needs a chat model")
        if not pages:
            return {"answer": "", "points": [], "sources": []}
        chat = await self._chat(kb)
        sem = asyncio.Semaphore(max(1, int(self.wiki.config(kb)["ingest_max_inflight"])))
        language = "the language of the question"

        async def map_one(page):
            data = await self._ask_json(chat, sem, render("wiki_graph", "global_map_system"),
                                        render("wiki_graph", "global_map_user", query=query, slug=page.slug,
                                               content=f"{page.title}\n{page.summary}\n\n{page.content}"[:6000]), 600)
            try:
                score = float(data.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            return {"slug": page.slug, "title": page.title, "score": max(0.0, min(100.0, score)),
                    "answer": str(data.get("answer") or "").strip()}

        points = [p for p in await asyncio.gather(*(map_one(p) for p in pages[:40])) if p["score"] > 0 and p["answer"]]
        points.sort(key=lambda p: (-p["score"], p["slug"]))
        top = points[:5]
        if not top:
            return {"answer": "", "points": [], "sources": []}
        reply = await self._ask(chat, sem, render("wiki_graph", "global_reduce_system", language=language),
                                render("wiki_graph", "global_reduce_user", query=query,
                                       points="\n\n".join(f"[[{p['slug']}|{p['title']}]] (score {p['score']:.0f}): "
                                                          f"{p['answer']}" for p in top)), 1500)
        return {"answer": (reply.get("content") or "").strip(), "points": top, "sources": [p["slug"] for p in top]}
