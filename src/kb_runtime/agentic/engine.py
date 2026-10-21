"""
Agentic search (docs/agentic-search.md): one request, one question, read-only.

    AgenticService.prepare(ctx, body)   validate; resolve model, scope, preset; take a per-owner slot
      └─ Run.frames()                   async iterator of StreamResponse frames, ending with `complete`
           ├─ quick   rewrite → hybrid search → merge → answer (or the evidence)
           └─ deep    observe (trim) → think (model + tool schemas) → act (tools in parallel) → record

The run is an asyncio task bound to the request: closing the iterator (client disconnect)
cancels it. Nothing is persisted; the steps are in the response and the log.

Model calls are not streamed (kb_runtime.models.clients.ChatModel): frames are emitted per step.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.kb import Chunk, Knowledge, KnowledgeBase
from kb_runtime import access
from kb_runtime.agentic import citations
from kb_runtime.agentic.handles import KIND_NAMES, Handles
from kb_runtime.agentic.presets import PRESETS, Preset, describe, tools_for
from kb_runtime.agentic.tools import Scope, ToolContext, ToolError
from kb_runtime.agentic.tools.knowledge import chunk_handle, doc_handle, hit_handle, run_searches
from kb_runtime.errors import Busy, Forbidden, Invalid, NotFound
from kb_runtime.faq.rules import apply_priority, priority_settings
from kb_runtime.models.clients import ModelError
from kb_runtime.prompts import render
from kb_runtime.retrieval.merge import merge
from kb_runtime.retrieval.rerank import jaccard, tokens
from kb_runtime.retrieval.search import SearchOptions, retrieval_defaults

LOG = get_logger(__name__)

TRIMMED = "[earlier tool result omitted to save context; call the tool again if you need it]"
STOPS = ("answered", "evidence_complete", "no_results", "max_iterations", "token_limit", "timeout", "cancelled",
         "error")
_CLAUSE = re.compile(r"[,.;:?!，。；：？！、]+")
_STOPWORDS = frozenset("a an and are as at be by can do does for from how i in is it me my of on or the to was what "
                       "when where which who why will with you".split())
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


@dataclass
class Settings:
    model: str = ""
    max_iterations: int = 10
    max_iterations_limit: int = 20
    timeout: float = 180
    max_input_tokens: int = 200_000
    per_owner: int = 2
    tool_timeout: float = 60
    max_tool_output_chars: int = 16_000
    max_evidence: int = 30
    stall_timeout: float = 120
    quick_rewrite: bool = True
    quick_query_expansion: bool = True
    citations: bool = True
    answer_max_tokens: int = 2048
    fallback_strategy: str = "model"          # model | fixed
    fallback_response: str = "Sorry, I could not find relevant information in the knowledge base."
    sources_footer: bool = True                 # end every answer with its numbered "Sources" list
    max_uncited_sources: int = 5                # sources listed when the model cited nothing

    @classmethod
    def from_core_config(cls) -> "Settings":
        out = cls()
        for name, value in vars(cls()).items():
            if config_has_option("agentic", name):
                raw = config_get("agentic", name).strip()
                if not raw:
                    continue
                if isinstance(value, bool):
                    setattr(out, name, raw.lower() in ("1", "true", "yes", "on"))
                elif isinstance(value, str):
                    setattr(out, name, raw)
                else:
                    setattr(out, name, type(value)(raw))
        if out.fallback_strategy not in ("model", "fixed"):
            raise ValueError("[agentic] fallback_strategy must be model or fixed")
        return out


def frame(run_id: str, response_type: str, content: str = "", **fields) -> dict:
    return {"id": run_id, "response_type": response_type, "content": content, "done": response_type == "complete",
            "knowledge_references": fields.pop("knowledge_references", None), "tool_calls": fields.pop("tool_calls", None),
            "data": fields.pop("data", None), "usage": fields.pop("usage", None),
            "finish_reason": fields.pop("finish_reason", None)}


def language_of(query: str) -> str:
    return "Chinese" if _CJK.search(query or "") else "the language of the question"


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("not an object")
    return value


@dataclass
class Run:
    service: "AgenticService"
    auth: AuthContext
    query: str
    preset: Preset
    output: str
    model_id: str
    chat: object
    context_window: int
    scope: Scope
    max_iterations: int
    settings: Settings
    release: object
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    handles: Handles = field(default_factory=Handles)
    steps: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    stopped: str = ""
    answer: str | None = None
    evidence: list = field(default_factory=list)
    cited: list = field(default_factory=list)
    selected: list = field(default_factory=list)       # (handle, reason) with output=evidence
    route: dict | None = None                         # the auto preset's routing decision
    allow_writes: bool = False                        # write tools offered (the request's allow_writes)
    writes: list = field(default_factory=list)        # what the write tools changed
    web_cache: Any = None                             # FetchCache for web_fetch / web_search
    sources: list = field(default_factory=list)       # the answer's numbered source citations
    _queue: asyncio.Queue | None = None
    _seen_call_ids: set = field(default_factory=set)

    # -- frames --------------------------------------------------------------------------------
    async def emit(self, payload: dict) -> None:
        if self._queue is not None:
            response_type = payload.pop("response_type")
            content = payload.pop("content", "")
            await self._queue.put(frame(self.run_id, response_type, content, **payload))

    async def frames(self) -> AsyncIterator[dict]:
        self._queue = asyncio.Queue()
        task = asyncio.create_task(self._main())
        try:
            while True:
                item = await self._queue.get()
                yield item
                if item["response_type"] == "complete":
                    break
        finally:
            if not task.done():
                task.cancel()
                self.stopped = "cancelled"
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                self._log()
            self.release()

    def result(self) -> dict:
        data = {"query": self.query, "preset": self.preset.name, "output": self.output,
                **({"route": self.route} if self.route else {}),
                **({"writes": self.writes} if self.allow_writes else {}),
                "sources": self.sources,
                "evidence": self.evidence, "steps": self.steps, "stopped": self.stopped,
                "usage": {"model_id": self.model_id, **self.usage}}
        if self.output == "answer" and self.answer is not None:
            data["answer"] = self.answer
        return data

    def _log(self) -> None:
        LOG.info("agentic search %s owner=%s preset=%s stopped=%s llm_calls=%d input_tokens=%d output_tokens=%d",
                 self.run_id, self.auth.subject, self.preset.name, self.stopped, self.usage["llm_calls"],
                 self.usage["input_tokens"], self.usage["output_tokens"])

    async def _main(self) -> None:
        await self.emit({"response_type": "agent_query", "data": {
            "query": self.query, "preset": self.preset.name, "output": self.output,
            "knowledge_base_ids": list(self.scope.kb_ids)}})
        try:
            body = {"single_pass": self._quick, "router": self._auto}.get(self.preset.engine, self._loop)()
            await asyncio.wait_for(body, timeout=self.settings.timeout)
        except asyncio.TimeoutError:
            self.stopped, self.answer = "timeout", None
            self.evidence = self._ranked_evidence([])
        except ModelError as exc:
            self.stopped = "error"
            self.evidence = self.evidence or self._ranked_evidence([])
            await self.emit({"response_type": "error", "content": f"model call failed: {exc}"})
        except Exception as exc:          # never leave the caller without a `complete` frame
            LOG.exception("agentic search %s failed", self.run_id)
            self.stopped = "error"
            await self.emit({"response_type": "error", "content": f"internal error ({type(exc).__name__})"})
        self._log()
        await self.emit({"response_type": "complete", "data": self.result(),
                         "usage": {"model_id": self.model_id, **self.usage}, "finish_reason": self.stopped})

    # -- model calls ------------------------------------------------------------------------------
    async def _call(self, messages: list[dict], temperature: float, tools: list[dict] | None = None,
                    max_tokens: int | None = None) -> dict:
        reply = await self.chat.complete(messages, max_tokens=max_tokens, temperature=temperature, tools=tools,
                                         timeout=self.settings.stall_timeout)
        usage = reply.get("usage") or {}
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.usage["output_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return reply

    def _answer_tokens(self) -> int | None:
        return self.settings.answer_max_tokens or None

    def tool_context(self) -> ToolContext:
        return ToolContext(self.auth, self.service.services, self.scope, self.handles, self.emit,
                           self.settings.max_tool_output_chars, web_cache=self.web_cache, writes=self.writes)

    # -- prompts ----------------------------------------------------------------------------------------
    async def _directory(self, tools: list) -> str:
        if not self.scope.kb_ids:
            listing = "No knowledge base is in scope."
        else:
            async with async_session_scope(commit=False) as session:
                counts = await self.service.services.knowledge_bases._knowledge_counts(session, self.auth,
                                                                                       self.scope.kb_ids)
            lines = []
            for kb_id in self.scope.kb_ids:
                kb = self.scope.kbs[kb_id]
                handle = self.handles.issue("b", kb_id, {"name": kb.name})
                lines.append(f'<knowledge_base id="{handle}" name="{citations._attr(kb.name)}" type="{kb.type}" '
                             f'documents="{counts.get(kb_id, 0)}">{citations._attr(kb.description[:300])}'
                             f'</knowledge_base>')
            listing = "\n".join(lines)
        now = datetime.datetime.now(datetime.timezone.utc)
        return render("agentic_search", "directory", knowledge_bases=listing,
                      tools=", ".join(t.name for t in tools) or "none",
                      current_time=now.strftime("%Y-%m-%d %H:%M UTC"), weekday=now.strftime("%A"),
                      language=language_of(self.query))

    async def _system(self, template: str, tools: list) -> str:
        parts = [render("agentic_search", template),
                 render("agentic_search", "writes") if self.allow_writes else "",
                 render("agentic_search", "evidence" if self.output == "evidence" else "answer",
                        language=language_of(self.query)),
                 await self._directory(tools), render("agentic_search", "trust"),
                 render("agentic_search", "protocol") if self.settings.citations or self.output == "evidence" else ""]
        return "\n\n".join(p.strip() for p in parts if p.strip())

    # -- finishing ----------------------------------------------------------------------------------
    def _evidence_entry(self, handle, **extra) -> dict:
        p = handle.payload
        entry = {"handle": handle.handle, "kind": KIND_NAMES[handle.prefix]}
        if p.get("citation_text"):
            entry["citation_text"] = p["citation_text"]
            entry["url"] = p.get("url")
        if handle.prefix == "c":
            entry.update(chunk_id=handle.key, knowledge_id=p.get("knowledge_id"),
                         knowledge_title=p.get("knowledge_title"), knowledge_base_id=p.get("knowledge_base_id"),
                         content=p.get("content", ""), score=p.get("score"), chunk_type=p.get("chunk_type", "text"),
                         **({"match": p["match"]} if p.get("match") else {}))
        else:
            entry.update({k: v for k, v in p.items() if k != "kind"})
        entry.update(extra)
        return entry

    def _ranked_evidence(self, first: list[str], **marks) -> list[dict]:
        """The `first` handles, then the other citable ones by score, capped at max_evidence."""
        citable = self.handles.citable()
        by_handle = {h.handle: h for h in citable}
        rest = sorted((h for h in citable if h.handle not in first),
                      key=lambda h: (-(h.payload.get("score") or 0), h.order))
        ordered = [by_handle[h] for h in first if h in by_handle] + rest
        return [self._evidence_entry(h, **({k: (h.handle in v) for k, v in marks.items()}))
                for h in ordered[:self.settings.max_evidence]]

    async def _references(self) -> None:
        refs = [self._evidence_entry(h) for h in self.handles.citable()]
        await self.emit({"response_type": "references", "knowledge_references": refs, "data": {"references": refs}})

    async def _finish_answer(self, text: str, stopped: str = "answered") -> None:
        await self._hydrate_citations()
        expanded, cited = citations.expand(text, self.handles) if self.settings.citations else (text, [])
        self.sources = self._sources(cited, stopped)
        if self.settings.sources_footer:
            expanded += citations.sources_footer(self.sources)
        self.answer, self.cited, self.stopped = expanded, cited, stopped
        await self._references()
        for piece in citations.pieces(expanded):
            await self.emit({"response_type": "answer", "content": piece})
        self.evidence = self._ranked_evidence(cited, cited=cited)

    def _sources(self, cited: list[str], stopped: str) -> list[dict]:
        """Every answer names its sources: what it cited, in citation order; when the model cited
        nothing, the best passages it was given (marked uncited); none when nothing was found."""
        if stopped == "no_results":
            return []
        chosen = [(self.handles.lookup(h), True) for h in cited]
        if not chosen:
            ranked = sorted(self.handles.citable(), key=lambda h: (-(h.payload.get("score") or 0), h.order))
            chosen = [(h, False) for h in ranked[:self.settings.max_uncited_sources]]
        out = []
        for n, (handle, was_cited) in enumerate(chosen, start=1):
            p = handle.payload
            out.append({"n": n, "handle": handle.handle, "kind": KIND_NAMES[handle.prefix],
                        "title": p.get("knowledge_title") or p.get("name") or p.get("title") or "",
                        "citation_text": p.get("citation_text") or p.get("knowledge_title") or p.get("name")
                        or p.get("title") or handle.key, "url": p.get("url"), "cited": was_cited,
                        **({"chunk_id": handle.key, "knowledge_id": p.get("knowledge_id")} if handle.prefix == "c" else {}),
                        **({"record_id": handle.key} if handle.prefix == "r" else {}),
                        **({"slug": p.get("slug")} if handle.prefix == "p" else {})})
        return out

    async def _hydrate_citations(self) -> None:
        """Passages that reached the run without their citation (grep, read_document) get it from
        their document: its citation, with section and page."""
        from kb_runtime.retrieval.search import chunk_citation

        missing = [h for h in self.handles.citable() if h.prefix == "c" and not h.payload.get("citation_text")]
        if not missing:
            return
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(Chunk, Knowledge).join(Knowledge, Knowledge.id == Chunk.knowledge_id)
                                          .where(Chunk.id.in_([h.key for h in missing])))).all()
        by_id = {chunk.id: chunk_citation(item, chunk) for chunk, item in rows}
        for handle in missing:
            found = by_id.get(handle.key)
            if found:
                handle.payload["citation_text"] = found["citation_text"]
                handle.payload["url"] = found["citation"].get("url")

    async def _finish_evidence(self, text: str, stopped: str = "evidence_complete") -> None:
        await self._hydrate_citations()
        self.selected = citations.parse_evidence(text, self.handles)
        self.stopped = stopped
        await self._references()
        if self.selected:
            self.evidence = [self._evidence_entry(self.handles.lookup(h), reason=reason)
                             for h, reason in self.selected][:self.settings.max_evidence]
        else:
            self.evidence = self._ranked_evidence([])

    async def _finish(self, text: str, stopped: str | None = None) -> None:
        if self.output == "evidence":
            await self._finish_evidence(text, stopped or "evidence_complete")
        else:
            await self._finish_answer(text, stopped or "answered")

    # -- deep: the ReAct loop -------------------------------------------------------------------------
    def _budget(self) -> int:
        return int((self.context_window or 32_768) * 3 * 0.75)

    @staticmethod
    def _size(messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += len(m.get("content") or "")
            for call in m.get("tool_calls") or []:
                total += len(call["function"].get("arguments") or "")
        return total

    def _trim(self, messages: list[dict]) -> bool:
        trimmed = False
        budget = self._budget()
        while self._size(messages) > budget:
            oldest = next((m for m in messages if m["role"] == "tool" and m["content"] != TRIMMED), None)
            if oldest is None:
                break
            oldest["content"] = TRIMMED
            trimmed = True
        return trimmed

    async def _run_tool(self, tc: ToolContext, tools: dict, call: dict) -> tuple[dict, str]:
        name, call_id = call["name"], call["id"]
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            args = None
        await self.emit({"response_type": "tool_call", "data": {"tool_call_id": call_id, "tool_name": name,
                                                               "arguments": args if isinstance(args, dict) else
                                                               call.get("arguments")},
                         "tool_calls": [{"id": call_id, "name": name, "arguments": call.get("arguments")}]})
        started = time.monotonic()
        status = "ok"
        if name not in tools:
            output, status = f"Error: unknown tool {name!r}. Available tools: {', '.join(tools)}", "error"
        elif not isinstance(args, dict):
            output, status = "Error: the arguments must be a JSON object", "error"
        else:
            try:
                output = await asyncio.wait_for(tools[name].run(tc, args), timeout=self.settings.tool_timeout)
            except ToolError as exc:
                output, status = f"Error: {exc}", "error"
            except asyncio.TimeoutError:
                output, status = f"Error: {name} timed out after {self.settings.tool_timeout:.0f} s", "error"
            except (NotFound, Invalid) as exc:
                output, status = f"Error: {exc}", "error"
            except Exception as exc:
                LOG.exception("agentic tool %s failed", name)
                output, status = f"Error: {name} failed internally ({type(exc).__name__})", "error"
        duration = int((time.monotonic() - started) * 1000)
        await self.emit({"response_type": "tool_result", "data": {
            "tool_call_id": call_id, "tool_name": name, "success": status == "ok",
            "output_preview": output[:500], "duration_ms": duration}})
        return ({"id": call_id, "name": name, "args": args if isinstance(args, dict) else call.get("arguments"),
                 "result_summary": output[:300], "duration_ms": duration, "status": status}, output)

    async def _loop(self) -> None:
        kinds = {kb.type for kb in self.scope.kbs.values()}
        has_wiki = any((kb.indexing_strategy or {}).get("wiki_enabled") for kb in self.scope.kbs.values())
        tools = tools_for(self.preset, has_documents=bool(kinds & {"document", "faq"}) or not kinds,
                          has_records="records" in kinds, extra=self.service.plugin_tools(), has_wiki=has_wiki,
                          allow_writes=self.allow_writes)
        by_name = {t.name: t for t in tools}
        schemas = [t.schema() for t in tools]
        messages = [{"role": "system", "content": await self._system(self.preset.template, tools)},
                    {"role": "user", "content": self.query}]
        tc = self.tool_context()
        stopped = "max_iterations"
        for iteration in range(1, self.max_iterations + 1):
            if self.usage["input_tokens"] > self.settings.max_input_tokens:
                stopped = "token_limit"
                break
            if self._trim(messages):
                await self.emit({"response_type": "context_compacted", "data": {"iteration": iteration}})
            reply = await self._call(messages, self.preset.temperature, tools=schemas)
            if reply.get("reasoning"):
                await self.emit({"response_type": "thinking", "content": reply["reasoning"]})
            calls = reply.get("tool_calls") or []
            if not calls:
                text = (reply.get("content") or "").strip()
                if not text:
                    reply = await self._call(messages + [{"role": "user", "content": render(
                        "agentic_search", "answer_now")}], 0.3, max_tokens=self._answer_tokens())
                    text = (reply.get("content") or "").strip()
                self.steps.append({"iteration": iteration, "thought": "", "tool_calls": [], "final": True})
                await self._finish(text)
                return
            thought = (reply.get("content") or "").strip()
            if thought:
                await self.emit({"response_type": "thinking", "content": thought})
            for index, call in enumerate(calls):
                if not call.get("id") or call["id"] in self._seen_call_ids:
                    call["id"] = f"call_{iteration}_{index}"
                self._seen_call_ids.add(call["id"])
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or "{}"}}
                for c in calls]})
            results = await asyncio.gather(*(self._run_tool(tc, by_name, call) for call in calls))
            for (record, output), call in zip(results, calls):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            self.steps.append({"iteration": iteration, "thought": thought, "tool_calls": [r for r, _ in results]})
        # forced final call, without tools
        key = "force_final_evidence" if self.output == "evidence" else "force_final"
        self._trim(messages)
        reply = await self._call(messages + [{"role": "user", "content": render("agentic_search", key)}], 0.3,
                                 max_tokens=self._answer_tokens())
        self.steps.append({"iteration": len(self.steps) + 1, "thought": "", "tool_calls": [], "final": True,
                           "forced": True})
        await self._finish((reply.get("content") or "").strip(), stopped)

    # -- quick: one pass ------------------------------------------------------------------------------
    async def _rewrite(self) -> str:
        if not self.settings.quick_rewrite:
            return self.query
        try:
            reply = await self._call([
                {"role": "system", "content": render("agentic_search", "rewrite_system")},
                {"role": "user", "content": render("agentic_search", "rewrite_user", query=self.query)}],
                0.3, max_tokens=150)
            rewritten = str(_json_object(reply.get("content") or "").get("rewrite_query") or "").strip()
            return rewritten or self.query
        except (ValueError, ModelError):
            return self.query

    async def _quick_search(self, tc: ToolContext, query: str) -> list[dict]:
        cfg = retrieval_defaults()
        top_k = int(cfg["rerank_top_k"])
        hits = await run_searches(tc, [query], list(self.scope.kb_ids), list(self.scope.knowledge_ids), "hybrid",
                                  top_k)
        seen = {h["id"] for h in hits}
        if self.settings.quick_query_expansion and len(hits) < int(cfg["embedding_top_k"]):
            stripped = " ".join(w for w in re.findall(r"\w+", query) if w.lower() not in _STOPWORDS)
            candidates = [stripped] + [v.strip() for v in _CLAUSE.split(query)]
            variants = [v for v in dict.fromkeys(candidates) if len(v) >= 2 and v != query][:3]
            options = SearchOptions(top_k=top_k, knowledge_ids=list(self.scope.knowledge_ids),
                                    tag_ids=list(self.scope.tag_ids), disable_vector=True)
            for variant in variants:
                for hit in await self.service.services.search.search(self.auth, variant, list(self.scope.kb_ids),
                                                                     options):
                    if hit["id"] not in seen:
                        seen.add(hit["id"])
                        hits.append({**hit, "score": hit["score"] * 0.9})
        # near-duplicates: one text inside the other, or token overlap ≥ 0.85 of the smaller set
        kept: list[dict] = []
        for hit in sorted(hits, key=lambda h: -h["score"]):
            words = tokens(hit["content"])
            duplicate = False
            for other in kept:
                other_words = other["_tokens"]
                small = min(len(words), len(other_words)) or 1
                if hit["content"] in other["content"] or other["content"] in hit["content"] \
                        or len(words & other_words) / small >= 0.85:
                    duplicate = True
                    break
            if not duplicate:
                kept.append({**hit, "_tokens": words})
        for hit in kept:
            hit.pop("_tokens", None)
        contexts = merge(kept, await self._lookup(kept), self._chunk_size())
        contexts.sort(key=lambda c: (-c["score"], c["knowledge_id"], c.get("chunk_index", 0)))
        return apply_priority(contexts, top_k, priority_settings())

    def _chunk_size(self) -> int:
        sizes = [int((kb.chunking_config or {}).get("chunk_size") or 512) for kb in self.scope.kbs.values()]
        return min(sizes) if sizes else 512

    async def _lookup(self, hits: list[dict]):
        ids = {i for h in hits for i in (h.get("parent_chunk_id"), h.get("pre_chunk_id"), h.get("next_chunk_id")) if i}
        rows = {}
        if ids:
            async with async_session_scope(commit=False) as session:
                for chunk in (await session.execute(sa.select(Chunk).where(Chunk.id.in_(ids)))).scalars():
                    rows[chunk.id] = {"id": chunk.id, "content": chunk.content, "start_at": chunk.start_at,
                                      "end_at": chunk.end_at, "is_enabled": chunk.is_enabled}
        return rows.get

    def _contexts_text(self, tc: ToolContext, contexts: list[dict]) -> str:
        faq = [c for c in contexts if c.get("chunk_type") == "faq"]
        records = [c for c in contexts if c.get("chunk_type") in ("record", "wiki_page")]
        docs = [c for c in contexts if c.get("chunk_type") not in ("faq", "record", "wiki_page")]
        out = []
        if faq:
            out.append('<source type="faq" priority="high">')
            for c in faq:
                handle = chunk_handle(tc, c)
                exact = ' match="exact"' if c.get("match") == "exact" else ""
                out.append(f'<context id="{handle}"{exact}>\n{c["content"]}\n</context>')
            out.append("</source>")
        if records:
            out.append('<source type="records and wiki pages">')
            for c in records:
                out.append(f'<context id="{hit_handle(tc, c)}">\n{c["content"]}\n</context>')
            out.append("</source>")
        by_doc: dict[str, list[dict]] = {}
        for c in docs:
            by_doc.setdefault(c["knowledge_id"], []).append(c)
        for knowledge_id, group in by_doc.items():
            first = group[0]
            doc_handle(tc, knowledge_id, first.get("knowledge_title", ""), first.get("knowledge_base_id", ""))
            out.append(f'<document title="{citations._attr(first.get("knowledge_title"))}" '
                       f'source="{citations._attr((first.get("citation") or {}).get("text") or "")}" '
                       f'date="{(first.get("knowledge_date") or "")[:10]}" '
                       f'description="{citations._attr((first.get("knowledge_description") or "")[:300])}">')
            for c in group:
                out.append(f'<context id="{chunk_handle(tc, c)}">\n{c["content"]}\n</context>')
            out.append("</document>")
        return "\n".join(out)

    async def _quick(self) -> None:
        tc = self.tool_context()
        query = await self._rewrite()
        call_id = "call_1_0"
        await self.emit({"response_type": "tool_call", "data": {"tool_call_id": call_id, "tool_name": "search_knowledge",
                                                               "arguments": {"queries": [query]}},
                         "tool_calls": [{"id": call_id, "name": "search_knowledge",
                                         "arguments": json.dumps({"queries": [query]})}]})
        started = time.monotonic()
        contexts = await self._quick_search(tc, query)
        rendered = self._contexts_text(tc, contexts)
        duration = int((time.monotonic() - started) * 1000)
        summary = f"{len(contexts)} passages"
        await self.emit({"response_type": "tool_result", "data": {"tool_call_id": call_id,
                                                                 "tool_name": "search_knowledge", "success": True,
                                                                 "output_preview": rendered[:500] or summary,
                                                                 "duration_ms": duration}})
        self.steps.append({"iteration": 1, "thought": "", "final": True, "tool_calls": [
            {"id": call_id, "name": "search_knowledge", "args": {"queries": [query]}, "result_summary": summary,
             "duration_ms": duration, "status": "ok"}]})
        language = language_of(self.query)
        if not contexts:
            if self.output == "evidence":
                self.stopped, self.evidence = "no_results", []
                await self._references()
                return
            if self.settings.fallback_strategy == "fixed":
                await self._finish_answer(self.settings.fallback_response, "no_results")
                return
            reply = await self._call([{"role": "system", "content": render("agentic_search", "fallback",
                                                                           language=language)},
                                      {"role": "user", "content": self.query}], 0.3, max_tokens=self._answer_tokens())
            await self._finish_answer((reply.get("content") or "").strip(), "no_results")
            return
        if self.output == "evidence":
            self.stopped = "evidence_complete"
            await self._references()
            order = [hit_handle(tc, c) for c in contexts]
            self.evidence = self._ranked_evidence(order)[:len(order)]
            return
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        messages = [{"role": "system", "content": await self._system("quick", [])},
                    {"role": "user", "content": render("agentic_search", "quick_context", query=self.query,
                                                       contexts=rendered, current_time=now, language=language)}]
        reply = await self._call(messages, self.preset.temperature, max_tokens=self._answer_tokens())
        await self._finish_answer((reply.get("content") or "").strip())


    # -- auto: route → gather → merge ------------------------------------------------------------
    async def _auto(self) -> None:
        """The `auto` preset: one routing call picks the sources, they run in parallel, one call merges."""
        from kb_runtime.agentic.tools.knowledge import hit_handle as _hit_handle

        tc = self.tool_context()
        kbs = self.scope.kbs.values()
        kinds = {kb.type for kb in kbs}
        wiki_kbs = [kb.id for kb in kbs if (kb.indexing_strategy or {}).get("wiki_enabled")]
        available = []
        if kinds & {"document", "faq"} or not kinds:
            available.append("documents")
        if "faq" in kinds:
            available.append("faq")
        if wiki_kbs and _feature("wiki"):
            available.append("wiki")
        if "records" in kinds and _feature("records"):
            available.append("records")
        if _feature("memory"):
            available.append("memory")
        route = await self._route(available)
        await self.emit({"response_type": "thinking", "content": f"route: {', '.join(route['sources']) or 'none'} — "
                                                                 f"{route.get('reason', '')}"})
        gathered: dict[str, dict] = {}

        async def run(name, coro):
            call_id = f"call_route_{name}"
            await self.emit({"response_type": "tool_call", "data": {"tool_call_id": call_id, "tool_name": f"auto:{name}",
                                                                   "arguments": {"queries": route["queries"]}}})
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(coro, timeout=self.settings.tool_timeout)
                status = "ok"
            except (ToolError, NotFound, Invalid, ModelError, asyncio.TimeoutError) as exc:
                result, status = {"error": str(exc) or type(exc).__name__, "items": []}, "error"
            duration = int((time.monotonic() - started) * 1000)
            gathered[name] = result
            summary = result.get("summary") or f"{len(result.get('items') or [])} item(s)"
            await self.emit({"response_type": "tool_result", "data": {"tool_call_id": call_id, "tool_name": f"auto:{name}",
                                                                     "success": status == "ok", "output_preview": summary,
                                                                     "duration_ms": duration}})
            return {"id": call_id, "name": f"auto:{name}", "args": {"queries": route["queries"]},
                    "result_summary": summary[:300], "duration_ms": duration, "status": status}

        jobs = []
        if "documents" in route["sources"] or "faq" in route["sources"]:
            jobs.append(run("documents", self._gather_documents(tc, route["queries"])))
        if "wiki" in route["sources"]:
            jobs.append(run("wiki", self._gather_wiki(tc, route["queries"], wiki_kbs, route.get("global"))))
        if "records" in route["sources"]:
            jobs.append(run("records", self._gather_records(route.get("records_question") or self.query)))
        if "memory" in route["sources"]:
            jobs.append(run("memory", self._gather_memory(route["queries"])))
        records = await asyncio.gather(*jobs)
        self.steps.append({"iteration": 1, "thought": route.get("reason", ""), "tool_calls": list(records),
                           "route": {k: route[k] for k in ("sources", "queries", "global")}})
        self.route = {"sources": route["sources"], "queries": route["queries"], "global": bool(route.get("global")),
                      "reason": route.get("reason", ""), "available": available,
                      "counts": {k: len(v.get("items") or []) for k, v in gathered.items()}}
        has_evidence = any(v.get("items") or v.get("text") for v in gathered.values())
        if self.output == "evidence":
            self.stopped = "evidence_complete" if has_evidence else "no_results"
            await self._references()
            order = [h for v in gathered.values() for h in v.get("handles") or []]
            self.evidence = self._ranked_evidence(order)[:max(len(order), 0)] if order else []
            return
        language = language_of(self.query)
        if not has_evidence:
            if self.settings.fallback_strategy == "fixed":
                await self._finish_answer(self.settings.fallback_response, "no_results")
                return
            reply = await self._call([{"role": "system", "content": render("agentic_search", "fallback", language=language)},
                                      {"role": "user", "content": self.query}], 0.3, max_tokens=self._answer_tokens())
            await self._finish_answer((reply.get("content") or "").strip(), "no_results")
            return
        blocks = [v["text"] for k in ("documents", "wiki", "records", "memory") if (v := gathered.get(k)) and v.get("text")]
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        messages = [{"role": "system", "content": await self._system("auto", [])},
                    {"role": "user", "content": render("agentic_search", "quick_context", query=self.query,
                                                       contexts="\n\n".join(blocks), current_time=now, language=language)}]
        reply = await self._call(messages, self.preset.temperature, max_tokens=self._answer_tokens())
        await self._finish_answer((reply.get("content") or "").strip())


    async def _route(self, available: list[str]) -> dict:
        fallback = {"sources": [s for s in available if s != "memory"] or available, "queries": [self.query],
                    "global": False, "records_question": "", "reason": "default route"}
        if not available:
            return {**fallback, "sources": []}
        listing = "\n".join(f"- {AUTO_SOURCES[s]}" for s in available)
        kb_lines = "\n".join(f"- {kb.name} ({kb.type}): {kb.description[:200]}" for kb in self.scope.kbs.values())
        try:
            reply = await self._call([
                {"role": "system", "content": render("agentic_search", "route_system", sources=listing)},
                {"role": "user", "content": render("agentic_search", "route_user", query=self.query,
                                                   knowledge_bases=kb_lines or "(none)")}], 0.0, max_tokens=300)
            data = _json_object(reply.get("content") or "")
        except (ValueError, ModelError):
            return fallback
        sources = [s for s in data.get("sources") or [] if s in available]
        queries = [str(q).strip() for q in data.get("queries") or [] if str(q).strip()][:3]
        return {"sources": list(dict.fromkeys(sources)) or fallback["sources"], "queries": queries or [self.query],
                "global": bool(data.get("global")), "records_question": str(data.get("records_question") or ""),
                "reason": str(data.get("reason") or "")[:300]}


    async def _gather_documents(self, tc, queries: list[str]) -> dict:
        from kb_runtime.agentic.tools.knowledge import hit_handle

        kb_ids = [kb.id for kb in self.scope.kbs.values() if kb.type in ("document", "faq")]
        if not kb_ids:
            return {"items": []}
        # wiki passages come from the wiki part; in a wiki-enabled KB they can fill a short list,
        # so search wide, drop them, then cut
        hits = await run_searches(tc, list(dict.fromkeys([self.query, *queries]))[:4], kb_ids,
                                  list(self.scope.knowledge_ids), "hybrid", 30)
        hits = [h for h in hits if h.get("chunk_type") != "wiki_page"][:10]
        text = self._contexts_text(tc, hits) if hits else ""
        return {"items": hits, "handles": [hit_handle(tc, h) for h in hits], "text": text,
                "summary": f"{len(hits)} passage(s)"}


    async def _gather_wiki(self, tc, queries: list[str], kb_ids: list[str], global_question: bool) -> dict:
        from kb_runtime.agentic.tools.wiki import page_handle

        services = self.service.services
        pages, seen = [], set()
        async with async_session_scope(commit=False) as session:
            for q in list(dict.fromkeys([self.query, *queries]))[:4]:
                words = [w for w in re.findall(r"\w{3,}", q) if w.lower() not in _STOPWORDS][:6]
                for term in words or [q]:
                    for page in await services.wiki.search_pages(session, kb_ids, term, regex=False, limit=5):
                        if page.id not in seen and page.page_type not in ("index",):
                            seen.add(page.id)
                            session.expunge(page)
                            pages.append(page)
        # rank: pages whose title or aliases appear in the question first, then by how many terms hit them
        question = self.query.lower()
        pages.sort(key=lambda p: (-(p.title.lower() in question), -len(p.in_links or []), p.title))
        pages = pages[:6]
        handles, parts = [], []
        for page in pages:
            handle = page_handle(tc, page)
            handles.append(handle)
            body = re.sub(r"\[src:\d+\]", "", page.content or "")[:1500]
            parts.append(f'<page id="{handle}" title="{citations._attr(page.title)}" type="{page.page_type}">\n'
                         f'{page.summary}\n{body}\n</page>')
        global_text = ""
        if global_question:
            for kb_id in kb_ids:
                try:
                    result = await services.wiki_pipeline.global_search(self.auth, kb_id, self.query, 0)
                except (Invalid, NotFound):
                    continue
                if result.get("answer"):
                    global_text += f'<global_summary kb="{citations._attr(self.scope.kbs[kb_id].name)}">\n' \
                                   f'{result["answer"]}\n</global_summary>\n'
        text = ""
        if parts or global_text:
            text = '<source type="wiki">\n' + global_text + "\n".join(parts) + "\n</source>"
        return {"items": pages, "handles": handles, "text": text,
                "summary": f"{len(pages)} page(s)" + (" + global summary" if global_text else "")}


    async def _gather_records(self, question: str) -> dict:
        """A short records-only tool loop: the model writes the queries; numbers come from tool output."""
        from kb_runtime.agentic.tools import records as record_tools

        tc = self.tool_context()
        tools = {t.name: t for t in record_tools.TOOLS + self.service.plugin_tools()}
        schemas = [t.schema() for t in tools.values()]
        # the type descriptions up front: small models skip describe_record_types and invent names
        try:
            types = await record_tools.describe_record_types(tc, {})
        except ToolError:
            types = ""
        messages = [{"role": "system", "content": render("agentic_search", "records_subloop") + "\n\n"
                     + render("agentic_search", "protocol") + (f"\n\nRecord types in scope:\n{types}" if types else "")},
                    {"role": "user", "content": question}]
        before = {h.handle for h in self.handles.citable()}
        text = ""
        outputs: list[str] = []                      # what the records tools actually returned
        nudged = False
        for iteration in range(1, RECORDS_SUBLOOP_ITERATIONS + 1):
            reply = await self._call(messages, 0.2, tools=schemas if iteration < RECORDS_SUBLOOP_ITERATIONS else None)
            calls = reply.get("tool_calls") or []
            if not calls:
                text = (reply.get("content") or "").strip()
                if not outputs and not nudged and iteration < RECORDS_SUBLOOP_ITERATIONS - 1 and len(messages) > 2:
                    # the calls so far failed: one nudge to fix them instead of answering without data
                    nudged = True
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": render("agentic_search", "records_retry")}]
                    continue
                break
            for index, call in enumerate(calls):
                call["id"] = f"call_records_{iteration}_{index}"
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or "{}"}}
                for c in calls]})
            results = await asyncio.gather(*(self._run_tool(tc, tools, c) for c in calls))
            for (record, output), call in zip(results, calls):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                if record["status"] == "ok" and call["name"] != "describe_record_types":
                    outputs.append(output)
        handles = [h.handle for h in self.handles.citable() if h.prefix == "r" and h.handle not in before]
        tables = [o for o in outputs if "| " in o or "<record " in o]
        if not text or not tables:
            # findings that no tool output backs are the model's own numbers: never pass them on
            return {"items": [], "handles": handles,
                    "summary": "no findings" if not text else "discarded: no records tool returned data"}
        text = text + "\n\nTool output (the numbers to use):\n" + "\n\n".join(tables)[:4000]
        return {"items": handles or [text], "handles": handles, "text": f'<source type="records">\n{text}\n</source>',
                "summary": f"{len(handles)} record(s): {text[:240]}"}


    async def _gather_memory(self, queries: list[str]) -> dict:
        facts, seen = [], set()
        for q in list(dict.fromkeys([self.query, *queries]))[:3]:
            for fact in await self.service.services.memory.search(self.auth, {"query": q, "limit": 6}):
                if fact["id"] not in seen:
                    seen.add(fact["id"])
                    facts.append(fact)
        if not facts:
            return {"items": []}
        lines = "\n".join(f'<fact valid="{(f["valid_at"] or "?")[:10]}">{citations._attr(f["fact"])}</fact>' for f in facts[:10])
        return {"items": facts, "handles": [], "text": f'<source type="memory" note="the user\'s own memory">\n{lines}\n</source>',
                "summary": f"{len(facts)} fact(s)"}


# --- the auto preset's sources ------------------------------------------------------------------------
AUTO_SOURCES = {
    "documents": "documents: hybrid search over document passages (and FAQ entries)",
    "faq": "faq: curated question-answer entries",
    "wiki": "wiki: pages compiled from the documents (overviews of entities and concepts)",
    "records": "records: structured records (runs, jobs, measurements): filters, aggregates, numbers",
    "memory": "memory: the user's own long-term memory (facts about people, decisions, history)",
}
RECORDS_SUBLOOP_ITERATIONS = 4




def _feature(name: str) -> bool:
    from kb_runtime.features import enabled

    return enabled(name)


class AgenticService:
    def __init__(self, services, settings: Settings | None = None):
        self.services = services
        self.settings = settings or Settings.from_core_config()
        self._running: dict[str, int] = {}

    def plugin_tools(self) -> list:
        from kb_runtime.agentic.tools.records import operation_tool

        records = self.services.records
        if records is None:
            return []
        return [operation_tool(qualified, op) for qualified, op in records.read_only_operations()]

    def presets(self) -> list[dict]:
        return describe(self.plugin_tools())

    async def _model(self, ctx: AuthContext, ref: str | None, preset: Preset):
        models = self.services.models
        if ref:
            model_id = await models.usable(ctx, ref, "KnowledgeQA")
        else:
            model_id = self.settings.model or await models.default_id("KnowledgeQA")
        if not model_id:
            raise Invalid("no chat model is configured for agentic search ([agentic] model)")
        spec = await models.spec(model_id)
        if preset.engine == "react" and not spec.supports_tools:
            raise Invalid(f"model {spec.name} does not support tool calling (supports_tools); use preset=quick "
                          "or another model")
        return spec.id, await models.chat(spec.id), spec.context_window

    async def _scope(self, ctx: AuthContext, body: dict) -> Scope:
        refs = [str(r) for r in body.get("knowledge_base_ids") or [] if r]
        async with async_session_scope(commit=False) as session:
            kb_ids = await self.services.knowledge_bases.readable_ids(session, ctx, refs or None)
            rows = (await session.execute(sa.select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids)))).scalars().all() \
                if kb_ids else []
            kbs = {}
            for row in rows:
                session.expunge(row)
                kbs[row.id] = row
            knowledge_ids = [str(k) for k in body.get("knowledge_ids") or [] if k]
            for knowledge_id in knowledge_ids:
                item = await session.get(Knowledge, knowledge_id)
                if (item is None or item.deleted_at is not None or item.knowledge_base_id not in kbs
                        or not access.can_read_item(ctx, item)):
                    raise NotFound(f"knowledge {knowledge_id} not found")
        ordered = [kb_id for kb_id in kb_ids if kb_id in kbs]
        return Scope(kb_ids=ordered, kbs=kbs, knowledge_ids=knowledge_ids,
                     tag_ids=[str(t) for t in body.get("tag_ids") or [] if t])

    async def prepare(self, ctx: AuthContext, body: dict) -> Run:
        query = str(body.get("query") or "").strip()
        if not query:
            raise Invalid("query is required")
        preset = PRESETS.get(body.get("preset") or "deep")
        if preset is None:
            raise Invalid(f"preset must be one of {', '.join(PRESETS)}")
        from kb_runtime.agentic.presets import available

        if not available(preset):
            raise Invalid(f"the {preset.name} preset is not available in this deployment")
        output = body.get("output") or "answer"
        if output not in ("answer", "evidence"):
            raise Invalid("output must be answer or evidence")
        limit = self.settings.max_iterations_limit
        try:
            iterations = int(body.get("max_iterations") or self.settings.max_iterations)
        except (TypeError, ValueError):
            raise Invalid("max_iterations must be an integer")
        if not 1 <= iterations <= limit:
            raise Invalid(f"max_iterations must be between 1 and {limit}")
        scope = await self._scope(ctx, body)
        allow_writes = bool(body.get("allow_writes"))
        if allow_writes and "knowledge:write" not in ctx.scopes:
            raise Forbidden("allow_writes needs the knowledge:write scope")
        model_id, chat, window = await self._model(ctx, body.get("model_id"), preset)
        owner = ctx.subject
        if self._running.get(owner, 0) >= self.settings.per_owner:
            raise Busy(f"at most {self.settings.per_owner} agentic searches run at once per user")
        self._running[owner] = self._running.get(owner, 0) + 1
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                self._running[owner] -= 1
                if self._running[owner] <= 0:
                    self._running.pop(owner, None)

        from kb_runtime.websearch.fetch import FetchCache

        return Run(self, ctx, query, preset, output, model_id, chat, window, scope, iterations, self.settings, release,
                   allow_writes=allow_writes, web_cache=FetchCache())

