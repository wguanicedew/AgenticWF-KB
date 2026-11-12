"""
Contextual chunking (docs/ingestion.md, Contextual chunking): before a document's chunks are
embedded, a chat model writes one or two sentences per chunk that place it in the document (which
system, section, version, what "it" refers to), and that text goes into what is indexed (vector
and BM25), not into the chunk's content. A chunk like "it must stay below 1.2 bar" then matches
"cryostat pressure limit". The idea is Anthropic's "contextual retrieval".

One call per window of WINDOW chunks: the document's start (DOC_CHARS) comes first in every call,
so a provider can cache it, then the stretch of the document around the window, then the
numbered chunks. Windows run CONCURRENCY at a time. A window whose call fails or whose answer
does not give one context per chunk leaves those chunks without context: never a failed document.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Protocol

from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.runtime.prompts import render

LOG = get_logger(__name__)

WINDOW = 12            # chunks per call
DOC_CHARS = 6000       # the document's start, in every call (cacheable prefix)
AROUND_CHARS = 1500    # of the document before the window and after it
PASSAGE_CHARS = 1500   # of each chunk
CONTEXT_CHARS = 400    # of each context kept
CONCURRENCY = 3
_THINK = re.compile(r"<think>.*?</think>", re.S)


class Chat(Protocol):
    async def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float = 0.3,
                       **kw) -> dict: ...


def parse_contexts(text: str, n: int) -> list[str] | None:
    """`{"contexts": [...]}` with exactly `n` strings, else None."""
    text = _THINK.sub("", text or "")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        values = json.loads(match.group(0)).get("contexts")
    except (ValueError, AttributeError):
        return None
    if not isinstance(values, list) or len(values) != n:
        return None
    return [" ".join(str(v).split())[:CONTEXT_CHARS] for v in values]


async def chunk_contexts(chat: Chat, title: str, document: str, spans: list[tuple[int, int, str]]) -> list[str]:
    """One context per chunk (`spans`: (start, end, content) into `document`), "" where none was written."""
    out = [""] * len(spans)
    if not spans:
        return out
    head = document[:DOC_CHARS] + (" […]" if len(document) > DOC_CHARS else "")
    gate = asyncio.Semaphore(CONCURRENCY)

    async def window(first: int) -> None:
        part = spans[first:first + WINDOW]
        lo = max(0, part[0][0] - AROUND_CHARS)
        hi = min(len(document), part[-1][1] + AROUND_CHARS)
        excerpt = "" if hi <= DOC_CHARS else document[lo:hi]       # the start is in <document> already
        passages = "\n\n".join(f"<passage n=\"{i + 1}\">\n{content[:PASSAGE_CHARS]}\n</passage>"
                               for i, (_, _, content) in enumerate(part))
        messages = [{"role": "system", "content": render("enrichment", "context_system")},
                    {"role": "user", "content": render("enrichment", "context_user", title=title, document=head,
                                                       excerpt=excerpt or "(the document's start, above)",
                                                       passages=passages, count=len(part))}]
        async with gate:
            try:
                reply = await chat.complete(messages, max_tokens=120 * len(part) + 200, temperature=0.2)
            except Exception as exc:   # noqa: BLE001 - a model error leaves these chunks without context
                LOG.warning("contextual chunking: window at chunk %d failed: %s", first, exc)
                return
        contexts = parse_contexts(reply.get("content") or "", len(part))
        if contexts is None:
            LOG.info("contextual chunking: window at chunk %d did not parse", first)
            return
        out[first:first + len(part)] = contexts

    await asyncio.gather(*(window(i) for i in range(0, len(spans), WINDOW)))
    return out
