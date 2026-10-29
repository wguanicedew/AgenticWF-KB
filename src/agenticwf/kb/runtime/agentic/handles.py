"""
Request-local evidence handles (docs/agentic-search.md, Evidence handles).

Tools never show raw ids to the model: `bN` a knowledge base, `dN` a document, `cN` a chunk,
`rN` a record, `pN` a wiki page, `wN` a web page, `mN` a fact from the caller's memory. The same object always gets the same handle within a run;
handles are case-insensitive and mean nothing outside the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agenticwf.kb.runtime.agentic.tools import ToolError

KINDS = {"b": "knowledge_base", "d": "document", "c": "chunk", "r": "record", "p": "wiki_page", "w": "web_page",
         "m": "memory_fact"}
CITABLE = ("c", "r", "p", "w", "m")
KIND_NAMES = {"c": "chunk", "r": "record", "p": "wiki_page", "w": "web_page", "m": "memory_fact"}
# Where a cited source came from, as the answer's Sources list names it.
ORIGINS = {"c": "documents", "r": "records", "p": "wiki", "w": "web", "m": "memory"}
_HANDLE = re.compile(r"^\s*([bdcrpwm])(\d+)\s*$", re.IGNORECASE)


@dataclass
class Handle:
    handle: str
    prefix: str
    key: str                 # the real id
    payload: dict
    order: int


@dataclass
class Handles:
    by_handle: dict[str, Handle] = field(default_factory=dict)
    by_key: dict[tuple[str, str], Handle] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def issue(self, prefix: str, key: str, payload: dict | None = None) -> str:
        existing = self.by_key.get((prefix, key))
        if existing is not None:
            if payload:
                # keep the best score seen, refresh the rest
                best = max(existing.payload.get("score") or 0, payload.get("score") or 0)
                existing.payload.update(payload)
                if "score" in existing.payload or "score" in payload:
                    existing.payload["score"] = best
            return existing.handle
        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        handle = Handle(f"{prefix}{number}", prefix, key, dict(payload or {}), len(self.by_handle))
        self.by_handle[handle.handle] = handle
        self.by_key[(prefix, key)] = handle
        return handle.handle

    def get(self, text: str, prefixes: str | tuple = tuple(KINDS)) -> Handle:
        match = _HANDLE.match(str(text or ""))
        if not match:
            raise ToolError(f"{text!r} is not a handle from this search")
        prefix = match.group(1).lower()
        handle = self.by_handle.get(f"{prefix}{match.group(2)}")
        if handle is None:
            raise ToolError(f"unknown handle {text!r}: use handles returned by the tools")
        if prefix not in prefixes:
            wanted = ", ".join(f"{p}N" for p in prefixes)
            raise ToolError(f"{handle.handle} is a {KINDS[prefix]}; this argument takes {wanted}")
        return handle

    def key(self, text: str, prefixes) -> str:
        return self.get(text, prefixes).key

    def citable(self) -> list[Handle]:
        return [h for h in self.by_handle.values() if h.prefix in CITABLE]

    def lookup(self, handle_text: str) -> Handle | None:
        match = _HANDLE.match(str(handle_text or ""))
        return self.by_handle.get(f"{match.group(1).lower()}{match.group(2)}") if match else None
