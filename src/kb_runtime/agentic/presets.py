"""Presets (docs/agentic-search.md, Presets): quick (one pass, no tools), deep (the ReAct loop
over the knowledge and planning tools); wiki arrives with phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from kb_runtime.agentic.tools import knowledge, memory, planning, records, web, wiki, writes
from kb_runtime.features import enabled as feature_enabled


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    engine: str                 # single_pass | react | router
    temperature: float
    template: str
    tool_groups: tuple[str, ...] = ()
    available: bool = True


PRESETS = {
    "quick": Preset("quick", "One pass: query rewrite, hybrid search with rerank, then an answer or the evidence.",
                    "single_pass", 0.3, "quick"),
    "deep": Preset("deep", "A tool loop: the model searches, greps, lists and reads documents over several steps.",
                   "react", 0.7, "deep", ("knowledge", "records", "wiki", "memory", "planning", "write")),
    "smart": Preset("smart", "Smart reasoning: the tool loop over every part of the knowledge service and the public "
                             "web (web_search, web_fetch), merging what it finds with citations.",
                    "react", 0.7, "smart", ("knowledge", "records", "wiki", "memory", "web", "planning", "write")),
    "wiki": Preset("wiki", "Wiki pages first, then the knowledge tools for exact evidence.",
                   "react", 0.5, "wiki", ("wiki", "knowledge", "records", "memory", "planning", "write")),
    "auto": Preset("auto", "Route the question to the parts that can answer it (documents and FAQ, wiki, records, "
                           "memory), gather from them in parallel, merge into one answer.",
                   "router", 0.3, "auto", ("records",)),
}

ALL_TOOLS = {tool.name: tool for tool in wiki.TOOLS + knowledge.TOOLS + records.TOOLS + memory.TOOLS + web.TOOLS
             + planning.TOOLS + writes.TOOLS}


def available(preset: Preset) -> bool:
    needs = {"wiki": "wiki", "smart": "web"}.get(preset.name)
    return preset.available and (needs is None or feature_enabled(needs))


def tools_for(preset: Preset, has_documents: bool = True, has_records: bool = True, extra: list | None = None,
              has_wiki: bool = True, allow_writes: bool = False) -> list:
    """The preset's tools for what is in scope: document tools without document KBs, records
    tools without records KBs, wiki tools without a wiki, are dropped. `extra`: read-only
    plugin operations (records group)."""
    tools = [t for t in list(ALL_TOOLS.values()) + list(extra or []) if t.group in preset.tool_groups]
    if not has_documents:
        tools = [t for t in tools if t.name not in ("grep_knowledge", "list_documents", "read_document")]
    if not has_records:
        tools = [t for t in tools if t.group != "records"]
    if not has_wiki or not feature_enabled("wiki"):
        tools = [t for t in tools if t.group != "wiki"]
    if not feature_enabled("memory"):
        tools = [t for t in tools if t.group != "memory" and t.name != "remember"]
    if not feature_enabled("web"):
        tools = [t for t in tools if t.group != "web"]
    if not allow_writes:
        tools = [t for t in tools if t.group != "write"]
    elif not has_wiki or not feature_enabled("wiki"):
        tools = [t for t in tools if not (t.group == "write" and t.name.startswith("wiki_"))]
    return tools


def describe(extra: list | None = None) -> list[dict]:
    return [{"name": p.name, "description": p.description, "engine": p.engine, "temperature": p.temperature,
             "available": available(p), "tools": [t.name for t in tools_for(p, extra=extra)],
             "write_tools": [t.name for t in tools_for(p, extra=extra, allow_writes=True) if t.group == "write"]}
            for p in PRESETS.values()]
