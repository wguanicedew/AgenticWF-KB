"""Memory read tools (docs/agentic-search.md, Engine): recall and memory_history, over the
caller's own temporal memory only. Offered when `[features] memory` is on."""

from __future__ import annotations

from kb_runtime.agentic.tools import Tool, ToolContext, ToolError, attr, int_arg
from kb_runtime.errors import Invalid, NotFound


def _line(f: dict) -> str:
    span = f"{(f['valid_at'] or '?')[:10]} → {(f['invalid_at'] or 'now')[:10]}"
    return (f'<fact source="{attr(f["source"])}" relation="{f["relation"]}" target="{attr(f["target"])}" '
            f'valid="{span}"{" invalid=\"true\"" if f["invalid_at"] else ""}>{attr(f["fact"])}</fact>')


async def recall(tc: ToolContext, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("query is required")
    body = {"query": query, "limit": int_arg(args, "limit", 10, 1, 30)}
    for key in ("as_of", "focal_entity", "group_id"):
        if args.get(key):
            body[key] = args[key]
    try:
        facts = await tc.services.memory.search(tc.auth, body)
    except (Invalid, NotFound) as exc:
        raise ToolError(str(exc))
    if not facts:
        return "Nothing in your memory matches."
    return tc.clip("<memory>\n" + "\n".join(_line(f) for f in facts) + "\n</memory>")


async def memory_history(tc: ToolContext, args: dict) -> str:
    entity = str(args.get("entity") or "").strip()
    if not entity:
        raise ToolError("entity is required")
    try:
        facts = await tc.services.memory.facts(tc.auth, entity=entity, include_invalid=True,
                                               group_id=args.get("group_id"))
    except (Invalid, NotFound) as exc:
        raise ToolError(str(exc))
    if not facts:
        return f"No facts about {entity}."
    facts.sort(key=lambda f: (f["valid_at"] or "", f["created_at"] or ""))
    return tc.clip(f'<history entity="{attr(entity)}">\n' + "\n".join(_line(f) for f in facts) + "\n</history>")


TOOLS = [
    Tool("recall", "Search the user's own long-term memory (facts with validity times). Use as_of for the state at a "
         "past time.", {"type": "object", "properties": {
             "query": {"type": "string"}, "as_of": {"type": "string", "description": "ISO time"},
             "focal_entity": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}},
             "required": ["query"]}, recall, group="memory"),
    Tool("memory_history", "Every fact about one entity in the user's memory, including facts that stopped being true.",
         {"type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"]},
         memory_history, group="memory"),
]
