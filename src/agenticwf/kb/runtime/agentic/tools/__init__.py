"""
Agentic-search tools (docs/agentic-search.md, Knowledge tools, Planning tools).

A Tool is a name, a description, JSON-schema parameters and an async `run(tc, args) -> str`.
Tools are read-only and run under the request's principal and scope (ToolContext); a ToolError
becomes an `Error: …` message the model sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from agenticwf.auth.context import AuthContext

    from agenticwf.kb.runtime.agentic.handles import Handles
    from agenticwf.kb.runtime.services import KBServices


class ToolError(Exception):
    """A tool failure the model can act on (wrong handle, bad argument, timeout)."""


@dataclass
class Scope:
    """What the request allows; every tool applies it, and the model cannot widen it."""

    kb_ids: list[str]
    kbs: dict[str, Any]                       # id -> KnowledgeBase (detached)
    knowledge_ids: list[str] = field(default_factory=list)
    tag_ids: list[str] = field(default_factory=list)


@dataclass
class ToolContext:
    auth: "AuthContext"
    services: "KBServices"
    scope: Scope
    handles: "Handles"
    emit: Callable[[dict], Awaitable[None]]
    max_output_chars: int = 16_000
    todos: list[dict] = field(default_factory=list)
    web_cache: Any = None                     # the run's FetchCache: paged reads of a page see one version
    writes: list = field(default_factory=list)  # what write tools changed, reported with the run

    def clip(self, text: str) -> str:
        limit = self.max_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n[output truncated at {limit} characters; narrow the request or page with offset]"

    def kb_ids(self, handles: list[str] | None) -> list[str]:
        """KB handles → ids, each inside the scope."""
        if not handles:
            return list(self.scope.kb_ids)
        ids = []
        for text in handles:
            kb_id = self.handles.key(text, ("b",))
            if kb_id not in self.scope.kb_ids:
                raise ToolError(f"{text} is outside this search's scope")
            ids.append(kb_id)
        return ids

    def knowledge_ids(self, handles: list[str] | None) -> list[str]:
        """Document handles → ids, intersected with the request's knowledge ids."""
        ids = [self.handles.key(text, ("d",)) for text in handles or []]
        if self.scope.knowledge_ids:
            if ids:
                outside = [i for i in ids if i not in self.scope.knowledge_ids]
                if outside:
                    raise ToolError("a document is outside this search's scope")
                return ids
            return list(self.scope.knowledge_ids)
        return ids


Runner = Callable[[ToolContext, dict], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Runner
    group: str = "knowledge"                  # knowledge | planning | records | wiki

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def attr(value) -> str:
    import html

    return html.escape(str(value if value is not None else ""), quote=True)


def int_arg(args: dict, name: str, default: int, low: int, high: int) -> int:
    value = args.get(name, default)
    try:
        value = int(value if value is not None else default)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be an integer")
    return max(low, min(high, value))


def list_arg(args: dict, name: str) -> list[str]:
    value = args.get(name)
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if not isinstance(value, list):
        raise ToolError(f"{name} must be a list")
    return [str(v) for v in value if v not in (None, "")]
