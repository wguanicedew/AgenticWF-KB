"""
Citations (docs/agentic-search.md, Citations): `<ref id="cN"/>` → `<kb …/>`, `rN` →
`<record …/>`, `pN` → `<wiki …/>`; refs that are not a citable handle of this run are dropped.
Evidence lines `<evidence id="cN">why</evidence>` are parsed the same way.
"""

from __future__ import annotations

import html
import re

from kb_runtime.agentic.handles import CITABLE, Handles

_REF = re.compile(r"""<ref\s+id\s*=\s*["']?([A-Za-z]\d+)["']?\s*/?>(?:\s*</ref>)?""")  # cN rN pN wN
_EVIDENCE = re.compile(r"""<evidence\s+id\s*=\s*["']?([A-Za-z]\d+)["']?\s*>(.*?)</evidence>""", re.DOTALL)
_STRAY = re.compile(r"</?(?:kb|record|wiki|web)\b[^>]*>")
PIECE = 160


def _attr(value) -> str:
    return html.escape(str(value or ""), quote=True)


def tag_for(handle, n: int | None = None) -> str:
    """The expanded citation; `n` is its number in the answer's sources list, `source` the
    source's citation text (and `url` when it has one)."""
    p = handle.payload
    extra = (f' n="{n}"' if n else "") + (f' source="{_attr(p["citation_text"])}"' if p.get("citation_text") else "") \
        + (f' url="{_attr(p["url"])}"' if p.get("url") else "")
    if handle.prefix == "c":
        return (f'<kb doc="{_attr(p.get("knowledge_title"))}" chunk_id="{_attr(handle.key)}" '
                f'kb_id="{_attr(p.get("knowledge_base_id"))}"{extra} />')
    if handle.prefix == "w":
        return f'<web title="{_attr(p.get("title"))}"{extra} />'
    if handle.prefix == "r":
        return (f'<record id="{_attr(handle.key)}" type="{_attr(p.get("record_type"))}" '
                f'name="{_attr(p.get("name"))}" kb_id="{_attr(p.get("knowledge_base_id"))}"{extra} />')
    return (f'<wiki slug="{_attr(p.get("slug"))}" title="{_attr(p.get("title"))}" '
            f'kb_id="{_attr(p.get("knowledge_base_id"))}"{extra} />')


def sources_footer(sources: list[dict]) -> str:
    """The answer's "Sources" list: one numbered line per source."""
    if not sources:
        return "\n\nSources: none — nothing relevant was found in the knowledge bases."
    lines = []
    for s in sources:
        lines.append(f"[{s['n']}] {s['citation_text']}" + ("" if s["cited"] else " (consulted, not cited inline)"))
    return "\n\nSources:\n" + "\n".join(lines)


def expand(text: str, handles: Handles) -> tuple[str, list[str]]:
    """The text with refs expanded, and the handles cited (in order of first citation)."""
    cited: list[str] = []
    text = _STRAY.sub("", text or "")              # the model must not write the expanded tags

    def replace(match):
        handle = handles.lookup(match.group(1))
        if handle is None or handle.prefix not in CITABLE:
            return ""
        if handle.handle not in cited:
            cited.append(handle.handle)
        return tag_for(handle, cited.index(handle.handle) + 1)

    return _REF.sub(replace, text), cited


def parse_evidence(text: str, handles: Handles) -> list[tuple[str, str]]:
    """[(handle, reason)] for valid citable handles, in the model's order, without repeats."""
    out, seen = [], set()
    for handle_text, reason in _EVIDENCE.findall(text or ""):
        handle = handles.lookup(handle_text)
        if handle is None or handle.prefix not in CITABLE or handle.handle in seen:
            continue
        seen.add(handle.handle)
        out.append((handle.handle, reason.strip()))
    return out


def pieces(text: str, size: int = PIECE) -> list[str]:
    """Split an answer into ~size-character pieces, never inside a tag."""
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        open_at = text.rfind("<", start, end)
        if open_at > start and text.find(">", open_at, end) == -1:
            end = open_at                                     # do not cut an open tag
        elif open_at == start and text.find(">", start, end) == -1:
            close = text.find(">", start)
            end = len(text) if close == -1 else close + 1     # a tag longer than a piece
        out.append(text[start:end])
        start = end
    return out
