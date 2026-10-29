"""
The chunker (docs/ingestion.md, Chunker): normalised markdown in, chunks with exact character
offsets out.

    chunk(text, config) -> list[Piece]

Strategies: `legacy` (recursive splitter), `heading` (markdown sections, breadcrumb in
`context_header`), `heuristic` (numbered / Chapter / 第N章 titles when there are no markdown
headings), `auto` (picks one of them). With `enable_parent_child`, parents of
`parent_chunk_size` are split into children of `child_chunk_size`; only children are indexed.

Offsets are code points into `text`. A chunk's content is `text[start:end]`, except that a table
continued from an earlier chunk gets its header rows prepended (so the chunk is a valid table);
`content_offset` says how many characters were prepended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULTS = {"chunk_size": 512, "chunk_overlap": 50, "separators": ["\n\n", "\n", "。"], "strategy": "auto",
            "token_limit": 0, "enable_parent_child": False, "parent_chunk_size": 4096, "child_chunk_size": 384}
HARD_CAP = 7500
STRATEGIES = ("legacy", "heading", "heuristic", "auto")


@dataclass
class Piece:
    content: str
    start: int
    end: int
    context_header: str = ""
    chunk_type: str = "text"            # text | parent_text
    parent: int | None = None           # index of the parent piece (parent-child mode)
    content_offset: int = 0             # characters prepended to text[start:end] (table header)


@dataclass
class _Unit:
    start: int
    end: int
    protected: bool = False
    table: int | None = None            # table number, for table units
    header: bool = False                # the table's header + separator rows


# --- protected spans ----------------------------------------------------------------------------
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.M | re.S)
_MATH = re.compile(r"\$\$.+?\$\$", re.S)
_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_LINK = re.compile(r"\[[^\]\n]*\]\([^)\n]*\)")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.M)
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")


def _protected_spans(text: str) -> list[_Unit]:
    spans: list[_Unit] = []
    taken = [False] * (len(text) + 1)

    def add(start, end, **kw):
        if start >= end or any(taken[start:end]):
            return
        for i in range(start, end):
            taken[i] = True
        spans.append(_Unit(start, end, protected=True, **kw))

    for m in _FENCE.finditer(text):
        add(m.start(), m.end())
    for m in _MATH.finditer(text):
        add(m.start(), m.end())
    # tables: a header row followed by a separator row starts a table; following rows belong to it
    table_no, i = 0, 0
    line_spans = []
    pos = 0
    for line in text.split("\n"):
        line_spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    while i < len(line_spans):
        s, e = line_spans[i]
        line = text[s:e]
        if (_TABLE_ROW.fullmatch(line) and i + 1 < len(line_spans)
                and _TABLE_SEP.fullmatch(text[line_spans[i + 1][0]:line_spans[i + 1][1]])):
            table_no += 1
            add(s, line_spans[i + 1][1], table=table_no, header=True)
            i += 2
            while i < len(line_spans) and _TABLE_ROW.fullmatch(text[line_spans[i][0]:line_spans[i][1]]):
                add(line_spans[i][0], line_spans[i][1], table=table_no)
                i += 1
            continue
        i += 1
    for pattern in (_IMAGE, _LINK, _INLINE_CODE):
        for m in pattern.finditer(text):
            add(m.start(), m.end())
    return sorted(spans, key=lambda u: u.start)


# --- splitting into units ---------------------------------------------------------------------------
def _split_free(text: str, start: int, end: int, size: int, separators: list[str]) -> list[_Unit]:
    """Split text[start:end] (no protected spans inside) into units no longer than `size` where
    possible; each separator is a unit of its own."""
    if end - start <= size:
        return [_Unit(start, end)] if end > start else []
    for index, sep in enumerate(separators):
        if sep and sep in text[start:end]:
            units: list[_Unit] = []
            pos = start
            while pos < end:
                hit = text.find(sep, pos, end)
                if hit < 0:
                    units.extend(_split_free(text, pos, end, size, separators[index + 1:]))
                    break
                units.extend(_split_free(text, pos, hit, size, separators[index + 1:]))
                units.append(_Unit(hit, hit + len(sep)))
                pos = hit + len(sep)
            return units
    # no separator left: cut at spaces where possible
    units, pos = [], start
    while pos < end:
        cut = min(end, pos + size)
        if cut < end:
            space = text.rfind(" ", pos + size // 2, cut)
            cut = space + 1 if space > pos else cut
        units.append(_Unit(pos, cut))
        pos = cut
    return units


def _hard_cap(text: str, unit: _Unit) -> list[_Unit]:
    if unit.end - unit.start <= HARD_CAP:
        return [unit]
    parts, pos = [], unit.start
    while pos < unit.end:
        cut = min(unit.end, pos + HARD_CAP)
        if cut < unit.end:
            nl = max(text.rfind("\n", pos + 1, cut), text.rfind(" ", pos + 1, cut))
            cut = nl + 1 if nl > pos else cut
        parts.append(_Unit(pos, cut, protected=unit.protected, table=unit.table, header=unit.header and pos == unit.start))
        pos = cut
    return parts


def _units(text: str, start: int, end: int, size: int, separators: list[str]) -> list[_Unit]:
    units: list[_Unit] = []
    pos = start
    for span in _protected_spans(text[start:end]):
        span = _Unit(span.start + start, span.end + start, True, span.table, span.header)
        units.extend(_split_free(text, pos, span.start, size, separators))
        units.extend(_hard_cap(text, span))
        pos = span.end
    units.extend(_split_free(text, pos, end, size, separators))
    return units


# --- merging units into chunks ------------------------------------------------------------------------
def _legacy(text: str, start: int, end: int, size: int, overlap: int, separators: list[str],
            header: str = "") -> list[Piece]:
    units = _units(text, start, end, size, separators)
    overlap = max(0, min(overlap, size - 1))
    table_headers: dict[int, str] = {u.table: text[u.start:u.end] for u in units if u.header and u.table}
    pieces: list[Piece] = []
    current: list[_Unit] = []

    def emit(group: list[_Unit]):
        if not group or not text[group[0].start:group[-1].end].strip():
            return
        s, e = group[0].start, group[-1].end
        content = text[s:e]
        first = next((u for u in group if text[u.start:u.end].strip()), group[0])
        prefix = ""
        if first.table and not first.header and table_headers.get(first.table) \
                and not any(u.header and u.table == first.table for u in group):
            prefix = table_headers[first.table] + "\n"
        pieces.append(Piece(prefix + content, s, e, header, content_offset=len(prefix)))

    length = 0
    for unit in units:
        n = unit.end - unit.start
        if current and length + n > size:
            emit(current)
            carried: list[_Unit] = []
            carried_len = 0
            for u in reversed(current):
                ul = u.end - u.start
                if carried_len + ul > overlap or len(carried) + 1 >= len(current):
                    break
                carried.insert(0, u)
                carried_len += ul
            # never start a chunk with a bare separator
            while carried and not text[carried[0].start:carried[0].end].strip():
                carried_len -= carried[0].end - carried[0].start
                carried.pop(0)
            current, length = carried, carried_len
        current.append(unit)
        length += n
    emit(current)
    return pieces


# --- sections ---------------------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.M)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})\.?[ \t]+([A-Z一-鿿][^\n]{0,120})$", re.M)
_WORD_TITLE = re.compile(r"^((?:Chapter|Section|Part)[ \t]+[\dIVXLC]+[^\n]{0,120})$", re.M)
_CJK_TITLE = re.compile(r"^(第[一二三四五六七八九十百千\d]+[章节部分][^\n]{0,80})$", re.M)


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FENCE.finditer(text)]


def _outside(ranges, pos) -> bool:
    return not any(s <= pos < e for s, e in ranges)


def markdown_headings(text: str) -> list[tuple[int, int, str]]:
    """(position, level, title) of markdown headings outside fenced code."""
    fenced = _fenced_ranges(text)
    return [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _MD_HEADING.finditer(text)
            if _outside(fenced, m.start())]


def heuristic_titles(text: str) -> list[tuple[int, int, str]]:
    fenced = _fenced_ranges(text)
    found = []
    for m in _NUMBERED.finditer(text):
        found.append((m.start(), m.group(1).count(".") + 1, m.group(0).strip()))
    for pattern in (_WORD_TITLE, _CJK_TITLE):
        for m in pattern.finditer(text):
            found.append((m.start(), 1, m.group(0).strip()))
    return sorted(t for t in found if _outside(fenced, t[0]))


def _sections(text: str, start: int, end: int, titles: list[tuple[int, int, str]]):
    """(start, end, breadcrumb) per section; text before the first title is a preamble."""
    titles = [t for t in titles if start <= t[0] < end]
    if not titles:
        return [(start, end, "")]
    out = []
    if titles[0][0] > start:
        out.append((start, titles[0][0], ""))
    stack: list[tuple[int, str]] = []
    for index, (pos, level, title) in enumerate(titles):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        stop = titles[index + 1][0] if index + 1 < len(titles) else end
        out.append((pos, stop, " > ".join(t for _, t in stack)))
    return out


def resolve_strategy(text: str, strategy: str) -> str:
    if strategy not in STRATEGIES:
        return "legacy"
    if strategy != "auto":
        return strategy
    headings = markdown_headings(text)
    if len(headings) >= 2 and len(text) / (len(headings) + 1) >= 80:
        return "heading"
    if len(heuristic_titles(text)) >= 2:
        return "heuristic"
    return "legacy"


def _by_strategy(text: str, start: int, end: int, size: int, overlap: int, separators: list[str],
                 strategy: str) -> list[Piece]:
    if strategy == "heading":
        titles = markdown_headings(text)
    elif strategy == "heuristic":
        titles = markdown_headings(text) or heuristic_titles(text)
    else:
        titles = []
    pieces: list[Piece] = []
    for s, e, crumb in _sections(text, start, end, titles):
        pieces.extend(_legacy(text, s, e, size, overlap, separators, crumb))
    return pieces


# --- the entry point ----------------------------------------------------------------------------------
def settings(config: dict | None, default_strategy: str = "auto") -> dict:
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (config or {}).items() if v is not None})
    if not merged.get("strategy"):
        merged["strategy"] = default_strategy
    size = int(merged["chunk_size"])
    if int(merged.get("token_limit") or 0) > 0:
        size = max(50, min(size, int(merged["token_limit"]) * 3))
    merged["chunk_size"] = max(1, size)
    merged["chunk_overlap"] = max(0, min(int(merged["chunk_overlap"]), merged["chunk_size"] - 1))
    merged["separators"] = list(merged.get("separators") or DEFAULTS["separators"])
    return merged


def chunk(text: str, config: dict | None = None, default_strategy: str = "auto") -> list[Piece]:
    cfg = settings(config, default_strategy)
    strategy = resolve_strategy(text, cfg["strategy"])
    separators = cfg["separators"]
    if not cfg.get("enable_parent_child"):
        return _by_strategy(text, 0, len(text), cfg["chunk_size"], cfg["chunk_overlap"], separators, strategy)
    parent_size = int(cfg["parent_chunk_size"])
    child_size = int(cfg["child_chunk_size"])
    pieces: list[Piece] = []
    for parent in _by_strategy(text, 0, len(text), parent_size, 0, separators, strategy):
        parent.chunk_type = "parent_text"
        parent_index = len(pieces)
        pieces.append(parent)
        for child in _legacy(text, parent.start, parent.end, child_size, child_size // 5, separators,
                             parent.context_header):
            child.parent = parent_index
            pieces.append(child)
    return pieces
