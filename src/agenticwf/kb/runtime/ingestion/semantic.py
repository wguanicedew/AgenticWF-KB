"""
Semantic chunking (docs/ingestion.md, Semantic chunking): `chunking_config.strategy = semantic`.

Structure first, as `auto` does: the document is cut into its sections (markdown headings, else
numbered titles), and each section into small units (a sentence or two, MIN_UNIT..MAX_UNIT
characters) with the legacy splitter, at sentence ends too (tables, code, math and links are
still never split). Every unit is embedded with the KB's embedding model; a
chunk ends where the meaning shifts, i.e. where the distance between neighbouring units is in
the document's top (100 - BREAK_PERCENTILE) %, once the chunk has MIN_SHARE of `chunk_size`,
and always before it would grow past `chunk_size`. Chunks never cross a section.

With `enable_parent_child` the parents stay structural (`parent_chunk_size`) and their children
(searched) are cut by meaning, up to `child_chunk_size`.

The unit embeddings are not stored. Embedding fails: the structural chunking (`auto`) instead.
"""

from __future__ import annotations

import math
from typing import Awaitable, Callable

from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.runtime.ingestion import chunker
from agenticwf.kb.runtime.ingestion.chunker import Piece

LOG = get_logger(__name__)

STRATEGY = "semantic"
BREAK_PERCENTILE = 85      # a distance at or above this percentile of the document's is a break
MIN_SHARE = 0.25           # a chunk has at least this share of its size before a break may end it
MIN_UNIT = 80              # units are a sentence or two (MIN_UNIT..MAX_UNIT characters), so a unit
MAX_UNIT = 300             # seldom straddles a change of topic
SENTENCE_ENDS = (". ", "? ", "! ", "; ", "。", "？", "！")

Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (na * nb) if na and nb else 0.0)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def sentence_units(text: str, start: int, end: int, separators: list[str], header: str) -> list[Piece]:
    """A range as units of a sentence or two: a unit ends at a paragraph break, at a sentence end
    once it has MIN_UNIT characters, and by MAX_UNIT; a protected block (table row, code block,
    display math) is a unit of its own, and inline spans (links, inline code) stay inside their
    sentence. Blank stretches are left out."""
    separators = separators + [sep for sep in SENTENCE_ENDS if sep not in separators]
    atoms = chunker._units(text, start, end, MAX_UNIT, separators)
    units: list[Piece] = []
    current: list[int] = []        # [start, end]

    def flush() -> None:
        if current and text[current[0]:current[1]].strip():
            units.append(Piece(text[current[0]:current[1]], current[0], current[1], header))
        current.clear()

    for atom in atoms:
        piece = text[atom.start:atom.end]
        if atom.protected and (atom.table is not None or "\n" in piece):   # a block: table row, code, display math
            flush()
            current[:] = [atom.start, atom.end]
            flush()
            continue
        if not current:
            current[:] = [atom.start, atom.end]
        else:
            current[1] = atom.end
        length = current[1] - current[0]
        if ("\n\n" in piece and not piece.strip()) or (piece in SENTENCE_ENDS or piece == "\n") and length >= MIN_UNIT \
                or length >= MAX_UNIT:
            flush()
    flush()
    return units


def group(text: str, units: list[Piece], distances: list[float], threshold: float, size: int,
          header: str) -> list[Piece]:
    """Units of one range into chunks: break at a distance ≥ threshold once the chunk has
    MIN_SHARE of `size`, and before it would pass `size`. distances[i] is between unit i and i+1."""
    out: list[Piece] = []
    current: list[Piece] = []

    def emit() -> None:
        first, last = current[0], current[-1]
        prepended = first.content[:first.content_offset] if first.content_offset else ""
        out.append(Piece(prepended + text[first.start:last.end], first.start, last.end, header,
                         content_offset=first.content_offset))

    for i, unit in enumerate(units):
        if current:
            length = current[-1].end - current[0].start
            shifted = distances[i - 1] >= threshold and length >= MIN_SHARE * size
            if shifted or length + (unit.end - current[-1].end) > size:
                emit()
                current = []
        current.append(unit)
    if current:
        emit()
    return out


async def chunk(text: str, config: dict | None, embed: Embed, default_strategy: str = "auto") -> list[Piece]:
    cfg = chunker.settings(config, default_strategy)
    separators = cfg["separators"]
    structural = chunker.resolve_strategy(text, "auto")
    titles = chunker.markdown_headings(text) if structural == "heading" else \
        chunker.heuristic_titles(text) if structural == "heuristic" else []

    # the ranges to cut by meaning, each with its size, its breadcrumb and (parent-child) its parent
    pieces: list[Piece] = []
    ranges: list[tuple[int, int, int, str, int | None]] = []
    if cfg.get("enable_parent_child"):
        child_size = int(cfg["child_chunk_size"])
        for parent in chunker._by_strategy(text, 0, len(text), int(cfg["parent_chunk_size"]), 0, separators,
                                           structural):
            parent.chunk_type = "parent_text"
            pieces.append(parent)
            ranges.append((parent.start, parent.end, child_size, parent.context_header, len(pieces) - 1))
    else:
        for start, end, crumb in chunker._sections(text, 0, len(text), titles):
            ranges.append((start, end, cfg["chunk_size"], crumb, None))

    units_per_range = [sentence_units(text, s, e, separators, crumb) for s, e, _, crumb, _ in ranges]
    flat = [u for units in units_per_range for u in units]
    try:
        vectors = await embed([u.content for u in flat]) if flat else []
    except Exception as exc:   # noqa: BLE001 - any model error: structural chunking instead
        LOG.warning("semantic chunking unavailable, chunking by structure: %s", exc)
        return chunker.chunk(text, {**cfg, "strategy": "auto"})
    distances_per_range, at = [], 0
    for units in units_per_range:
        vs = vectors[at:at + len(units)]
        at += len(units)
        distances_per_range.append([_cosine_distance(vs[i], vs[i + 1]) for i in range(len(vs) - 1)])
    threshold = percentile([d for ds in distances_per_range for d in ds], BREAK_PERCENTILE)

    if not cfg.get("enable_parent_child"):
        for (s, e, size, crumb, _), units, ds in zip(ranges, units_per_range, distances_per_range):
            pieces.extend(group(text, units, ds, threshold, size, crumb))
        return pieces
    out: list[Piece] = []
    for (s, e, size, crumb, parent_index), units, ds in zip(ranges, units_per_range, distances_per_range):
        out.append(pieces[parent_index])
        new_parent = len(out) - 1
        for child in group(text, units, ds, threshold, size, crumb):
            child.parent = new_parent
            out.append(child)
    return out
