"""Merge and expansion: search results into contexts (docs/retrieval.md, Merge and expansion).

Used when passages are assembled into context (agentic search, phase 2); pure functions over
result items and a chunk lookup, so they are unit-tested without a database.

1. resolve parents: a child hit is replaced by its parent's text; children of one parent collapse
2. pass through parent-resolved contexts and image / faq chunks unchanged
3. merge adjacent or overlapping chunks of the same item and chunk type
4. expand contexts shorter than chunk_size / 2 with their previous and next chunk
5. drop a context whose chunk ids are all covered by a higher-ranked one
"""

from __future__ import annotations

from typing import Callable

PASS_THROUGH = {"image_ocr", "image_caption", "faq", "wiki_page", "record"}

ChunkLookup = Callable[[str], dict | None]     # chunk id -> chunk dict (content, offsets, links) or None


def _join(a: dict, b: dict) -> str:
    """Join two chunks of one item by offsets, writing overlapping text once."""
    if b["start_at"] < a["end_at"]:
        overlap = a["end_at"] - b["start_at"]
        return a["content"] + b["content"][overlap:]
    return a["content"] + "\n" + b["content"]


def merge(results: list[dict], lookup: ChunkLookup, chunk_size: int = 512) -> list[dict]:
    # 1. parents
    contexts: list[dict] = []
    by_parent: dict[str, dict] = {}
    rest: list[dict] = []
    for item in results:
        parent_id = item.get("parent_chunk_id")
        parent = lookup(parent_id) if parent_id else None
        if parent is not None:
            if parent_id in by_parent:
                ctx = by_parent[parent_id]
                ctx["merged_chunk_ids"].append(item["id"])
                ctx["score"] = max(ctx["score"], item["score"])
                continue
            ctx = {**item, "content": parent["content"], "start_at": parent["start_at"], "end_at": parent["end_at"],
                   "merged_chunk_ids": [item["id"]], "resolved_parent": True}
            by_parent[parent_id] = ctx
            contexts.append(ctx)
        elif item.get("chunk_type") in PASS_THROUGH:
            contexts.append({**item, "merged_chunk_ids": [item["id"]]})
        else:
            rest.append(item)

    # 3. adjacent / overlapping within (item, chunk type)
    groups: dict[tuple, list[dict]] = {}
    for item in rest:
        groups.setdefault((item["knowledge_id"], item.get("chunk_type", "text")), []).append(item)
    for members in groups.values():
        members.sort(key=lambda m: m["chunk_index"])
        current = None
        for item in members:
            if current is not None and (item["chunk_index"] == current["_last_index"] + 1
                                        or item["start_at"] < current["end_at"]):
                current["content"] = _join(current, item)
                current["end_at"] = max(current["end_at"], item["end_at"])
                current["merged_chunk_ids"].append(item["id"])
                current["score"] = max(current["score"], item["score"])
                current["_last_index"] = item["chunk_index"]
                current["next_chunk_id"] = item.get("next_chunk_id", "")
                continue
            current = {**item, "merged_chunk_ids": [item["id"]], "_last_index": item["chunk_index"]}
            contexts.append(current)

    # 4. expand short contexts
    merged_ids = {cid for ctx in contexts for cid in ctx["merged_chunk_ids"]}
    for ctx in contexts:
        ctx.pop("_last_index", None)
        if ctx.get("resolved_parent") or ctx.get("chunk_type") in PASS_THROUGH or len(ctx["content"]) >= chunk_size / 2:
            continue
        before = lookup(ctx.get("pre_chunk_id") or "")
        if before is not None and before.get("is_enabled", True) and before["id"] not in merged_ids:
            ctx["content"] = _join(before, {**ctx, "start_at": ctx["start_at"]})
            ctx["start_at"] = before["start_at"]
            ctx["merged_chunk_ids"].insert(0, before["id"])
            merged_ids.add(before["id"])
        after = lookup(ctx.get("next_chunk_id") or "")
        if after is not None and after.get("is_enabled", True) and after["id"] not in merged_ids:
            ctx["content"] = _join(ctx, after)
            ctx["end_at"] = after["end_at"]
            ctx["merged_chunk_ids"].append(after["id"])
            merged_ids.add(after["id"])

    # order by score, 5. drop contexts fully covered by a better one
    contexts.sort(key=lambda c: -c["score"])
    seen: set[str] = set()
    out = []
    for ctx in contexts:
        ids = set(ctx["merged_chunk_ids"])
        if ids <= seen:
            continue
        seen |= ids
        out.append(ctx)
    return out
