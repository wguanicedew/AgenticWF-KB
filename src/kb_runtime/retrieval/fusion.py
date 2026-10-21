"""Weighted reciprocal rank fusion (docs/retrieval.md, Fusion).

    score = wv / (k + rank_v) + wk / (k + rank_k),  normalised by (wv + wk) / (k + 1) into [0, 1]

A result missing from one list contributes 0 for that term. When only one retriever returned
anything, its own score is kept (raw cosine for vector, bm25 / max for keyword).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fused:
    chunk_id: str
    score: float
    match_type: str             # vector | keyword | hybrid
    vector_score: float | None = None
    keyword_score: float | None = None


def best_per_chunk(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Several index rows may map to one chunk (questions, summaries): keep the best, in order."""
    best: dict[str, float] = {}
    for chunk_id, score in hits:
        if chunk_id not in best or score > best[chunk_id]:
            best[chunk_id] = score
    return sorted(best.items(), key=lambda item: -item[1])


def rrf(vector: list[tuple[str, float]], keyword: list[tuple[str, float]], k: int = 60,
        vector_weight: float = 0.7, keyword_weight: float = 0.3) -> list[Fused]:
    vector, keyword = best_per_chunk(vector), best_per_chunk(keyword)
    if not vector or not keyword:
        only, kind = (vector, "vector") if vector else (keyword, "keyword")
        return [Fused(cid, score, kind, score if kind == "vector" else None, score if kind == "keyword" else None)
                for cid, score in only]
    v_rank = {cid: rank for rank, (cid, _) in enumerate(vector, start=1)}
    k_rank = {cid: rank for rank, (cid, _) in enumerate(keyword, start=1)}
    v_score, k_score = dict(vector), dict(keyword)
    norm = (vector_weight + keyword_weight) / (k + 1)
    fused = []
    for cid in dict.fromkeys([c for c, _ in vector] + [c for c, _ in keyword]):
        score = (vector_weight / (k + v_rank[cid]) if cid in v_rank else 0.0) + \
                (keyword_weight / (k + k_rank[cid]) if cid in k_rank else 0.0)
        kind = "hybrid" if cid in v_rank and cid in k_rank else ("vector" if cid in v_rank else "keyword")
        fused.append(Fused(cid, score / norm, kind, v_score.get(cid), k_score.get(cid)))
    fused.sort(key=lambda f: -f.score)
    return fused
