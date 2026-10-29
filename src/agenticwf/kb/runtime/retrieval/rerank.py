"""Rerank, threshold degradation, composite score and MMR (docs/retrieval.md, Rerank).

1. Score at most 200 candidates with the rerank model.
2. Keep scores ≥ threshold; if none pass and the threshold is above 0.3, retry at
   max(0.3, 0.7 × threshold); if still none, keep the single best when its score ≥ 0.15, or
   always when the search was scoped to specific knowledge ids.
3. Composite = (0.6 · rerank + 0.3 · retrieval + 0.1 · source_weight) × recency_weight.
4. MMR (λ = 0.7, token Jaccard) down to top_k.
5. No rerank model, or it fails: the retrieval order cut to top_k.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

MAX_CANDIDATES = 200
MMR_LAMBDA = 0.7
_TOKENS = re.compile(r"\w+", re.UNICODE)

Scorer = Callable[[str, list[str]], Awaitable[list[float]]]


def tokens(text: str) -> set[str]:
    return set(_TOKENS.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def degrade(scored: list[tuple[dict, float]], threshold: float, scoped: bool) -> list[tuple[dict, float]]:
    kept = [(item, s) for item, s in scored if s >= threshold]
    if not kept and threshold > 0.3:
        lower = max(0.3, 0.7 * threshold)
        kept = [(item, s) for item, s in scored if s >= lower]
    if not kept and scored:
        best = max(scored, key=lambda pair: pair[1])
        if best[1] >= 0.15 or scoped:
            kept = [best]
    return kept


def mmr(items: list[dict], top_k: int, lam: float = MMR_LAMBDA) -> list[dict]:
    """Greedy maximal marginal relevance on `score`, similarity = token Jaccard of `content`."""
    remaining = list(items)
    token_sets = {id(item): tokens(item.get("content", "")) for item in items}
    chosen: list[dict] = []
    while remaining and len(chosen) < top_k:
        def value(item):
            redundancy = max((jaccard(token_sets[id(item)], token_sets[id(c)]) for c in chosen), default=0.0)
            return lam * item["score"] - (1 - lam) * redundancy
        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen


async def rerank(query: str, items: list[dict], scorer: Scorer | None, top_k: int, threshold: float = 0.3,
                 scoped: bool = False) -> list[dict]:
    """`items` carry `score` (retrieval) and `content`; returns the final list."""
    if scorer is None or not items:
        return items[:top_k]
    candidates = items[:MAX_CANDIDATES]
    try:
        scores = await scorer(query, [item.get("content", "") for item in candidates])
    except Exception:
        return items[:top_k]            # degrade to retrieval order
    kept = degrade(list(zip(candidates, scores)), threshold, scoped)
    out = []
    for item, rerank_score in kept:
        composite = (0.6 * rerank_score + 0.3 * item["score"] + 0.1 * item.get("source_weight", 1.0)) \
            * item.get("recency_weight", 1.0)
        out.append({**item, "rerank_score": rerank_score, "score": composite})
    out.sort(key=lambda item: -item["score"])
    return mmr(out, top_k)
