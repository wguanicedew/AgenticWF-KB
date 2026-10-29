"""
Similar records (docs/structured-records.md, Similar records):

    S = wt·T + wf·F + wx·X      weights from the type's `similarity`, renormalised over the parts present

    T  text: cosine of the text-role embeddings
    F  facets: share of the probe's facets with an equal value
    X  features: mean over features present in both; numeric 1 − |a−b|/range (log scale for
       `x-log` fields, range from the field statistics), categorical equal → 1, sets Jaccard

Missing features are unknown and never count against a record. With `diversify`, the results
are picked by MMR, redundancy = the facets and features part of S between two results.
"""

from __future__ import annotations

import math

DEFAULT_WEIGHTS = {"text": 0.3, "facets": 0.3, "features": 0.4}
MMR_LAMBDA = 0.7


def as_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return max(0.0, dot / norm) if norm else 0.0


def feature_similarity(a, b, low: float | None = None, high: float | None = None, log: bool = False) -> float | None:
    if a is None or b is None:
        return None
    if isinstance(a, list) or isinstance(b, list):
        sa = {str(x) for x in (a if isinstance(a, list) else [a])}
        sb = {str(x) for x in (b if isinstance(b, list) else [b])}
        return len(sa & sb) / len(sa | sb) if sa | sb else None
    na, nb = as_number(a), as_number(b)
    if na is not None and nb is not None:
        if log and na > 0 and nb > 0 and low and high and low > 0:
            na, nb, low, high = math.log(na), math.log(nb), math.log(low), math.log(high)
        if low is None or high is None or high <= low:
            return 1.0 if na == nb else 0.0
        return max(0.0, 1.0 - abs(na - nb) / (high - low))
    return 1.0 if str(a) == str(b) else 0.0


def facet_similarity(probe: dict, other: dict) -> float | None:
    if not probe:
        return None
    return sum(1 for k, v in probe.items() if k in other and str(other[k]) == str(v)) / len(probe)


def features_similarity(probe: dict, other: dict, ranges: dict[str, tuple], logs: set[str]) -> float | None:
    scores = []
    for path, value in probe.items():
        if path not in other:
            continue
        low, high = ranges.get(path, (None, None))
        s = feature_similarity(value, other[path], low, high, path in logs)
        if s is not None:
            scores.append(s)
    return sum(scores) / len(scores) if scores else None


def combine(parts: dict[str, float | None], weights: dict | None = None) -> float:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    present = {k: v for k, v in parts.items() if v is not None and weights.get(k, 0) > 0}
    total = sum(weights[k] for k in present)
    return sum(weights[k] * v for k, v in present.items()) / total if total else 0.0


def record_similarity(a: dict, b: dict, ranges: dict[str, tuple], logs: set[str], weights: dict | None = None) -> float:
    """How alike two results are (facets and features, as S without text); 0 across types."""
    if a.get("record_type") != b.get("record_type"):
        return 0.0
    parts = {"facets": facet_similarity(a.get("facets") or {}, b.get("facets") or {}),
             "features": features_similarity(a.get("features") or {}, b.get("features") or {}, ranges, logs)}
    return combine(parts, weights)


def mmr(items: list[dict], top_k: int, alike, lam: float = MMR_LAMBDA) -> list[dict]:
    """Greedy maximal marginal relevance on `score`: λ·score − (1−λ)·max alike(item, chosen)."""
    remaining = list(items)
    chosen: list[dict] = []
    while remaining and len(chosen) < top_k:
        def value(item):
            return lam * item["score"] - (1 - lam) * max((alike(item, c) for c in chosen), default=0.0)
        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen
