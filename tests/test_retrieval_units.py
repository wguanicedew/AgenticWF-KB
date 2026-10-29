"""Fusion, rerank degradation, MMR and merge (docs/retrieval.md): pure functions, no database."""

import asyncio

import pytest

from agenticwf.kb.runtime.retrieval.fusion import rrf
from agenticwf.kb.runtime.retrieval.merge import merge
from agenticwf.kb.runtime.retrieval.rerank import degrade, mmr, rerank


def test_rrf_weights_and_normalisation():
    fused = rrf([("a", 0.9), ("b", 0.8)], [("b", 12.0), ("c", 5.0)], k=60, vector_weight=0.7, keyword_weight=0.3)
    by_id = {f.chunk_id: f for f in fused}
    assert by_id["b"].match_type == "hybrid" and by_id["a"].match_type == "vector" and by_id["c"].match_type == "keyword"
    norm = (0.7 + 0.3) / 61
    assert by_id["b"].score == pytest.approx((0.7 / 62 + 0.3 / 61) / norm)
    assert by_id["a"].score == pytest.approx((0.7 / 61) / norm)
    assert fused[0].chunk_id == "b"                       # in both lists
    assert all(0 <= f.score <= 1 for f in fused)


def test_rrf_single_retriever_keeps_its_scores():
    fused = rrf([("a", 0.42)], [])
    assert fused[0].score == 0.42 and fused[0].match_type == "vector"
    fused = rrf([], [("x", 1.0), ("y", 0.5)])
    assert [(f.chunk_id, f.score) for f in fused] == [("x", 1.0), ("y", 0.5)]


def test_rrf_keeps_the_best_row_per_chunk():
    fused = rrf([("a", 0.3), ("a", 0.9)], [])
    assert len(fused) == 1 and fused[0].score == 0.9


def test_threshold_degradation():
    items = [({"id": 1}, 0.25), ({"id": 2}, 0.18)]
    assert degrade(items, 0.7, scoped=False) == [({"id": 1}, 0.25)]      # 0.7 → 0.49 → best ≥ 0.15
    assert degrade([({"id": 1}, 0.1)], 0.5, scoped=False) == []
    assert degrade([({"id": 1}, 0.1)], 0.5, scoped=True) == [({"id": 1}, 0.1)]
    assert degrade([({"id": 1}, 0.4), ({"id": 2}, 0.35)], 0.5, scoped=False) == [({"id": 1}, 0.4), ({"id": 2}, 0.35)]


def test_mmr_prefers_diverse_results():
    items = [{"id": 1, "score": 0.9, "content": "pilot restart stage out retry"},
             {"id": 2, "score": 0.89, "content": "pilot restart stage out retry again"},
             {"id": 3, "score": 0.7, "content": "cryostat pressure alarm threshold"}]
    assert [i["id"] for i in mmr(items, 2)] == [1, 3]


def test_rerank_composite_and_failure_fallback():
    items = [{"id": "a", "score": 0.5, "content": "alpha beta"}, {"id": "b", "score": 0.4, "content": "gamma delta"}]

    async def scorer(query, docs):
        return [0.2, 0.9]

    out = asyncio.run(rerank("q", items, scorer, top_k=2, threshold=0.1))
    assert [i["id"] for i in out] == ["b", "a"]
    assert out[0]["score"] == pytest.approx(0.6 * 0.9 + 0.3 * 0.4 + 0.1)

    async def broken(query, docs):
        raise RuntimeError("reranker down")

    assert [i["id"] for i in asyncio.run(rerank("q", items, broken, top_k=1))] == ["a"]
    assert [i["id"] for i in asyncio.run(rerank("q", items, None, top_k=5))] == ["a", "b"]


def _chunk(i, start, end, content, **kw):
    return {"id": f"c{i}", "knowledge_id": "k", "chunk_type": "text", "chunk_index": i, "start_at": start,
            "end_at": end, "content": content, "pre_chunk_id": f"c{i - 1}" if i else "", "next_chunk_id": f"c{i + 1}",
            "parent_chunk_id": "", "is_enabled": True, **kw}


def test_merge_adjacent_and_overlapping():
    text = "0123456789abcdefghij"
    c0, c1, c3 = _chunk(0, 0, 10, text[0:10]), _chunk(1, 8, 18, text[8:18]), _chunk(3, 30, 40, "x" * 10)
    results = [{**c1, "score": 0.9}, {**c0, "score": 0.5}, {**c3, "score": 0.4}]
    contexts = merge(results, lambda cid: None, chunk_size=10)
    first = contexts[0]
    assert first["merged_chunk_ids"] == ["c0", "c1"] and first["content"] == text[0:18] and first["score"] == 0.9
    assert contexts[1]["merged_chunk_ids"] == ["c3"]


def test_merge_resolves_parents_and_expands_short_contexts():
    parent = {"id": "p", "content": "PARENT TEXT", "start_at": 0, "end_at": 11}
    child_a = _chunk(1, 0, 5, "PAREN", parent_chunk_id="p")
    child_b = _chunk(2, 5, 11, "T TEXT", parent_chunk_id="p")
    neighbours = {"n0": {"id": "n0", "content": "before.", "start_at": 0, "end_at": 7, "is_enabled": True},
                  "n2": {"id": "n2", "content": "after.", "start_at": 12, "end_at": 18, "is_enabled": True}}
    short = {**_chunk(1, 7, 11, "tiny"), "knowledge_id": "other", "id": "s", "pre_chunk_id": "n0",
             "next_chunk_id": "n2"}
    lookup = {"p": parent, **neighbours}.get
    contexts = merge([{**child_a, "score": 0.8}, {**child_b, "score": 0.6}, {**short, "score": 0.5}], lookup,
                     chunk_size=100)
    by_first = {c["merged_chunk_ids"][0]: c for c in contexts}
    assert by_first["c1"]["content"] == "PARENT TEXT" and by_first["c1"]["merged_chunk_ids"] == ["c1", "c2"]
    expanded = by_first["n0"]
    assert expanded["merged_chunk_ids"] == ["n0", "s", "n2"] and expanded["content"].startswith("before.")


def test_merge_drops_fully_covered_contexts():
    a = {**_chunk(0, 0, 10, "a" * 10), "score": 0.9, "chunk_type": "faq"}
    contexts = merge([a, {**a, "score": 0.1}], lambda cid: None)
    assert len(contexts) == 1
