"""Contextual and semantic chunking (docs/ingestion.md): pure functions with a fake chat model
and a fake embedder, no database."""

import asyncio
import json

from agenticwf.kb.runtime.ingestion import chunker, contextual, semantic
from agenticwf.kb.runtime.services.knowledge import index_text


# --- contextual chunking ------------------------------------------------------------------------------

class FakeChat:
    def __init__(self, fail_window=None, short_window=None):
        self.calls, self.fail_window, self.short_window = [], fail_window, short_window

    async def complete(self, messages, max_tokens=None, temperature=0.3, **kw):
        user = messages[-1]["content"]
        self.calls.append(user)
        n = user.count("<passage n=")
        window = len(self.calls) - 1
        if window == self.fail_window:
            raise RuntimeError("gateway down")
        if window == self.short_window:
            n -= 1
        return {"content": "<think>x</think>" + json.dumps({"contexts": [f"ctx {window}.{i}" for i in range(n)]})}


def test_contexts_per_window_and_failures_leave_chunks_without():
    doc = "# Cryostat\n" + "word " * 3000
    spans = [(i * 100, i * 100 + 90, f"chunk {i}") for i in range(30)]           # 3 windows of 12, 12, 6
    chat = FakeChat()
    out = asyncio.run(contextual.chunk_contexts(chat, "Cryostat manual", doc, spans))
    assert len(chat.calls) == 3 and all("Cryostat manual" in c and "# Cryostat" in c for c in chat.calls)
    assert out[0].startswith("ctx ") and all(out)

    out = asyncio.run(contextual.chunk_contexts(FakeChat(fail_window=0), "t", doc, spans))
    assert sum(1 for c in out if not c) == 12                                      # one failed window
    out = asyncio.run(contextual.chunk_contexts(FakeChat(short_window=0), "t", doc, spans))
    assert sum(1 for c in out if not c) == 12                                      # one context short: none kept
    assert asyncio.run(contextual.chunk_contexts(FakeChat(), "t", doc, [])) == []


def test_parse_contexts_and_index_text():
    assert contextual.parse_contexts('{"contexts": ["a  b", "c"]}', 2) == ["a b", "c"]
    assert contextual.parse_contexts('{"contexts": ["a"]}', 2) is None and contextual.parse_contexts("nope", 1) is None
    assert index_text("T", "", "body") == "T\nbody"                                # unchanged without extras
    assert index_text("T", "A > B", "body") == "T\nA > B\n\nbody"
    assert index_text("T", "A > B", "body", "From the cryostat manual.") == "T\nA > B\nFrom the cryostat manual.\n\nbody"


# --- semantic chunking -----------------------------------------------------------------------------------

TOPIC_A = "The cryostat keeps the magnet cold. Its pressure alarm fires at 1.2 bar. Call the cryo expert. "
TOPIC_B = "Shift handover happens at seven. The shift leader signs the logbook. Read the e-log first. "


async def topic_embed(texts):
    """A 2-d 'meaning': cryostat words vs shift words."""
    return [[t.lower().count("cryo") + t.count("bar") + t.count("magnet") + 0.01,
             t.lower().count("shift") + t.count("logbook") + t.count("e-log") + 0.01] for t in texts]


def test_semantic_breaks_where_the_topic_changes():
    text = TOPIC_A * 4 + "\n\n" + TOPIC_B * 4
    pieces = asyncio.run(semantic.chunk(text, {"strategy": "semantic", "chunk_size": 1000}, topic_embed))
    assert len(pieces) >= 2
    for p in pieces:                                                               # exact offsets, as the chunker's
        assert p.content == text[p.start:p.end]
    assert "cryostat" in pieces[0].content and "shift" not in pieces[0].content.lower()
    assert "shift" in pieces[-1].content.lower() and "cryostat" not in pieces[-1].content
    # by size alone, the same text is cut without regard to topics
    plain = chunker.chunk(text, {"chunk_size": 1000, "strategy": "legacy"})
    assert len(plain) == 1


def test_semantic_respects_size_sections_and_parent_child():
    text = "# A\n" + TOPIC_A * 8 + "\n# B\n" + TOPIC_B * 8
    pieces = asyncio.run(semantic.chunk(text, {"strategy": "semantic", "chunk_size": 400}, topic_embed))
    assert all(len(p.content) <= 400 + 120 for p in pieces)                        # a unit may overhang slightly
    assert {p.context_header for p in pieces} == {"A", "B"}                        # never across a section
    tree = asyncio.run(semantic.chunk(text, {"strategy": "semantic", "enable_parent_child": True,
                                             "parent_chunk_size": 1200, "child_chunk_size": 300}, topic_embed))
    parents = [i for i, p in enumerate(tree) if p.chunk_type == "parent_text"]
    children = [p for p in tree if p.chunk_type == "text"]
    assert parents and children and all(c.parent in parents for c in children)
    for c in children:
        parent = tree[c.parent]
        assert parent.start <= c.start and c.end <= parent.end


def test_semantic_falls_back_to_structure_when_embedding_fails():
    async def broken(texts):
        raise RuntimeError("embedding service down")
    text = "# A\n" + TOPIC_A * 3 + "\n# B\n" + TOPIC_B * 3
    pieces = asyncio.run(semantic.chunk(text, {"strategy": "semantic", "chunk_size": 500}, broken))
    assert pieces and [p.context_header for p in pieces][0] == "A"
    assert chunker.resolve_strategy(text, "semantic") == "heading"                 # the sync chunker: by structure


def test_sentence_units_keep_links_inline_and_blocks_whole():
    text = "See [the manual](http://x/m) for the limits. It is long enough to stand alone here.\n\n```\ncode\n\nmore\n```\nAfter."
    units = semantic.sentence_units(text, 0, len(text), ["\n\n", "\n"], "")
    assert any("[the manual](http://x/m) for the limits" in u.content for u in units)
    assert any(u.content.startswith("```") and u.content.rstrip().endswith("```") for u in units)
    assert all(u.content == text[u.start:u.end] for u in units)


# --- end to end: a document through the pipeline -------------------------------------------------------

def test_contextual_and_semantic_chunks_are_indexed(database):
    from conftest import Caller, drain, make_client, sql_rows, user

    rw = ("app:access", "knowledge:read", "knowledge:write")
    with make_client("authsvc") as client:
        chat = FakeChat()

        async def any_chat(ref):
            return chat

        client.app.state.manager.services.models.chat = any_chat
        alice = Caller(client, user("alice", rw))
        model = alice.post("/kb/api/v1/models", json={"name": "c", "type": "KnowledgeQA", "parameters": {
            "base_url": "http://127.0.0.1:9/v1"}}).json()["data"]["id"]
        kb = alice.post("/kb/api/v1/knowledge-bases", json={
            "name": "Ops", "summary_model_id": model,
            "indexing_strategy": {"contextual_enabled": True},
            "chunking_config": {"strategy": "semantic", "chunk_size": 1000}}).json()["data"]
        doc = alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/knowledge/manual",
                         json={"title": "Ops guide", "content": TOPIC_A * 4 + "\n\n" + TOPIC_B * 4}).json()["data"]
        drain(client)
        chunks = sql_rows("SELECT content, metadata FROM kb_chunks WHERE knowledge_id = :k AND chunk_type = 'text' "
                          "ORDER BY chunk_index", k=doc["id"])
        indexed = [r[0] for r in sql_rows("SELECT content FROM kb_embeddings WHERE knowledge_id = :k AND source_type = 0",
                                          k=doc["id"])]
    assert len(chunks) >= 2                                                            # cut where the topic changes
    assert "cryostat" in chunks[0][0] and "shift" not in chunks[0][0].lower()
    assert all(c[1]["context"].startswith("ctx ") for c in chunks)                     # kept on the chunk
    assert all("ctx " in text for text in indexed)                                     # and searched
    assert all("ctx " not in c[0] for c in chunks)                                     # not in what readers see
    assert any("<passage n=" in call for call in chat.calls)
