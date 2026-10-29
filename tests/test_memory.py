"""Temporal memory: extraction, entity resolution, bi-temporal facts, contradiction and
invalidation, as_of / known_at, deleting episodes, search, access, the KB feed and recall."""

import json
import re

import pytest

from conftest import Caller, drain, make_client, user

RW = ("app:access", "knowledge:read", "knowledge:write")

EXTRACTIONS = {
    "Alice works at CERN": {
        "entities": [{"name": "Alice", "type": "person", "summary": "a physicist"},
                     {"name": "CERN", "type": "organisation", "summary": "a laboratory"}],
        "facts": [{"source": "Alice", "relation": "works at", "target": "CERN", "fact": "Alice works at CERN."}]},
    "Alice moved to DESY": {
        "entities": [{"name": "Alice Smith", "type": "person", "aliases": ["Alice"]},
                     {"name": "DESY", "type": "organisation"}],
        "facts": [{"source": "Alice Smith", "relation": "WORKS_AT", "target": "DESY", "fact": "Alice works at DESY."}]},
    "Pilot doc": {
        "entities": [{"name": "Pilot", "type": "system"}, {"name": "Stage-out", "type": "concept"}],
        "facts": [{"source": "Pilot", "relation": "RETRIES", "target": "Stage-out", "fact": "The pilot retries stage-out 5 times."}]},
}


class MemoryLLM:
    def __init__(self):
        self.contradictions = 0

    async def complete(self, messages, max_tokens=None, temperature=0.0, tools=None, **kw):
        system, user_text = messages[0]["content"], messages[-1]["content"]
        if "You extract a knowledge graph" in system:
            for key, value in EXTRACTIONS.items():
                if key.lower() in user_text.lower() or ("stage-out" in user_text.lower() and key == "Pilot doc"):
                    return self._json(value)
            return self._json({"entities": [], "facts": []})
        if "You compare a new fact" in system:
            self.contradictions += 1
            ids = re.findall(r"^(\S+): Alice works at CERN", user_text, re.MULTILINE)
            return self._json({"contradicted": ids if "DESY" in user_text.split("Existing facts")[0] else []})
        if "same real-world thing" in system:
            return self._json({"same": False})
        return self._json({})

    @staticmethod
    def _json(value):
        return {"content": json.dumps(value), "tool_calls": [], "usage": {}, "finish_reason": "stop"}


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        llm = MemoryLLM()

        async def chat(ref):
            return llm

        c.app.state.manager.services.models.chat = chat
        c.llm = llm
        alice = Caller(c, user("alice", RW, org="eic"))
        alice.post("/kb/api/v1/models", json={"name": "mem-chat", "type": "KnowledgeQA", "system": False,
                                              "parameters": {"base_url": "http://127.0.0.1:9/v1"}})
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def ok(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def episode(alice, content, when):
    return ok(alice.post("/kb/api/v1/memory/episodes", json={"content": content, "reference_time": when,
                                                             "source": "conversation"}), 202)


def chat_default(client, alice):
    """Memory uses [memory] model or the default chat model: point the default at the fake."""
    async def default_id(kind, _orig=client.app.state.manager.services.models.default_id):
        return "fake" if kind == "KnowledgeQA" else await _orig(kind)
    client.app.state.manager.services.models.default_id = default_id


def test_contradiction_invalidates_and_as_of(client, alice):
    chat_default(client, alice)
    first = episode(alice, "Alice works at CERN since 2025.", "2026-01-01T00:00:00Z")
    drain(client)
    assert ok(alice.get(f"/kb/api/v1/memory/episodes/{first['id']}"))["status"] == "processed"
    second = episode(alice, "Alice moved to DESY.", "2026-06-01T00:00:00Z")
    drain(client)
    current = ok(alice.get("/kb/api/v1/memory/facts"))
    assert [f["fact"] for f in current] == ["Alice works at DESY."]
    assert current[0]["source"] == "Alice" and current[0]["relation"] == "WORKS_AT"   # one entity, aliases merged
    then = ok(alice.get("/kb/api/v1/memory/facts?as_of=2026-03-01T00:00:00Z"))
    assert [f["fact"] for f in then] == ["Alice works at CERN."]
    everything = ok(alice.get("/kb/api/v1/memory/facts?include_invalid=true"))
    old = [f for f in everything if "CERN" in f["fact"]][0]
    assert old["invalid_at"].startswith("2026-06-01") and old["invalidated_by"] == current[0]["id"]
    before_desy = ok(alice.get("/kb/api/v1/memory/facts?known_at=2026-01-01T00:00:00Z&include_invalid=true"))
    assert before_desy == []                                    # system time: nothing was known yet
    entities = ok(alice.get("/kb/api/v1/memory/entities"))
    person = [e for e in entities if e["name"] == "Alice"][0]
    assert "Alice Smith" in person["aliases"]
    history = ok(alice.get(f"/kb/api/v1/memory/entities/{person['id']}"))
    assert len(history["facts"]) == 2

    # search: current by default, as_of for the past, focal entity neighbourhood
    hits = ok(alice.post("/kb/api/v1/memory/search", json={"query": "where does Alice work"}))
    assert hits[0]["fact"] == "Alice works at DESY." and all("CERN" not in h["fact"] for h in hits)
    past = ok(alice.post("/kb/api/v1/memory/search", json={"query": "Alice work", "as_of": "2026-03-01T00:00:00Z"}))
    assert [h["fact"] for h in past] == ["Alice works at CERN."]
    focal = ok(alice.post("/kb/api/v1/memory/search", json={"query": "laboratory", "focal_entity": "Alice"}))
    assert focal and focal[0]["fact"] == "Alice works at DESY."

    # the same fact again: one more supporting episode
    third = episode(alice, "Alice moved to DESY last week.", "2026-06-02T00:00:00Z")
    drain(client)
    desy = ok(alice.get("/kb/api/v1/memory/facts"))[0]
    assert set(desy["episode_ids"]) == {second["id"], third["id"]}
    # deleting one supporter keeps the fact; deleting the last removes it and reverts the invalidation
    assert ok(alice.delete(f"/kb/api/v1/memory/episodes/{second['id']}"))["facts_removed"] == 0
    result = ok(alice.delete(f"/kb/api/v1/memory/episodes/{third['id']}"))
    assert result == {"deleted": True, "facts_removed": 1, "invalidations_reverted": 1}
    assert [f["fact"] for f in ok(alice.get("/kb/api/v1/memory/facts"))] == ["Alice works at CERN."]

    bob = Caller(client, user("bob", RW, org="atlas"))
    assert ok(bob.get("/kb/api/v1/memory/facts")) == []
    assert bob.get(f"/kb/api/v1/memory/episodes/{first['id']}").status_code == 404
    assert ok(bob.post("/kb/api/v1/memory/search", json={"query": "Alice"})) == []


def test_kb_documents_feed_memory_and_recall_tool(client, alice):
    chat_default(client, alice)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Ops", "memory_config": {"enabled": True,
                                                                                            "group_id": "ops"}}), 201)
    ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/knowledge/manual",
                  json={"title": "Pilot doc", "content": "# Pilot\n\nThe pilot retries stage-out 5 times."}), 201)
    drain(client)
    episodes = alice.get("/kb/api/v1/memory/episodes?group_id=ops").json()["data"]
    assert len(episodes) == 1 and episodes[0]["source"] == "document" and episodes[0]["status"] == "processed"
    assert ok(alice.get("/kb/api/v1/memory/facts?group_id=ops"))[0]["fact"] == "The pilot retries stage-out 5 times."

    from test_agentic import ScriptedChat, call, chat_model

    model = chat_model(alice)
    fake = ScriptedChat([{"tool_calls": [call("recall", query="pilot stage-out retries"),
                                         call("memory_history", entity="Pilot")]},
                         {"content": "Five retries."}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    ok(alice.post("/kb/api/v1/agentic-search", json={"query": "how many retries?", "knowledge_base_ids": [kb["id"]],
                                                     "model_id": model}))
    recalled, history = [m["content"] for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert "The pilot retries stage-out 5 times." in recalled and 'relation="RETRIES"' in recalled
    assert '<history entity="Pilot">' in history


def test_forget_one_fact_or_an_entity(client, alice):
    chat_default(client, alice)
    episode(alice, "Alice works at CERN since 2025.", "2026-01-01T00:00:00Z")
    drain(client)
    episode(alice, "Alice moved to DESY.", "2026-06-01T00:00:00Z")
    drain(client)
    desy = ok(alice.get("/kb/api/v1/memory/facts"))[0]
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.delete(f"/kb/api/v1/memory/facts/{desy['id']}").status_code == 404

    # one fact: the invalidation it caused is reverted, its episodes are named (they keep the text)
    result = ok(alice.delete(f"/kb/api/v1/memory/facts/{desy['id']}"))
    assert result["invalidations_reverted"] == 1 and result["episode_ids"] == desy["episode_ids"]
    assert [f["fact"] for f in ok(alice.get("/kb/api/v1/memory/facts"))] == ["Alice works at CERN."]

    # an entity, with every fact about it
    cern = [e for e in ok(alice.get("/kb/api/v1/memory/entities")) if e["name"] == "CERN"][0]
    assert bob.delete(f"/kb/api/v1/memory/entities/{cern['id']}").status_code == 404
    result = ok(alice.delete(f"/kb/api/v1/memory/entities/{cern['id']}"))
    assert result["facts_removed"] == 1
    assert ok(alice.get("/kb/api/v1/memory/facts?include_invalid=true")) == []
    assert "CERN" not in [e["name"] for e in ok(alice.get("/kb/api/v1/memory/entities"))]
    assert alice.get(f"/kb/api/v1/memory/entities/{cern['id']}").status_code == 404


def test_redact_memory(client, alice):
    import sqlalchemy as sa

    from agenticwf.kb.core.orm.base.session import get_engine

    chat_default(client, alice)
    first = episode(alice, "Alice works at CERN since 2025.", "2026-01-01T00:00:00Z")
    drain(client)
    preview = ok(alice.post("/kb/api/v1/memory/redact", json={"terms": ["CERN"], "dry_run": True}))
    assert preview == {"dry_run": True, "episodes": 1, "facts": 1, "entities": 1, "matches": 3}
    assert ok(alice.get("/kb/api/v1/memory/facts"))[0]["fact"] == "Alice works at CERN."

    done = ok(alice.post("/kb/api/v1/memory/redact", json={"terms": ["CERN"]}))
    assert done["matches"] == 3
    assert ok(alice.get(f"/kb/api/v1/memory/episodes/{first['id']}"))["content"] == \
        "Alice works at [REDACTED] since 2025."
    assert ok(alice.get("/kb/api/v1/memory/facts"))[0]["fact"] == "Alice works at [REDACTED]."
    assert "CERN" not in str(ok(alice.get("/kb/api/v1/memory/entities")))
    with get_engine().connect() as conn:          # the old vectors are gone until re-embedded
        assert conn.execute(sa.text("SELECT count(*) FROM kb_memory_facts WHERE embedding IS NULL")).scalar() == 1
    drain(client)
    with get_engine().connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM kb_memory_facts WHERE embedding IS NULL")).scalar() == 0
        assert conn.execute(sa.text("SELECT count(*) FROM kb_memory_entities WHERE embedding IS NULL")).scalar() == 0
    # another user's redaction never touches alice's memory
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert ok(bob.post("/kb/api/v1/memory/redact", json={"terms": ["Alice"]}))["matches"] == 0


def test_redacting_a_document_reaches_the_memory_it_fed(client, alice):
    chat_default(client, alice)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Ops", "memory_config": {"enabled": True,
                                                                                            "group_id": "ops"}}), 201)
    item = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/knowledge/manual",
                         json={"title": "Pilot doc", "content": "# Pilot\n\nThe pilot retries stage-out 5 times."}), 201)
    episode(alice, "Unrelated: the stage-out of the other system.", "2026-01-01T00:00:00Z")
    drain(client)
    report = ok(alice.post(f"/kb/api/v1/knowledge/{item['id']}/redact", json={"terms": ["5 times"]}))
    assert report["memory"]["episodes"] == 1 and report["memory"]["facts"] == 1
    facts = ok(alice.get("/kb/api/v1/memory/facts?group_id=ops"))
    assert facts[0]["fact"] == "The pilot retries stage-out [REDACTED]."
    episodes = ok(alice.get("/kb/api/v1/memory/episodes?group_id=ops"))
    assert all("5 times" not in e["content"] for e in episodes)


def test_deleting_a_document_forgets_the_memory_it_fed(client, alice):
    chat_default(client, alice)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Ops", "memory_config": {"enabled": True,
                                                                                            "group_id": "ops"}}), 201)
    item = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/knowledge/manual",
                         json={"title": "Pilot doc", "content": "# Pilot\n\nThe pilot retries stage-out 5 times."}), 201)
    mine = episode(alice, "Alice works at CERN since 2025.", "2026-01-01T00:00:00Z")
    drain(client)
    assert len(ok(alice.get("/kb/api/v1/memory/facts?group_id=ops"))) == 1
    ok(alice.delete(f"/kb/api/v1/knowledge/{item['id']}"))
    assert ok(alice.get("/kb/api/v1/memory/episodes?group_id=ops")) == []
    assert ok(alice.get("/kb/api/v1/memory/facts?group_id=ops&include_invalid=true")) == []
    assert ok(alice.get(f"/kb/api/v1/memory/episodes/{mine['id']}"))["status"] == "processed"   # her own stays


def test_knowledge_bases_section_sets_what_a_new_document_kb_starts_with(client, alice, monkeypatch):
    from agenticwf.kb.runtime import features

    assert ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Plain"}), 201)["memory_config"] == {
        "enabled": False}
    has, get = features.config_has_option, features.config_get
    on = {("knowledge_bases", "wiki_enabled"): "true", ("knowledge_bases", "memory_enabled"): "yes"}
    monkeypatch.setattr(features, "config_has_option", lambda s, o: (s, o) in on or has(s, o))
    monkeypatch.setattr(features, "config_get", lambda s, o: on.get((s, o)) or get(s, o))
    assert ok(alice.get("/kb/api/v1/system/info"))["kb_defaults"] == {"wiki_enabled": True, "memory_enabled": True}
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Defaults"}), 201)
    assert kb["indexing_strategy"]["wiki_enabled"] is True and kb["memory_config"]["enabled"] is True
    off = ok(alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "Off", "indexing_strategy": {"wiki_enabled": False}, "memory_config": {"enabled": False}}), 201)
    assert off["indexing_strategy"]["wiki_enabled"] is False and off["memory_config"]["enabled"] is False
    faq = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Answers", "type": "faq"}), 201)
    assert faq["indexing_strategy"]["wiki_enabled"] is False and not faq["memory_config"].get("enabled")
