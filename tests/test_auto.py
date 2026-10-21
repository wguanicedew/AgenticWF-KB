"""The `auto` preset: route → gather in parallel (documents, wiki, records, memory) → merge."""

import io
import json
import re

import pytest

from conftest import Caller, drain, make_client, user
from test_agentic import chat_model
from test_records import EXPERIMENT_RUN

RW = ("app:access", "knowledge:read", "knowledge:write")


class AutoLLM:
    def __init__(self, route):
        self.route = route
        self.calls = []

    async def complete(self, messages, max_tokens=None, temperature=0.3, tools=None, **kw):
        system = messages[0]["content"]
        self.calls.append({"system": system[:60], "messages": messages, "tools": tools})
        if "You route one question" in system:
            return self._reply(json.dumps(self.route))
        if "structured records only" in system:
            if not any(m["role"] == "tool" for m in messages):
                return self._reply("", [{"id": "", "name": "query_records", "arguments": json.dumps({
                    "record_type": "experiment_run", "sort": [{"field": "metrics.accuracy", "dir": "desc"}],
                    "limit": 1})}])
            return self._reply('The best run is r3 with accuracy 0.86 <ref id="r1"/>.')
        if "evidence gathered from several parts" in system:
            text = messages[1]["content"]
            refs = [r for r in re.findall(r'id="([cpr]\d+)"', text)]
            return self._reply("Answer: " + " ".join(f'<ref id="{r}"/>' for r in dict.fromkeys(refs)))
        return self._reply("{}")

    @staticmethod
    def _reply(content, calls=()):
        return {"content": content, "tool_calls": list(calls), "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "finish_reason": "stop"}


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


def ok(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def test_auto_routes_gathers_and_merges(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    model = chat_model(alice)
    docs = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs", "indexing_strategy": {
        "wiki_enabled": True}}), 201)["id"]
    alice.post(f"/kb/api/v1/knowledge-bases/{docs}/knowledge/file", files={"file": ("pilot.md", io.BytesIO(
        b"# Pilot\n\nThe pilot retries stage-out five times since 3.7.2.\n"), "text/markdown")})
    ok(alice.post(f"/kb/api/v1/knowledgebase/{docs}/wiki/pages", json={
        "title": "Pilot", "page_type": "entity", "content": "The pilot runs jobs on worker nodes.",
        "summary": "The job wrapper."}), 201)
    ok(alice.post("/kb/api/v1/record-types", json=EXPERIMENT_RUN), 201)
    runs = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Runs", "type": "records",
                                                             "records_config": {"default_type": "experiment_run"}}), 201)["id"]
    ok(alice.post(f"/kb/api/v1/knowledge-bases/{runs}/records", json={"records": [
        {"run_id": "r3", "model": "vit", "dataset": "cifar100", "metrics": {"accuracy": 0.86}},
        {"run_id": "r1", "model": "resnet", "dataset": "cifar100", "metrics": {"accuracy": 0.81}}]}))
    drain(client)

    async def remembered(auth, body):
        return [{"id": "f1", "fact": "Alice prefers the pilot logs over dashboards.", "valid_at": "2026-01-01"}]

    client.app.state.manager.services.memory.search = remembered
    fake = AutoLLM({"sources": ["documents", "wiki", "records", "memory"], "queries": ["pilot stage-out retries"],
                    "global": False, "records_question": "Which run is the most accurate?", "reason": "mixed"})

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={
        "query": "How often does the pilot retry stage-out, and which run is the most accurate?",
        "preset": "auto", "knowledge_base_ids": [docs, runs], "model_id": model}))
    route = data["route"]
    assert route["sources"] == ["documents", "wiki", "records", "memory"]
    assert route["counts"]["documents"] >= 1 and route["counts"]["wiki"] == 1 and route["counts"]["memory"] == 1
    assert "five times" in [c for c in fake.calls if "evidence gathered" in c["system"]][0]["messages"][1]["content"]
    assert set(route["available"]) == {"documents", "wiki", "records", "memory"}
    merge = [c for c in fake.calls if "evidence gathered" in c["system"]][0]["messages"][1]["content"]
    for part in ('<source type="wiki">', '<source type="records">', '<source type="memory"', "<document "):
        assert part in merge
    assert "five times" in merge and "accuracy 0.86" in merge and "pilot logs" in merge
    assert "<kb doc=" in data["answer"] and "<wiki slug=" in data["answer"] and "<record " in data["answer"]
    assert {e["kind"] for e in data["evidence"]} >= {"chunk", "wiki_page", "record"}
    names = [t["name"] for t in data["steps"][0]["tool_calls"]]
    assert names == ["auto:documents", "auto:wiki", "auto:records", "auto:memory"]
    routes = [c for c in fake.calls if "You route one question" in c["system"]]
    assert len(routes) == 1 and "records:" in routes[0]["messages"][0]["content"]


def test_auto_bad_route_falls_back_and_evidence_output(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    model = chat_model(alice)
    docs = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}), 201)["id"]
    alice.post(f"/kb/api/v1/knowledge-bases/{docs}/knowledge/file", files={"file": ("pilot.md", io.BytesIO(
        b"# Pilot\n\nThe pilot retries stage-out five times.\n"), "text/markdown")})
    drain(client)
    fake = AutoLLM("not json at all")
    fake.route = None

    async def complete(messages, **kw):
        return AutoLLM._reply("I will not answer in JSON")

    fake.complete = complete

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "pilot stage-out retries", "preset": "auto",
                                                            "output": "evidence", "knowledge_base_ids": [docs],
                                                            "model_id": model}))
    assert data["route"]["reason"] == "default route" and data["route"]["sources"] == ["documents"]
    assert data["stopped"] == "evidence_complete" and data["evidence"][0]["kind"] == "chunk"
    presets = {p["name"]: p for p in ok(alice.get("/kb/api/v1/agentic-search/presets"))}
    assert presets["auto"]["engine"] == "router"


def test_auto_discards_ungrounded_records_findings(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    model = chat_model(alice)
    ok(alice.post("/kb/api/v1/record-types", json=EXPERIMENT_RUN), 201)
    runs = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Runs", "type": "records",
                                                             "records_config": {"default_type": "experiment_run"}}), 201)["id"]
    fake = AutoLLM({"sources": ["records"], "queries": ["best optimizer"], "records_question": "best optimizer"})

    async def complete(messages, max_tokens=None, temperature=0.3, tools=None, **kw):
        system = messages[0]["content"]
        fake.calls.append({"system": system[:60], "messages": messages, "tools": tools})
        if "You route one question" in system:
            return AutoLLM._reply(json.dumps(fake.route))
        if "structured records only" in system:          # answers from imagination, no tool call
            return AutoLLM._reply("Adam reached 82.1% accuracy.")
        return AutoLLM._reply("Nothing found.")

    fake.complete = complete

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "best optimizer?", "preset": "auto",
                                                            "knowledge_base_ids": [runs], "model_id": model}))
    assert data["stopped"] == "no_results" and data["route"]["counts"]["records"] == 0
    assert data["steps"][0]["tool_calls"][0]["result_summary"] == "discarded: no records tool returned data"
    assert not any("82.1" in json.dumps(c["messages"]) for c in fake.calls if "several parts" in c["system"])
