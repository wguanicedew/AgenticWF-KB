"""Agentic search (docs/agentic-search.md) with a scripted chat model: the loop, tools,
citations, evidence, limits, the quick preset with FAQ priority, scope and SSE."""

import asyncio
import io
import json

import pytest

from conftest import Caller, drain, make_client, user

RW = ("app:access", "knowledge:read", "knowledge:write")

DOC = (b"# Pilot 3.7 release notes\n\n"
       b"Stage-out retries were raised from 3 to 5 in pilot 3.7.2 because of storage timeouts.\n\n"
       b"## Other changes\n\nThe heartbeat interval is now 30 minutes. Error E-2201 means the copy tool failed.\n")


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


class ScriptedChat:
    """Replies from a script: each step is a dict (content / tool_calls) or a function of the
    messages returning one. Records every call."""

    def __init__(self, script, delay=0.0):
        self.script = list(script)
        self.calls = []
        self.delay = delay

    async def complete(self, messages, max_tokens=None, temperature=0.3, tools=None, **kw):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools, "temperature": temperature})
        if self.delay:
            await asyncio.sleep(self.delay)
        step = self.script.pop(0) if self.script else {"content": "done"}
        if callable(step):
            step = step(messages)
        calls = [{"id": c.get("id", ""), "name": c["name"], "arguments": c["arguments"]}
                 for c in step.get("tool_calls", [])]
        return {"content": step.get("content", ""), "reasoning": step.get("reasoning", ""), "tool_calls": calls,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "finish_reason": "stop"}


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def script(client, steps, delay=0.0):
    fake = ScriptedChat(steps, delay)

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    return fake


def chat_model(caller, **params):
    return caller.post("/kb/api/v1/models", json={"name": "fake-chat", "type": "KnowledgeQA", "parameters": {
        "base_url": "http://127.0.0.1:9/v1", "supports_tools": True, **params}}).json()["data"]["id"]


def setup_docs(client, caller):
    kb_id = caller.post("/kb/api/v1/knowledge-bases", json={"name": "Pilot docs"}).json()["data"]["id"]
    doc = caller.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/file",
                      files={"file": ("pilot.md", io.BytesIO(DOC), "text/markdown")}).json()["data"]
    drain(client)
    return kb_id, doc["id"]


def agentic(caller, **body):
    response = caller.post("/kb/api/v1/agentic-search", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def tool_output(fake, index):
    """The tool messages the model saw in call `index`."""
    return [m["content"] for m in fake.calls[index]["messages"] if m["role"] == "tool"]


# --- the loop -----------------------------------------------------------------------------------------
def test_deep_answer_with_citations(client, alice):
    model = chat_model(alice)
    kb_id, doc_id = setup_docs(client, alice)
    fake = script(client, [
        {"content": "Searching first.", "tool_calls": [call("search_knowledge", queries=["stage-out retries"])]},
        {"content": 'Retries went from 3 to 5 in 3.7.2 <ref id="c1"/>, not 7 <ref id="c99"/>.'}])
    data = agentic(alice, query="Why were stage-out retries changed?", knowledge_base_ids=[kb_id], model_id=model)
    assert data["stopped"] == "answered"
    assert '<kb doc="pilot"' in data["answer"] and "c99" not in data["answer"]
    assert "<ref" not in data["answer"]
    first = data["evidence"][0]
    assert first["handle"] == "c1" and first["cited"] is True and first["knowledge_id"] == doc_id
    assert data["steps"][0]["thought"] == "Searching first."
    assert data["steps"][0]["tool_calls"][0]["status"] == "ok" and data["steps"][-1]["final"]
    assert data["usage"] == {"model_id": model, "input_tokens": 200, "output_tokens": 20, "llm_calls": 2}
    system = fake.calls[0]["messages"][0]["content"]
    assert '<knowledge_base id="b1" name="Pilot docs"' in system and "search_knowledge" in system
    assert fake.calls[0]["temperature"] == 0.7 and fake.calls[0]["tools"]
    assert '<result id="c1" doc="d1"' in tool_output(fake, 1)[0]


def test_tool_errors_and_parallel_calls(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    fake = script(client, [
        {"tool_calls": [call("grep_knowledge", pattern=r"E-\d{4}"), call("no_such_tool"),
                        {"name": "read_document", "arguments": "not json"}, call("read_document", id="d9"),
                        call("search_knowledge", queries=["x"], knowledge_base_ids=["b7"])]},
        {"content": "Error E-2201 is a copy-tool failure."}])
    data = agentic(alice, query="What is E-2201?", knowledge_base_ids=[kb_id], model_id=model)
    statuses = [c["status"] for c in data["steps"][0]["tool_calls"]]
    assert statuses == ["ok", "error", "error", "error", "error"]
    outputs = tool_output(fake, 1)
    assert "<match id=\"c1\"" in outputs[0] and "E-2201" in outputs[0]
    assert "unknown tool" in outputs[1] and "grep_knowledge" in outputs[1]
    assert "JSON object" in outputs[2] and "unknown handle" in outputs[3] and "unknown handle" in outputs[4]
    ids = [c["id"] for c in data["steps"][0]["tool_calls"]]
    assert len(set(ids)) == 5


def test_read_document_tool(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    fake = script(client, [
        {"tool_calls": [call("list_documents")]},
        {"tool_calls": [call("read_document", id="d1"), call("read_document", id="d1", query="heartbeat"),
                        call("read_document", id="d1", offset=10, limit=200)]},
        {"content": "ok"}])
    agentic(alice, query="heartbeat interval", knowledge_base_ids=[kb_id], model_id=model)
    listing = tool_output(fake, 1)[0]
    assert '<document id="d1" kb="b1" title="pilot"' in listing and 'status="live"' in listing
    whole, window, paged = tool_output(fake, 2)[1:]
    assert '<passage id="c' in whole and "Stage-out retries" in whole and 'next_offset=""' in whole
    assert 'windows="1"' in window and "heartbeat" in window
    assert 'offset="10"' in paged


def test_forced_final_at_iteration_limit(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    fake = script(client, [{"tool_calls": [call("thinking", thought="plan")]},
                           {"tool_calls": [call("todo_write", todos=[{"content": "find it", "status": "in_progress"}])]},
                           {"content": "Best effort answer."}])
    data = agentic(alice, query="q", knowledge_base_ids=[kb_id], model_id=model, max_iterations=2)
    assert data["stopped"] == "max_iterations" and data["answer"].startswith("Best effort answer.")
    assert data["answer"].endswith("Sources: none — nothing relevant was found in the knowledge bases.")
    assert data["sources"] == []
    assert data["steps"][-1]["forced"] is True and fake.calls[-1]["tools"] is None
    assert "step limit" in fake.calls[-1]["messages"][-1]["content"]
    assert "[~] find it" in tool_output(fake, 2)[1]
    assert alice.post("/kb/api/v1/agentic-search", json={"query": "q", "max_iterations": 21}).status_code == 400


def test_evidence_output(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    fake = script(client, [{"tool_calls": [call("search_knowledge", queries=["retries"], mode="keyword")]},
                           {"content": '<evidence id="c1">gives the new retry count</evidence>\n'
                                       '<evidence id="c42">invented</evidence>'}])
    data = agentic(alice, query="retries", knowledge_base_ids=[kb_id], model_id=model, output="evidence")
    assert data["stopped"] == "evidence_complete" and "answer" not in data
    assert [(e["handle"], e["reason"]) for e in data["evidence"]] == [("c1", "gives the new retry count")]
    assert "<evidence" in fake.calls[0]["messages"][0]["content"]


def test_context_trimming(client, alice):
    model = chat_model(alice, context_window=300)              # budget: 675 characters
    kb_id, _ = setup_docs(client, alice)
    fake = script(client, [{"tool_calls": [call("search_knowledge", queries=["retries"])]},
                           {"tool_calls": [call("search_knowledge", queries=["heartbeat"])]},
                           {"content": "ok"}])
    response = alice.post("/kb/api/v1/agentic-search", json={"query": "q", "knowledge_base_ids": [kb_id],
                                                             "model_id": model, "stream": True})
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert any(f["response_type"] == "context_compacted" for f in frames)
    assert "earlier tool result omitted" in tool_output(fake, 2)[0]


# --- limits ---------------------------------------------------------------------------------------------
def test_model_limits_busy_and_timeout(client, alice):
    plain = alice.post("/kb/api/v1/models", json={"name": "no-tools", "type": "KnowledgeQA", "parameters": {
        "base_url": "http://127.0.0.1:9/v1"}}).json()["data"]["id"]
    kb_id, _ = setup_docs(client, alice)
    assert alice.post("/kb/api/v1/agentic-search", json={"query": "q", "model_id": plain}).status_code == 400
    settings = client.app.state.manager.services.agentic.settings
    old = settings.per_owner, settings.timeout
    try:
        settings.per_owner = 0
        busy = alice.post("/kb/api/v1/agentic-search", json={"query": "q", "model_id": plain, "preset": "quick"})
        assert busy.status_code == 429
        settings.per_owner, settings.timeout = 2, 0.2
        script(client, [{"content": "{}"}, {"content": "late"}], delay=0.15)
        data = agentic(alice, query="retries", knowledge_base_ids=[kb_id], model_id=plain, preset="quick")
        assert data["stopped"] == "timeout" and "answer" not in data
    finally:
        settings.per_owner, settings.timeout = old
    assert client.app.state.manager.services.agentic._running == {}


# --- quick ------------------------------------------------------------------------------------------------
def test_quick_puts_faq_first_and_marks_exact(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    faq = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Help", "type": "faq"}).json()["data"]["id"]
    alice.post(f"/kb/api/v1/knowledge-bases/{faq}/faq/entries", json={"entries": [
        {"standard_question": "How many stage-out retries does the pilot make?", "answers": ["Five, since 3.7.2."]}]})
    fake = script(client, [{"content": '{"rewrite_query": "How many stage-out retries does the pilot make?", '
                                       '"intent": "lookup"}'},
                           {"content": 'Five <ref id="c1"/>.'}])
    data = agentic(alice, query="how many stage out retries does the pilot make", preset="quick", model_id=model,
                   knowledge_base_ids=[kb_id, faq])
    assert data["stopped"] == "answered" and "<kb " in data["answer"]
    context = fake.calls[1]["messages"][1]["content"]
    assert context.index('<source type="faq" priority="high">') < context.index("<document ")
    assert '<context id="c1" match="exact">' in context
    assert data["steps"][0]["tool_calls"][0]["args"] == {"queries": ["How many stage-out retries does the pilot make?"]}
    assert fake.calls[0]["tools"] is None and len(fake.calls) == 2


def test_quick_no_results(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    script(client, [{"content": "not json"}])
    data = agentic(alice, query="zzzqqq", preset="quick", output="evidence", model_id=model,
                   knowledge_base_ids=[kb_id])
    assert data["stopped"] == "no_results" and data["evidence"] == []


# --- scope, presets, SSE ------------------------------------------------------------------------------------
def test_scope_and_presets(client, alice):
    model = chat_model(alice)
    kb_id, doc_id = setup_docs(client, alice)
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.post("/kb/api/v1/agentic-search", json={"query": "q", "knowledge_base_ids": [kb_id],
                                                       "preset": "quick"}).status_code == 404
    assert alice.post("/kb/api/v1/agentic-search", json={"query": "q", "knowledge_ids": ["nope"],
                                                         "model_id": model}).status_code == 404
    fake = script(client, [{"tool_calls": [call("search_knowledge", queries=["stage-out retries"])]},
                           {"content": "nothing"}])
    bob_model = chat_model(bob)
    data = Caller(client, bob.ctx).post("/kb/api/v1/agentic-search", json={"query": "stage-out retries",
                                                                           "model_id": bob_model}).json()["data"]
    assert "Pilot docs" not in fake.calls[0]["messages"][0]["content"]
    assert data["evidence"] == [] and "No results" in tool_output(fake, 1)[0]
    presets = {p["name"]: p for p in alice.get("/kb/api/v1/agentic-search/presets").json()["data"]}
    assert presets["deep"]["tools"] == ["wiki_search", "wiki_read_page", "wiki_global_search", "search_knowledge",
                                        "grep_knowledge", "list_documents", "read_document", "describe_record_types",
                                        "query_records", "aggregate_records", "find_similar_records", "recall",
                                        "memory_history", "thinking", "todo_write", "suggest_initial_points"]
    assert presets["quick"]["tools"] == [] and presets["wiki"]["available"] is True     # [features] wiki in tests


def test_sse_frames(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    script(client, [{"reasoning": "hmm", "tool_calls": [call("search_knowledge", queries=["heartbeat"])]},
                    {"content": "Every 30 minutes " + "x" * 300 + ' <ref id="c1"/>'}])
    response = alice.post("/kb/api/v1/agentic-search", json={"query": "heartbeat", "knowledge_base_ids": [kb_id],
                                                             "model_id": model, "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    kinds = [f["response_type"] for f in frames]
    assert kinds[0] == "agent_query" and kinds[-1] == "complete" and frames[-1]["done"] is True
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("references") < kinds.index("answer")
    assert "thinking" in kinds and kinds.count("answer") >= 2
    answer = "".join(f["content"] for f in frames if f["response_type"] == "answer")
    assert answer == frames[-1]["data"]["answer"] and "<kb " in answer
    assert frames[-1]["finish_reason"] == "answered"


def test_switches_citations_and_fixed_fallback(client, alice):
    model = chat_model(alice)
    kb_id, _ = setup_docs(client, alice)
    settings = client.app.state.manager.services.agentic.settings
    old = settings.citations, settings.fallback_strategy, settings.quick_rewrite
    try:
        settings.citations, settings.fallback_strategy, settings.quick_rewrite = False, "fixed", False
        fake = script(client, [{"content": 'Five <ref id="c1"/>.'}])
        data = agentic(alice, query="stage-out retries", preset="quick", model_id=model, knowledge_base_ids=[kb_id])
        assert data["answer"].startswith('Five <ref id="c1"/>.') and len(fake.calls) == 1        # no rewrite call
        assert "(consulted, not cited inline)" in data["answer"]           # no citation protocol: still sourced
        assert "Cite with exactly" not in fake.calls[0]["messages"][0]["content"]
        data = agentic(alice, query="zzzqqq", preset="quick", model_id=model, knowledge_base_ids=[kb_id])
        assert data["stopped"] == "no_results" and data["answer"].startswith("Sorry")
    finally:
        settings.citations, settings.fallback_strategy, settings.quick_rewrite = old
