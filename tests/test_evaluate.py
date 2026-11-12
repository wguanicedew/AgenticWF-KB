"""The search eval harness (agenticwf.kb.api.evaluate) in-process: plain search, quick and deep on the
phase-2 eval set, with the hash embedder and a scripted model that asks for evidence of every
passage it saw. It checks the harness and the three paths end to end, not answer quality."""

import json
import re
from pathlib import Path

from conftest import Caller, make_client, signed, user
from agenticwf.kb.api.evaluate import EvalSet, Evaluator, summarise, table

EVAL = Path(__file__).resolve().parent.parent / "eval" / "phase2.yaml"


class EvidenceChat:
    """deep: one search_knowledge call with the question, then evidence for every cN seen.
    quick: the rewrite returns the question unchanged."""

    async def complete(self, messages, max_tokens=None, temperature=0.3, tools=None, **kw):
        usage = {"prompt_tokens": 50, "completion_tokens": 5}
        if tools and not any(m["role"] == "tool" for m in messages):
            question = next(m["content"] for m in messages if m["role"] == "user")
            return {"content": "", "tool_calls": [{"id": "", "name": "search_knowledge",
                                                   "arguments": json.dumps({"queries": [question]})}], "usage": usage}
        seen = re.findall(r'id="(c\d+)"', " ".join(m["content"] or "" for m in messages if m["role"] == "tool"))
        if any(m["role"] == "tool" for m in messages):
            return {"content": "\n".join(f'<evidence id="{h}">seen</evidence>' for h in dict.fromkeys(seen)),
                    "tool_calls": [], "usage": usage}
        return {"content": json.dumps({"rewrite_query": messages[-1]["content"].split(":", 1)[-1].strip()}),
                "tool_calls": [], "usage": usage}


def test_eval_harness(database):
    evalset = EvalSet.load(EVAL)
    assert any(len(q["expect"]) > 1 for q in evalset.questions)
    with make_client("authsvc") as client:
        fake = EvidenceChat()

        async def chat(ref):
            return fake

        client.app.state.manager.services.models.chat = chat
        ctx = user("eval", ("app:access", "knowledge:read", "knowledge:write"))
        caller = Caller(client, ctx)
        model = caller.post("/kb/api/v1/models", json={"name": "eval-chat", "type": "KnowledgeQA", "parameters": {
            "base_url": "http://127.0.0.1:9/v1", "supports_tools": True}}).json()["data"]["id"]
        ev = Evaluator(client, "/kb/api/v1", lambda method, path: signed(ctx, method, path.split("?", 1)[0]),
                       model_id=model)

        from conftest import drain
        ev.wait = lambda kb, timeout: drain(client)              # no workers in tests: run the queue
        kb, names = ev.load_corpus(evalset.corpus)
        assert sorted(names.values()) == sorted(d["name"] for d in evalset.corpus)
        outcomes = ev.run(kb, names, evalset.questions)
    summary = summarise(outcomes)
    assert set(summary) == {"search", "quick", "deep"}
    for mode, row in summary.items():
        assert row["errors"] == 0, [o.error for o in outcomes if o.error]
        assert row["questions"] == len(evalset.questions)
    assert summary["search"]["recall"] >= 0.6 and summary["search"]["input_tokens"] == 0
    assert summary["deep"]["llm_calls"] == 2 * len(evalset.questions)
    assert summary["quick"]["llm_calls"] == len(evalset.questions)
    print(table(summary))
    assert "multi_doc_f1" in table(summary) and 0 <= summary["search"]["precision"] <= 1


def test_answer_grading_summary_and_judge(tmp_path):
    """Questions with a reference answer and no expected document: judged, and kept out of the
    retrieval scores. No database."""
    from agenticwf.kb.api.evaluate import Outcome, openai_judge

    f = tmp_path / "real.yaml"
    f.write_text("questions:\n  - {q: 'Who signs the logbook?', answer: 'The shift leader, at the end of each shift.'}\n")
    [q] = EvalSet.load(f).questions
    assert q["expect"] == [] and q["answer"].startswith("The shift leader")

    rows = [Outcome("a", "deep", [], [], 1.0, correct=1.0, faithful=True),
            Outcome("b", "deep", [], [], 2.0, correct=0.0, faithful=False),
            Outcome("c", "deep", ["doc"], ["doc"], 1.0)]
    s = summarise(rows)["deep"]
    assert s["questions"] == 3 and s["correct"] == 0.5 and s["faithful"] == 0.5 and s["recall"] == 1.0
    assert "correct" in table({"deep": s})

    class FakeResponse:
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": self.content}}]}

    class FakeClient:
        def __init__(self, content): self.content, self.sent = content, None
        def post(self, url, **kw):
            self.sent = (url, kw)
            return FakeResponse(self.content)

    client = FakeClient('<think>..</think>{"correct": 0.5, "faithful": true, "reason": "misses the time"}')
    verdict = openai_judge(client, "https://gw.example.org/", "big")("q?", "ref", "ans", ["passage one"])
    assert verdict == {"correct": 0.5, "faithful": True, "reason": "misses the time"}
    assert client.sent[0] == "https://gw.example.org/chat/completions" and "passage one" in client.sent[1]["json"]["messages"][0]["content"]
    assert openai_judge(FakeClient("no json"), "u", "m")("q", "r", "a", [])["correct"] is None
