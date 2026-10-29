"""`agenticwf kb` through the `agenticwf` command (agenticwf-auth-client), against a mock KB.
Needs both client packages installed (the entry point): pip install -e ../AgenticWF-auth/cli -e client"""

import json

import httpx
import pytest

cli = pytest.importorskip("agenticwf_auth_cli.__main__")

KBS = [{"id": "k1", "name": "Docs", "type": "document", "ref": "alice:docs"}]
HITS = [{"score": 0.91, "knowledge_title": "Run 42", "knowledge_base_id": "k1", "content": "beam  energy\n10 GeV"}]
ANSWER = {"answer": "It doubled <ref>c1</ref>.", "sources": [{"n": 1, "citation_text": "Run 42", "url": None}]}


@pytest.fixture
def kb(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("Authorization") != "Bearer kb-token":
            return httpx.Response(401, json={"success": False, "error": {"code": 401, "message": "bad token"}})
        routes = {("GET", "/kb/api/v1/knowledge-bases"): KBS, ("POST", "/kb/api/v1/knowledge-search"): HITS,
                  ("POST", "/kb/api/v1/agentic-search"): ANSWER}
        data = routes.get((request.method, request.url.path))
        if data is None:
            return httpx.Response(404, json={"success": False, "error": {"code": 404, "message": "not found"}})
        return httpx.Response(200, json={"success": True, "data": data})

    class MockClient(httpx.Client):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setenv("AGENTICWF_KB_URL", "http://kb.test/kb/")
    monkeypatch.setenv("AGENTICWF_KB_TOKEN", "kb-token")
    return seen


def test_list_and_search(kb, capsys):
    assert cli.main(["kb", "list"]) == 0
    assert capsys.readouterr().out.strip() == "k1  Docs  (document, alice:docs)"

    assert cli.main(["kb", "search", "beam energy", "--kb", "k1", "--top-k", "3"]) == 0
    assert json.loads(kb[-1].content) == {"query": "beam energy", "knowledge_base_ids": ["k1"], "top_k": 3}
    assert "[1] 0.910  Run 42  (kb k1)\n    beam energy 10 GeV" in capsys.readouterr().out


def test_ask_and_json(kb, capsys):
    assert cli.main(["kb", "ask", "what changed?", "--preset", "quick"]) == 0
    body = json.loads(kb[-1].content)
    assert body["output"] == "answer" and body["preset"] == "quick" and body["knowledge_base_ids"] is None
    out = capsys.readouterr().out
    assert out.startswith("It doubled") and "[1] Run 42" in out

    assert cli.main(["kb", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == KBS


def test_errors(kb, monkeypatch, capsys):
    assert cli.main(["kb", "call", "GET", "/nothing"]) == 1
    assert "HTTP 404" in capsys.readouterr().err
    monkeypatch.setenv("AGENTICWF_KB_TOKEN", "wrong")
    assert cli.main(["kb", "list"]) == 1
    monkeypatch.delenv("AGENTICWF_KB_TOKEN")
    for name in ("AGENTICWF_AUTH_URL", "AGENTICWF_API_URL", "AGENTICWF_CLIENT_ID", "AGENTICWF_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    assert cli.main(["kb", "list"]) == 2                      # neither a KB token nor a Logto sign-in
    assert "missing configuration" in capsys.readouterr().err
