"""Web search (docs/web-search.md): the eight providers, DuckDuckGo's HTML, de-duplication, pages
as markdown through the parsers, the per-run cache, provider management with encrypted keys,
/web-search and /web-fetch, the `smart` preset, write tools and regenerate-summary."""

import asyncio
import http.server
import io
import json
import threading

import httpx
import pytest

from conftest import Caller, drain, make_client, user
from kb_runtime.websearch.fetch import FetchCache, fetch_page
from kb_runtime.websearch.providers import Provider, WebResult, canonical_url, dedupe, parse_duckduckgo
from test_agentic import ScriptedChat, call, chat_model
from test_citations import pdf

RW = ("app:access", "knowledge:read", "knowledge:write")


def run(coro):
    return asyncio.run(coro)


# --- providers ------------------------------------------------------------------------------------------
DDG_HTML = """<div class="result"><a rel="nofollow" class="result__a"
  href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frucio.cern.ch%2Fdocs%3Fx%3D1&amp;rut=abc">Rucio <b>docs</b></a>
  <a class="result__snippet" href="#">Rucio is a <b>data</b> manager.</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">An ad</a></div>
<div class="result"><a class="result__a" href="https://fts.web.cern.ch/">FTS</a>
  <div class="result__snippet">File transfer service.</div></div>"""


def test_duckduckgo_html_and_dedupe():
    results = parse_duckduckgo(DDG_HTML)
    assert [r.url for r in results] == ["https://rucio.cern.ch/docs?x=1", "https://fts.web.cern.ch/"]
    assert canonical_url("https://www.Rucio.cern.ch/docs/?utm_source=x#top") == canonical_url("https://rucio.cern.ch/docs")
    merged = dedupe([WebResult("A", "https://x.org/p/"), WebResult("A2", "https://www.x.org/p?utm_medium=y", content="text")])
    assert len(merged) == 1 and merged[0].content == "text"


@pytest.mark.parametrize("kind,params,expect", [
    ("duckduckgo", {}, ("POST", "/html/")),
    ("searxng", {"base_url": "http://searx.test"}, ("GET", "/search")),
    ("tavily", {"api_key": "k"}, ("POST", "/search")),
    ("brave", {"api_key": "k"}, ("GET", "/res/v1/web/search")),
    ("bing", {"api_key": "k"}, ("GET", "/v7.0/search")),
    ("google", {"api_key": "k", "cx": "engine"}, ("GET", "/customsearch/v1")),
    ("exa", {"api_key": "k"}, ("POST", "/search")),
    ("bocha", {"api_key": "k"}, ("POST", "/v1/web-search")),
])
def test_every_provider(kind, params, expect):
    seen = []
    answers = {
        "duckduckgo": DDG_HTML,
        "searxng": {"results": [{"title": "T", "url": "https://a.org/1", "content": "s"}]},
        "tavily": {"results": [{"title": "T", "url": "https://a.org/1", "content": "s", "raw_content": "full"}]},
        "brave": {"web": {"results": [{"title": "T", "url": "https://a.org/1", "description": "s"}]}},
        "bing": {"webPages": {"value": [{"name": "T", "url": "https://a.org/1", "snippet": "s"}]}},
        "google": {"items": [{"title": "T", "link": "https://a.org/1", "snippet": "s"}]},
        "exa": {"results": [{"title": "T", "url": "https://a.org/1", "text": "full text"}]},
        "bocha": {"data": {"webPages": {"value": [{"name": "T", "url": "https://a.org/1", "snippet": "s",
                                                   "summary": "sum"}]}}},
    }

    def handler(request):
        seen.append(request)
        answer = answers[kind]
        return httpx.Response(200, text=answer) if isinstance(answer, str) else httpx.Response(200, json=answer)

    results = run(Provider(kind, params, transport=httpx.MockTransport(handler)).search("rucio", 3))
    assert (seen[0].method, seen[0].url.path) == expect and results and results[0].provider == kind
    headers = seen[0].headers
    if kind == "bing":
        assert headers["Ocp-Apim-Subscription-Key"] == "k"
    if kind == "exa":
        assert headers["x-api-key"] == "k" and results[0].content == "full text"
    if kind == "google":
        assert seen[0].url.params["cx"] == "engine"
    if kind in ("tavily", "bocha"):
        assert headers["Authorization"] == "Bearer k" and results[0].content
    if kind in ("tavily", "brave", "bing", "google", "exa", "bocha"):
        from kb_runtime.websearch.providers import WebSearchError
        with pytest.raises(WebSearchError):
            run(Provider(kind, {k: v for k, v in params.items() if k != "api_key"},
                         transport=httpx.MockTransport(handler)).search("x"))


# --- pages ----------------------------------------------------------------------------------------------
class Site(http.server.BaseHTTPRequestHandler):
    version = 1

    def do_GET(self):
        if self.path.startswith("/report.pdf"):
            body, kind = pdf(["Beam report page one", "Luminosity was 2e34 on page two"], {"Title": "Beam report"}), \
                "application/pdf"
        elif self.path.startswith("/long"):
            body, kind = (f"<html><head><title>Long v{Site.version}</title></head><body><article>"
                          + "".join(f"<p>Paragraph {i} of version {Site.version}.</p>" for i in range(200))
                          + "</article></body></html>").encode(), "text/html"
        else:
            body, kind = (b"<html><head><title>Rucio</title></head><body><nav>menu</nav><article><h1>Rucio</h1>"
                          b"<p>Rucio places data on storage sites with replication rules.</p></article></body></html>"), \
                "text/html"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_pages_become_markdown_and_the_cache_keeps_one_version(database, site):
    page = run(fetch_page(f"{site}/rucio"))
    assert page.title == "Rucio" and "replication rules" in page.markdown and "<p>" not in page.markdown
    report = run(fetch_page(f"{site}/report.pdf"))
    assert report.title == "Beam report" and "Luminosity was 2e34" in report.markdown

    async def two_reads():
        cache = FetchCache()
        first = await cache.get(f"{site}/long")
        Site.version = 2                         # the page changes between two reads
        second = await cache.get(f"{site}/long")
        fresh = await FetchCache().get(f"{site}/long")
        return first, second, fresh

    first, second, fresh = run(two_reads())
    assert first is second and "version 1" in second.markdown and "version 2" in fresh.markdown
    Site.version = 1


# --- the service and the API ------------------------------------------------------------------------------
@pytest.fixture
def client(database, monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("KB_SECRETS_KEY", Fernet.generate_key().decode())
    with make_client("authsvc") as c:
        yield c


def ok(response, status=200):
    assert response.status_code in (status, 201, 202), response.text
    return response.json()["data"]


def mock_search(client, site, extra=None):
    """Providers answer from a mock; their result URLs point at the local site."""
    def handler(request):
        if request.url.path == "/html/":
            return httpx.Response(200, text=f'<a class="result__a" href="{site}/rucio">Rucio</a>'
                                            f'<a class="result__snippet">snippet</a>'
                                            f'<a class="result__a" href="{site}/report.pdf">Report</a>')
        return httpx.Response(200, json={"web": {"results": [
            {"title": "Rucio again", "url": f"{site}/rucio/?utm_source=brave", "description": "dup"},
            {"title": "Long", "url": f"{site}/long", "description": "long page"}]}})
    client.app.state.manager.services.web.transport = httpx.MockTransport(handler)


def test_providers_api_search_and_fetch(client, site):
    alice = Caller(client, user("alice", RW, org="eic"))
    kinds = {k["kind"] for k in ok(alice.get("/kb/api/v1/web-search/providers"))}
    assert kinds == {"duckduckgo", "searxng", "tavily", "brave", "bing", "google", "exa", "bocha"}
    assert alice.post("/kb/api/v1/web-search-providers", json={"name": "b", "kind": "brave"}).status_code == 400
    ddg = ok(alice.post("/kb/api/v1/web-search-providers", json={"name": "ddg", "kind": "duckduckgo",
                                                                 "is_default": True}))
    brave = ok(alice.post("/kb/api/v1/web-search-providers", json={"name": "brave", "kind": "brave",
                                                                   "parameters": {"api_key": "secret-key-1234"}}))
    assert brave["parameters"]["api_key"].startswith("******") and brave["parameters"]["api_key"].endswith("1234")
    from conftest import sql_rows
    stored = sql_rows("SELECT parameters->>'api_key' FROM kb_web_search_providers WHERE id = :i", i=brave["id"])[0][0]
    assert "secret-key" not in stored                                       # encrypted at rest
    ok(alice.put(f"/kb/api/v1/web-search-providers/{brave['id']}/credentials", json={"api_key": "other-key-9999"}))
    assert ok(alice.get(f"/kb/api/v1/web-search-providers/{brave['id']}"))["parameters"]["api_key"].endswith("9999")
    mock_search(client, site)
    tested = ok(alice.post(f"/kb/api/v1/web-search-providers/{brave['id']}/test", json={"query": "rucio"}))
    assert tested["ok"] and tested["results"]
    unsaved = ok(alice.post("/kb/api/v1/web-search-providers/test", json={"kind": "google", "parameters": {}}))
    assert unsaved["ok"] is False and "needs" in unsaved["error"]                       # cx / api key

    # the default provider; text of the top results is fetched (HTML and PDF)
    found = ok(alice.post("/kb/api/v1/web-search", json={"query": "rucio", "fetch_top": 2}))
    assert found["providers"] == ["duckduckgo"] and [r["title"] for r in found["results"]] == ["Rucio", "Report"]
    assert "replication rules" in found["results"][0]["content"] and "Luminosity" in found["results"][1]["content"]
    # two providers merged; the same page from both is kept once
    both = ok(alice.post("/kb/api/v1/web-search", json={"query": "rucio", "provider_ids": [ddg["id"], brave["id"]],
                                                        "fetch_top": 0}))
    assert len([r for r in both["results"] if "/rucio" in r["url"]]) == 1 and len(both["results"]) == 3
    page = ok(alice.post("/kb/api/v1/web-fetch", json={"url": f"{site}/long", "offset": 0, "limit": 500,
                                                       "cache_key": "task-1"}))
    assert page["title"] == "Long v1" and page["next_offset"] == 500 and page["returned_chars"] == 500
    Site.version = 2                                     # the page changes between two reads of one task
    later = ok(alice.post("/kb/api/v1/web-fetch", json={"url": f"{site}/long", "offset": 500, "cache_key": "task-1"}))
    other = ok(alice.post("/kb/api/v1/web-fetch", json={"url": f"{site}/long", "cache_key": "task-2"}))
    Site.version = 1
    assert "version 1" in later["content"] and other["title"] == "Long v2"
    # another user sees neither provider
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert ok(bob.get("/kb/api/v1/web-search-providers")) == []
    assert bob.get(f"/kb/api/v1/web-search-providers/{brave['id']}").status_code == 404
    assert bob.post("/kb/api/v1/web-search", json={"query": "x"}).status_code == 400      # no provider for bob
    ok(alice.delete(f"/kb/api/v1/web-search-providers/{brave['id']}"))
    assert alice.get(f"/kb/api/v1/web-search-providers/{brave['id']}").status_code == 404


def test_smart_preset_web_and_kb_with_citations(client, site):
    alice = Caller(client, user("alice", RW, org="eic"))
    ok(alice.post("/kb/api/v1/web-search-providers", json={"name": "ddg", "kind": "duckduckgo", "is_default": True}))
    mock_search(client, site)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}))["id"]
    alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file", files={"file": ("ops.md", io.BytesIO(
        b"# Ops\n\nOur Rucio instance runs rule evaluation every hour.\n"), "text/markdown")})
    drain(client)
    model = chat_model(alice)
    fake = ScriptedChat([
        {"tool_calls": [call("search_knowledge", queries=["rucio rule evaluation"]),
                        call("web_search", query="rucio replication rules", fetch_top=1)]},
        {"tool_calls": [call("web_fetch", id="w1", limit=300)]},
        {"content": 'Rules are evaluated hourly here <ref id="c1"/>; Rucio places data with rules <ref id="w1"/>.'}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "how does rucio place data and how often here?",
                                                            "preset": "smart", "knowledge_base_ids": [kb],
                                                            "model_id": model}))
    tools = [t["function"]["name"] for t in fake.calls[0]["tools"]]
    assert "web_search" in tools and "web_fetch" in tools and "wiki_write_page" not in tools
    outputs = [m["content"] for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert '<web_result id="w1" title="Rucio"' in outputs[1] and "replication rules" in outputs[1]
    page = [m["content"] for m in fake.calls[2]["messages"] if m["role"] == "tool"][-1]
    assert '<web_page id="w1"' in page
    kinds = {s["kind"] for s in data["sources"]}
    assert kinds == {"chunk", "web_page"}
    web = [s for s in data["sources"] if s["kind"] == "web_page"][0]
    assert web["url"] == f"{site}/rucio" and "(accessed " in web["citation_text"]
    assert '<web title="Rucio"' in data["answer"]


def test_write_tools_are_opt_in(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    model = chat_model(alice)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Wiki", "indexing_strategy": {"wiki_enabled": True}}))["id"]
    reader = Caller(client, user("rita", ("app:access", "knowledge:read"), org="eic"))
    assert reader.post("/kb/api/v1/agentic-search", json={"query": "q", "allow_writes": True,
                                                          "model_id": model}).status_code == 403
    fake = ScriptedChat([
        {"tool_calls": [call("wiki_write_page", title="Rucio vs FTS", content="Rucio decides, FTS copies.",
                             page_type="comparison"),
                        call("remember", content="Alice prefers comparisons as tables.")]},
        {"tool_calls": [call("wiki_replace_text", slug="comparison/rucio-vs-fts", old_text="FTS copies",
                             new_text="FTS copies the files"),
                        call("wiki_flag_issue", slug="comparison/rucio-vs-fts", issue_type="other",
                             description="needs sources")]},
        {"content": "I wrote the comparison page."}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "write a comparison page", "preset": "wiki",
                                                            "knowledge_base_ids": [kb], "model_id": model,
                                                            "allow_writes": True}))
    assert sorted(w["tool"] for w in data["writes"]) == ["remember", "wiki_flag_issue", "wiki_replace_text",
                                                         "wiki_write_page"]       # one turn's calls run in parallel
    assert "You may also change the knowledge service" in fake.calls[0]["messages"][0]["content"]
    page = ok(alice.get(f"/kb/api/v1/knowledgebase/{kb}/wiki/pages/comparison/rucio-vs-fts"))
    assert page["content"] == "Rucio decides, FTS copies the files." and page["version"] == 2
    assert ok(alice.get("/kb/api/v1/memory/episodes"))
    # without allow_writes the tools are not offered
    fake2 = ScriptedChat([{"content": "ok"}])

    async def chat2(ref):
        return fake2

    client.app.state.manager.services.models.chat = chat2
    ok(alice.post("/kb/api/v1/agentic-search", json={"query": "q", "preset": "wiki", "knowledge_base_ids": [kb],
                                                     "model_id": model}))
    assert not any(t["function"]["name"].startswith(("wiki_write", "remember")) for t in fake2.calls[0]["tools"])
    presets = {p["name"]: p for p in ok(alice.get("/kb/api/v1/agentic-search/presets"))}
    assert "wiki_write_page" in presets["smart"]["write_tools"] and presets["smart"]["available"]


def test_regenerate_summary(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    model = chat_model(alice)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs", "summary_model_id": model}))["id"]
    calls = []

    class Summarizer:
        async def complete(self, messages, **kw):
            calls.append(1)
            return {"content": json.dumps({"summary": f"Summary {len(calls)}.", "gist": "g", "topics": []}),
                    "tool_calls": [], "usage": {}}

    async def chat(ref):
        return Summarizer()

    client.app.state.manager.services.models.chat = chat
    doc = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={"title": "P", "content": "# P\n\ntext"}))
    drain(client)
    assert ok(alice.get(f"/kb/api/v1/knowledge/{doc['id']}"))["description"] == "Summary 1."
    assert ok(alice.post(f"/kb/api/v1/knowledge/{doc['id']}/regenerate-summary"))["queued"] is True
    drain(client)
    assert ok(alice.get(f"/kb/api/v1/knowledge/{doc['id']}"))["description"] == "Summary 2."
