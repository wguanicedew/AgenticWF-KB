"""Remote knowledge bases (agenticwf.kb.runtime.remote): listing, switches, search merge, reads,
credentials, and the /remotes API. The unit tests fake the rows and mock the remote's HTTP; the
API test needs the database."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import agenticwf.kb.runtime.services  # noqa: F401  (first: retrieval.search and services import each other)
from conftest import Caller, user
from agenticwf.kb.runtime.errors import NotFound
from agenticwf.kb.runtime.remote import Remote, RemoteService
from agenticwf.kb.runtime.retrieval.search import SearchOptions, SearchService, merge_sources
from agenticwf.kb.runtime.thread.thread import KBThread

ME = user("local:alice")
KBS = [{"id": "kb-1", "ref": "system:eic-docs", "name": "EIC docs", "type": "document", "owner_subject": "system"},
       {"id": "kb-2", "ref": "eic:ops", "name": "Ops", "type": "faq", "owner_subject": "eic"}]


def run(coro):
    return asyncio.run(coro)


def hit(chunk, score, kb="kb-1"):
    return {"id": chunk, "content": f"text {chunk}", "knowledge_id": f"k-{chunk}", "knowledge_base_id": kb,
            "chunk_type": "text", "score": score, "parent_chunk_id": "", "pre_chunk_id": "", "next_chunk_id": "",
            "sub_chunk_id": []}


class FakeRemoteKB:
    """A remote KB server: two KBs, a search answer, one document."""

    def __init__(self, token="secret-token"):
        self.calls, self.token = [], token

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"success": False, "error": {"code": 1001, "message": "bad token"}})
        path = request.url.path.split("/api/v1", 1)[1]
        if path == "/knowledge-bases":
            return httpx.Response(200, json={"success": True, "data": KBS, "total": len(KBS)})
        if path.startswith("/knowledge-bases/"):
            ref = path.rsplit("/", 1)[1]
            kb = next((k for k in KBS if ref in (k["id"], k["ref"])), None)
            return httpx.Response(200, json={"success": True, "data": kb}) if kb else \
                httpx.Response(404, json={"success": False, "error": {"code": 1003, "message": "not found"}})
        if path == "/knowledge-search":
            return httpx.Response(200, json={"success": True, "data": [hit("c1", 0.8), hit("c2", 0.4)]})
        if path == "/knowledge/k-c1/content":
            return httpx.Response(200, json={"success": True, "data": {"knowledge_id": "k-c1", "content": "full"}})
        return httpx.Response(404, json={"success": False, "error": {"code": 1003}})


def row(name="central", enabled=True, disabled=(), builtin=False, kbs=()):
    return SimpleNamespace(id=f"id-{name}", name=name, url="https://host/kb", auth={}, knowledge_bases=list(kbs),
                           disabled_kbs=list(disabled), is_enabled=enabled, is_builtin=builtin, timeout=10.0)


class FakeRows(RemoteService):
    """Rows without a database; the client gets a stored token, as a user's remote has."""

    def __init__(self, rows, transport):
        super().__init__(transport=transport)
        self.rows = rows

    async def _rows(self, ctx):
        return [r for r in self.rows if ctx.subject == "local:alice"]

    def _client(self, r):
        client = super()._client(r)
        client.token = "secret-token"
        return client


@pytest.fixture
def fake():
    return FakeRemoteKB()


def service(fake, *rows):
    return FakeRows(list(rows) or [row()], httpx.MockTransport(fake))


def test_list_prefixes_ids_filters_type_and_skips_disabled(fake):
    remotes = service(fake)
    kbs = run(remotes.list_kbs(ME))
    assert [k["id"] for k in kbs] == ["remote:central:kb-1", "remote:central:kb-2"]
    assert kbs[0]["ref"] == "remote:central:system:eic-docs" and kbs[0]["read_only"] and kbs[0]["remote_id"] == "kb-1"
    assert [k["id"] for k in run(remotes.list_kbs(ME, "faq"))] == ["remote:central:kb-2"]
    assert [k["id"] for k in run(service(fake, row(disabled=["kb-2"])).list_kbs(ME))] == ["remote:central:kb-1"]
    assert run(service(fake, row(enabled=False)).list_kbs(ME)) == []
    assert run(remotes.list_kbs(user("local:bob"))) == []
    got = run(service(fake, row(disabled=["kb-1"])).get_kb(ME, "remote:central:system:eic-docs"))
    assert got["id"] == "remote:central:kb-1" and got["enabled"] is False


def test_unreachable_remote_lists_nothing():
    down = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    remotes = FakeRows([row()], down)
    assert run(remotes.list_kbs(ME)) == []
    assert run(RemoteService.search("q", run(remotes.plan(ME, ["remote:central"])), {})) == [[]]


def test_plan(fake):
    remotes = service(fake, row("a"), row("b", disabled=["kb-2"]), row("c", enabled=False))
    plan = run(remotes.plan(ME, None))
    assert {name: refs for name, (_, refs) in plan.items()} == {"a": None, "b": ["kb-1"]}
    plan = run(remotes.plan(ME, ["mine", "remote:a:system:x", "remote:a:kb-9"]))
    assert {name: refs for name, (_, refs) in plan.items()} == {"a": ["system:x", "kb-9"]}
    assert {n: r for n, (_, r) in run(remotes.plan(ME, ["remote:b", "remote:b:kb-1"])).items()} == {"b": None}
    assert run(remotes.plan(ME, ["mine"])) == {}
    for missing in ("remote:nope", "remote:c"):          # unknown, or switched off
        with pytest.raises(NotFound):
            run(remotes.plan(ME, [missing]))
    assert run(service(fake, row(disabled=["kb-1", "kb-2"])).plan(ME, None)) == {}


class NoModels:
    async def default_id(self, kind):
        return None


def test_search_only_remote_kbs(fake):
    search = SearchService(NoModels(), kbs=None, remotes=service(fake))
    results = run(search.search(ME, "pilot restart", ["remote:central:system:eic-docs"], SearchOptions(top_k=5)))
    assert [r["id"] for r in results] == ["remote:central:c1", "remote:central:c2"]
    assert results[0]["knowledge_id"] == "remote:central:k-c1" and results[0]["remote"] == "central"
    body = json.loads(fake.calls[-1].content)
    assert body["knowledge_base_ids"] == ["system:eic-docs"] and body["match_count"] == 5


def test_no_remote_refs_without_remotes():
    search = SearchService(NoModels(), kbs=None, remotes=None)
    with pytest.raises(NotFound):
        run(search.search(ME, "q", ["remote:central"]))


def test_tags_keep_a_search_local(fake):
    search = SearchService(NoModels(), kbs=None, remotes=service(fake))
    assert run(search.search(ME, "q", ["remote:central"], SearchOptions(tag_ids=["t1"]))) == []
    assert not fake.calls


def test_merge_sources_normalises_each_source():
    local = [{"id": "l1", "score": 0.2}, {"id": "l2", "score": 0.1}]
    far = [{"id": "r1", "score": 9.0}, {"id": "r2", "score": 3.0}]
    merged = merge_sources([local, far])
    assert [m["id"] for m in merged] == ["l1", "r1", "l2", "r2"]
    assert merged[0]["score"] == merged[1]["score"] == 1.0


def test_read_content_through_the_remote(fake):
    remotes = service(fake)
    data = run(remotes.read(ME, "content", "remote:central:k-c1", offset=0, limit=10))
    assert data["knowledge_id"] == "remote:central:k-c1" and data["content"] == "full"
    with pytest.raises(NotFound):
        run(remotes.read(ME, "content", "remote:central:missing"))


class FakeKBs:
    def __init__(self, n):
        self.rows = [{"id": f"local-{i}"} for i in range(n)]

    async def ensure_personal(self, ctx):
        return None

    async def list(self, ctx, page, page_size, shared_only, kb_type):
        start = (page - 1) * page_size
        return self.rows[start:start + page_size], len(self.rows)


def test_listing_appends_remote_kbs_across_pages(fake):
    services = SimpleNamespace(knowledge_bases=FakeKBs(3), remote=service(fake))
    thread = KBThread(auth=ME, services=services)
    pages = [run(thread.list_kbs(page, 2)) for page in (1, 2, 3)]
    assert [[k["id"] for k in rows] for rows, _ in pages] == [
        ["local-0", "local-1"], ["local-2", "remote:central:kb-1"], ["remote:central:kb-2"]]
    assert {total for _, total in pages} == {5}
    assert run(KBThread(auth=user("local:bob"), services=services).list_kbs(1, 10))[1] == 3


def test_operators_remotes_serve_the_standalone_user_only():
    remotes = RemoteService()
    remotes.bind("standalone", "alice")
    assert "system" in remotes._owners(ME) and "system" not in remotes._owners(user("local:bob"))
    remotes.bind("authsvc", "alice")
    assert remotes._owners(ME) == ["local:alice"]


def test_personal_access_token_is_exchanged_once(tmp_path):
    (tmp_path / "pat").write_text("pat_abc")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={"token_endpoint": "https://auth/oidc/token"})
        if request.url.path == "/oidc/token":
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            assert form["subject_token"] == "pat_abc" and form["client_id"] == "cli"
            return httpx.Response(200, json={"access_token": "secret-token", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer secret-token"
        return httpx.Response(200, json={"success": True, "data": [], "total": 0})

    r = Remote.from_section("central", {"url": "https://host/kb", "pat_file": str(tmp_path / "pat"),
                                        "auth_url": "https://auth", "client_id": "cli", "resource": "https://api"})
    r.transport = httpx.MockTransport(handler)
    run(r.call("GET", "/knowledge-bases"))
    run(r.call("GET", "/knowledge-bases"))
    assert seen.count("/oidc/token") == 1


# --- the /remotes API (database) ---------------------------------------------------------------------------
RW = ("app:access", "knowledge:read", "knowledge:write")
URL = "http://127.0.0.1:9/kb"              # loopback is whitelisted in the tests' [security]


def ok(response):
    assert response.status_code < 300, response.text
    return response.json()["data"]


def test_remotes_api(authsvc_client, monkeypatch):
    from cryptography.fernet import Fernet

    from conftest import sql_rows

    monkeypatch.setenv("KB_SECRETS_KEY", Fernet.generate_key().decode())
    fake = FakeRemoteKB()
    remotes = authsvc_client.app.state.manager.services.remote
    remotes.transport, remotes._secrets = httpx.MockTransport(fake), None
    alice, bob = Caller(authsvc_client, user("alice", RW)), Caller(authsvc_client, user("bob", RW))

    bad = alice.post("/kb/api/v1/remotes", json={"name": "central", "url": URL, "token": "wrong"})
    assert bad.status_code == 400 and "401" in bad.json()["error"]["message"]
    created = ok(alice.post("/kb/api/v1/remotes", json={"name": "central", "url": URL, "token": "secret-token"}))
    assert created["auth"]["token"].startswith("******") and created["is_enabled"]
    stored = sql_rows("SELECT auth->>'token' FROM kb_remotes WHERE id = :i", i=created["id"])[0][0]
    assert "secret-token" not in stored                                      # encrypted at rest
    assert alice.post("/kb/api/v1/remotes", json={"name": "central", "url": URL, "token": "secret-token"}).status_code == 409

    listed = ok(alice.get("/kb/api/v1/knowledge-bases?page_size=100"))
    assert {"remote:central:kb-1", "remote:central:kb-2"} <= {k["id"] for k in listed}
    assert not any(k.get("remote") for k in ok(bob.get("/kb/api/v1/knowledge-bases?page_size=100")))
    assert bob.get(f"/kb/api/v1/remotes/{created['id']}").status_code == 404

    ok(alice.put(f"/kb/api/v1/remotes/{created['id']}/knowledge-bases/kb-2", json={"enabled": False}))
    kbs = ok(alice.get(f"/kb/api/v1/remotes/{created['id']}/knowledge-bases"))["knowledge_bases"]
    assert {k["remote_id"]: k["enabled"] for k in kbs} == {"kb-1": True, "kb-2": False}
    listed = {k["id"] for k in ok(alice.get("/kb/api/v1/knowledge-bases?page_size=100"))}
    assert "remote:central:kb-1" in listed and "remote:central:kb-2" not in listed

    hits = ok(alice.post("/kb/api/v1/knowledge-search", json={"query": "pilot restart"}))
    assert any(h.get("remote") == "central" for h in hits)
    assert json.loads(fake.calls[-1].content)["knowledge_base_ids"] == ["kb-1"]
    content = ok(alice.get("/kb/api/v1/knowledge/remote:central:k-c1/content"))
    assert content["content"] == "full"

    ok(alice.put(f"/kb/api/v1/remotes/{created['id']}", json={"is_enabled": False}))
    assert not any(k.get("remote") for k in ok(alice.get("/kb/api/v1/knowledge-bases?page_size=100")))
    ok(alice.delete(f"/kb/api/v1/remotes/{created['id']}"))
    assert ok(alice.get("/kb/api/v1/remotes")) == []
