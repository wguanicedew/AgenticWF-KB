"""Phase 1 end to end: models, knowledge bases, ingestion, search and visibility (authsvc mode,
several users)."""

import http.server
import io
import threading

import pytest

from conftest import Caller, drain, make_client, user

DOC = b"""# Pilot operations

## Restarting the pilot

When a stage-out fails three times, stop the pilot service and start it again with the new
retry configuration. The calorimeter shifter must be told before the restart.

## Monitoring

The dashboard shows the queue depth, the number of running jobs and failed transfers per site.
"""

ADMIN_SCOPES = ("app:access", "knowledge:read", "knowledge:write", "admin")
RW = ("app:access", "knowledge:read", "knowledge:write")


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


@pytest.fixture
def bob(client):
    return Caller(client, user("bob", RW, org="eic"))


@pytest.fixture
def carol(client):
    return Caller(client, user("carol", RW, org="atlas"))


@pytest.fixture
def admin(client):
    return Caller(client, user("root", ADMIN_SCOPES))


def upload(caller, kb, data=DOC, name="pilot.md", **form):
    return caller.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file",
                       files={"file": (name, io.BytesIO(data), "text/markdown")}, data=form)


def search(caller, query, kbs=None, **body):
    response = caller.post("/kb/api/v1/knowledge-search",
                           json={"query": query, **({"knowledge_base_ids": kbs} if kbs else {}), **body})
    assert response.status_code == 200, response.text
    return response.json()["data"]


# --- models -----------------------------------------------------------------------------------
def test_system_model_and_user_models(alice, bob):
    models = alice.get("/kb/api/v1/models").json()["data"]
    system = [m for m in models if m["ref"] == "system:test-hash"]
    assert system and system[0]["is_builtin"] and system[0]["type"] == "Embedding" and system[0]["is_default"]
    debug = alice.post("/kb/api/v1/models/system:test-hash/debug").json()["data"]
    assert debug == {"ok": True, "dimension": 64}

    created = alice.post("/kb/api/v1/models", json={
        "name": "my-reranker", "type": "Rerank", "parameters": {"base_url": "http://127.0.0.1:9/v1"}})
    assert created.status_code == 201
    model_id = created.json()["data"]["id"]
    assert bob.get(f"/kb/api/v1/models/{model_id}").status_code == 404         # private to alice
    assert alice.put("/kb/api/v1/models/system:test-hash", json={"name": "x"}).status_code == 403
    bad = alice.post("/kb/api/v1/models", json={"name": "e", "type": "Embedding", "parameters": {}})
    assert bad.status_code == 400 and "dimension" in bad.json()["error"]["message"]


# --- knowledge bases --------------------------------------------------------------------------
def test_kb_lifecycle_and_refs(alice, bob):
    created = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Pilot Docs", "description": "ops"})
    assert created.status_code == 201
    kb = created.json()["data"]
    assert kb["slug"] == "pilot-docs" and kb["ref"] == "alice:pilot-docs" and kb["visibility"] == "private"
    assert kb["embedding_model_id"]                                            # the default (test-hash)
    assert alice.get("/kb/api/v1/knowledge-bases/alice:pilot-docs").json()["data"]["id"] == kb["id"]
    assert bob.get(f"/kb/api/v1/knowledge-bases/{kb['id']}").status_code == 404

    listing = alice.get("/kb/api/v1/knowledge-bases").json()
    assert {k["slug"] for k in listing["data"]} == {"personal", "pilot-docs"} and listing["total"] == 2
    me = alice.get("/kb/api/v1/me").json()["data"]
    assert me["personal_knowledge_base_id"]

    renamed = alice.put(f"/kb/api/v1/knowledge-bases/{kb['id']}", json={"name": "Pilot", "chunking_config": {
        "chunk_size": 300}}).json()["data"]
    assert renamed["name"] == "Pilot" and renamed["chunking_config"]["chunk_size"] == 300
    assert bob.put(f"/kb/api/v1/knowledge-bases/{kb['id']}", json={"name": "x"}).status_code == 404
    assert alice.post("/kb/api/v1/knowledge-bases", json={"name": "Everyone", "visibility": "all"}).status_code == 403
    assert alice.delete(f"/kb/api/v1/knowledge-bases/{kb['id']}").status_code == 200
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb['id']}").status_code == 404


def test_group_kb_visibility(alice, bob, carol):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "EIC ops", "visibility": "groups", "groups": ["org:eic"]}).json()["data"]
    assert bob.get(f"/kb/api/v1/knowledge-bases/{kb['id']}").status_code == 200   # same org
    assert carol.get(f"/kb/api/v1/knowledge-bases/{kb['id']}").status_code == 404
    shared = bob.get("/kb/api/v1/shared-knowledge-bases").json()["data"]
    assert [k["id"] for k in shared] == [kb["id"]]
    bad = carol.post("/kb/api/v1/knowledge-bases", json={"name": "x", "visibility": "groups", "groups": ["org:eic"]})
    assert bad.status_code == 403                                               # not her group


# --- ingestion ------------------------------------------------------------------------------------
def test_file_upload_processing_and_reading(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    response = upload(alice, kb)
    assert response.status_code == 201
    item = response.json()["data"]
    assert item["outcome"] == "created" and item["parse_status"] == "pending" and item["visibility"] == "private"
    assert drain(client) == 1
    item = alice.get(f"/kb/api/v1/knowledge/{item['id']}").json()["data"]
    assert item["parse_status"] == "completed" and item["content_hash"] and item["processed_at"]

    chunks = alice.get(f"/kb/api/v1/chunks/{item['id']}").json()
    assert chunks["total"] >= 2
    headers = {c["context_header"] for c in chunks["data"]}
    assert "Pilot operations > Restarting the pilot" in headers
    one = alice.get(f"/kb/api/v1/chunks/by-id/{chunks['data'][0]['id']}").json()["data"]
    assert one["knowledge_id"] == item["id"]

    page = alice.get(f"/kb/api/v1/knowledge/{item['id']}/content?offset=0&limit=40").json()["data"]
    assert page["returned_chars"] == 40 and page["next_offset"] == 40 and page["content"].startswith("# Pilot")
    download = alice.get(f"/kb/api/v1/knowledge/{item['id']}/download")
    assert download.status_code == 200 and download.content == DOC

    duplicate = upload(alice, kb)
    assert duplicate.status_code == 409 and duplicate.json()["data"]["id"] == item["id"]
    unchanged = upload(alice, kb, mode="replace")
    assert unchanged.status_code == 200 and unchanged.json()["data"]["outcome"] == "unchanged"
    replaced = upload(alice, kb, data=DOC + b"\nA new paragraph.\n", mode="replace")
    assert replaced.json()["data"]["outcome"] == "replaced" and replaced.json()["data"]["id"] == item["id"]
    drain(client)
    assert alice.get(f"/kb/api/v1/knowledge/{item['id']}").json()["data"]["version"] == 2


def test_unsupported_and_broken_files(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    assert upload(alice, kb, name="clip.mp4").status_code == 400
    broken = upload(alice, kb, data=b"{not json", name="broken.json").json()["data"]
    drain(client)
    item = alice.get(f"/kb/api/v1/knowledge/{broken['id']}").json()["data"]
    assert item["parse_status"] == "failed" and "invalid JSON" in item["error_message"]   # no retry


def test_manual_draft_and_publish(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Notes"}).json()["data"]["id"]
    draft = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual",
                       json={"title": "Shift notes", "content": "Calorimeter HV trip at 03:00.", "status": "draft"})
    item = draft.json()["data"]
    assert item["parse_status"] == "draft" and drain(client) == 0
    alice.put(f"/kb/api/v1/knowledge/manual/{item['id']}", json={"status": "publish"})
    drain(client)
    assert alice.get(f"/kb/api/v1/knowledge/{item['id']}").json()["data"]["parse_status"] == "completed"
    assert search(alice, "calorimeter trip", [kb])


class _Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b"<html><head><title>Beamline status</title></head><body><article><h1>Beamline status</h1>"
                b"<p>The beamline magnets were ramped down for the maintenance window on Tuesday.</p>"
                b"</article></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_url_import(client, alice):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Web"}).json()["data"]["id"]
        url = f"http://127.0.0.1:{server.server_address[1]}/status"
        item = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/url", json={"url": url}).json()["data"]
        drain(client)
        item = alice.get(f"/kb/api/v1/knowledge/{item['id']}").json()["data"]
        assert item["parse_status"] == "completed" and item["title"] == "Beamline status"
        assert alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/url", json={"url": url}).status_code == 409
        assert search(alice, "magnets ramped down", [kb])
    finally:
        server.shutdown()


# --- search ------------------------------------------------------------------------------------
def test_hybrid_vector_and_keyword_search(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    upload(alice, kb)
    drain(client)
    hybrid = search(alice, "restart the pilot after stage-out failures", [kb])
    assert hybrid and "stage-out" in hybrid[0]["content"]
    first = hybrid[0]
    for field in ("knowledge_id", "knowledge_title", "knowledge_base_id", "chunk_index", "seq", "start_at",
                  "end_at", "score", "match_type", "knowledge_date", "knowledge_version"):
        assert field in first
    keyword = search(alice, "calorimeter shifter", [kb], disable_vector_match=True)
    assert keyword and all(r["match_type"] == "keyword" for r in keyword)
    vector = search(alice, "queue depth dashboard", [kb], disable_keywords_match=True)
    assert vector and all(r["match_type"] == "vector" for r in vector)
    odd = search(alice, 'error: "unbalanced (paren AND', [kb])        # no query-parser errors
    assert isinstance(odd, list)
    get = alice.get(f"/kb/api/v1/knowledge-bases/{kb}/hybrid-search?query_text=dashboard&match_count=1")
    assert get.status_code == 200 and len(get.json()["data"]) == 1


def test_item_visibility_in_a_shared_kb(client, alice, bob, carol):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "EIC shared", "visibility": "groups", "groups": ["org:eic"], "contribute": "readers"}).json()["data"]
    kb_id = kb["id"]
    private = upload(alice, kb_id, data=DOC, name="alice-notes.md", visibility="private").json()["data"]
    shared = upload(bob, kb_id, data=b"# Cryostat\n\nThe cryostat pressure alarm threshold is 1.2 bar.\n",
                    name="cryo.md").json()["data"]
    drain(client)

    def ids(results):
        return {r["knowledge_id"] for r in results}

    assert private["id"] in ids(search(alice, "restart the pilot", [kb_id]))      # her own private item
    assert private["id"] not in ids(search(bob, "restart the pilot", [kb_id]))    # hidden from bob
    assert bob.get(f"/kb/api/v1/knowledge/{private['id']}").status_code == 404
    assert [k["id"] for k in bob.get(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge").json()["data"]] == [shared["id"]]
    assert shared["id"] in ids(search(alice, "cryostat pressure alarm", [kb_id]))   # bob's item, the group sees it
    assert carol.post("/kb/api/v1/knowledge-search", json={"query": "cryostat", "knowledge_base_ids": [kb_id]}
                      ).status_code == 404                                    # carol cannot see the KB

    # sharing the private item with everyone in the KB: applies at the next search
    changed = alice.put(f"/kb/api/v1/knowledge/{private['id']}", json={"visibility": "all"})
    assert changed.status_code == 200
    assert private["id"] in ids(search(bob, "restart the pilot", [kb_id]))
    # a duplicate check never reveals someone else's private file
    hidden = upload(alice, kb_id, data=b"# Secret\n\nonly alice\n", name="s.md", visibility="private").json()["data"]
    again = upload(bob, kb_id, data=b"# Secret\n\nonly alice\n", name="s.md")
    assert again.status_code == 201 and again.json()["data"]["id"] != hidden["id"]
    # bob may not change alice's item
    assert bob.put(f"/kb/api/v1/knowledge/{private['id']}", json={"title": "x"}).status_code == 403


def test_contribution_rules(alice, bob):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "Owner only", "visibility": "groups", "groups": ["org:eic"]}).json()["data"]["id"]
    assert upload(bob, kb).status_code == 403                                 # contribute = owner
    private = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Mine"}).json()["data"]["id"]
    assert upload(bob, private).status_code == 404                            # cannot even see it


def test_disable_and_delete_remove_from_search(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    item = upload(alice, kb).json()["data"]
    drain(client)
    assert search(alice, "dashboard queue depth", [kb])
    alice.put(f"/kb/api/v1/knowledge/{item['id']}", json={"enable_status": "disabled"})
    assert not search(alice, "dashboard queue depth", [kb])
    alice.put(f"/kb/api/v1/knowledge/{item['id']}", json={"enable_status": "enabled"})
    assert search(alice, "dashboard queue depth", [kb])
    assert alice.delete(f"/kb/api/v1/knowledge/{item['id']}").status_code == 200
    assert not search(alice, "dashboard queue depth", [kb])
    assert alice.get(f"/kb/api/v1/knowledge/{item['id']}").status_code == 404


def test_validity_window(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    item = upload(alice, kb, valid_until="2001-01-01T00:00:00Z").json()["data"]
    drain(client)
    assert not search(alice, "dashboard queue depth", [kb])                  # expired
    alice.put(f"/kb/api/v1/knowledge/{item['id']}", json={"valid_until": None})
    assert search(alice, "dashboard queue depth", [kb])


def test_chunk_edit_reindexes(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    item = upload(alice, kb).json()["data"]
    drain(client)
    chunk = alice.get(f"/kb/api/v1/chunks/{item['id']}").json()["data"][0]
    edited = alice.put(f"/kb/api/v1/chunks/{item['id']}/{chunk['id']}",
                       json={"content": "Neutrino detector photomultiplier calibration."})
    assert edited.status_code == 200 and edited.json()["data"]["content_revision"] == 2
    assert search(alice, "photomultiplier calibration", [kb], disable_vector_match=True)


# --- the other modes -----------------------------------------------------------------------------
def test_service_mode_scopes_knowledge_to_the_named_user(database):
    from kb_api.service import encode_principal

    with make_client("service", required_scopes=frozenset()) as client:
        def as_(name, org=None):
            return {"X-KB-Principal": encode_principal(user(name, ("knowledge:read", "knowledge:write"), org=org))}

        kb = client.post("/kb/api/v1/knowledge-bases", json={"name": "Agent KB"}, headers=as_("alice"))
        assert kb.status_code == 201
        kb_id = kb.json()["data"]["id"]
        client.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/manual", headers=as_("alice"),
                    json={"title": "Note", "content": "The hadronic calorimeter was recalibrated."})
        drain(client)
        mine = client.post("/kb/api/v1/knowledge-search", headers=as_("alice"),
                           json={"query": "calorimeter recalibrated", "knowledge_base_ids": [kb_id]})
        assert mine.status_code == 200 and mine.json()["data"]
        other = client.post("/kb/api/v1/knowledge-search", headers=as_("bob"),
                            json={"query": "calorimeter recalibrated", "knowledge_base_ids": [kb_id]})
        assert other.status_code == 404
        everything = client.post("/kb/api/v1/knowledge-search", headers=as_("bob"), json={"query": "calorimeter"})
        assert everything.status_code == 200 and everything.json()["data"] == []


def test_standalone_local_user_owns_what_it_adds(database):
    from kb_api.settings import token_path
    from kb_api.standalone import ensure_token

    token = ensure_token(token_path())[0]
    with make_client("standalone", local_user="wguan") as client:
        headers = {"Authorization": f"Bearer {token}"}
        kb = client.post("/kb/api/v1/knowledge-bases", json={"name": "Laptop"}, headers=headers).json()["data"]
        assert kb["owner_subject"] == "local:wguan" and kb["ref"] == "local:wguan:laptop"
        assert client.get("/kb/api/v1/knowledge-bases/local:wguan:laptop", headers=headers).status_code == 200


def test_reassign_owner(client, alice):
    from kb_api.db import reassign_owner

    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    upload(alice, kb)
    drain(client)
    counts = reassign_owner("alice", "user_9f")
    assert counts["kb_knowledge_bases"] >= 1 and counts["kb_knowledges"] == 1 and counts["kb_embeddings"] >= 1
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb}").status_code == 404
    moved = Caller(client, user("user_9f", RW))
    assert moved.get(f"/kb/api/v1/knowledge-bases/{kb}").status_code == 200
    assert search(moved, "dashboard queue depth", [kb])
