"""Phase 2: versions and replace, expiry, URL refresh, recency, tags, staleness, conflicts,
housekeeping, reindex, events, enrichment (summary, questions) and VLM."""

import asyncio
import datetime
import http.server
import io
import json
import threading

import pytest
import sqlalchemy as sa

from conftest import Caller, drain, make_client, user

RW = ("app:access", "knowledge:read", "knowledge:write")


class FakeChat:
    """Answers enrichment prompts deterministically; records the calls."""

    def __init__(self):
        self.calls = []

    async def complete(self, messages, **kw):
        self.calls.append(messages)
        system = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        if "describe documents for a search index" in system:
            text = {"summary": "How the pilot is restarted after failures.", "gist": "Pilot restart procedure.",
                    "topics": ["pilot", "restart"], "doc_type": "manual", "typical_question": "How to restart?"}
            return {"content": json.dumps(text), "tool_calls": [], "usage": {}}
        if "questions a reader would ask" in system:
            return {"content": "1. Which detector needs the hypervelocity flux capacitor?\n- How often?", "tool_calls": [],
                    "usage": {}}
        if "transcribe the text in images" in system:
            return {"content": "Beam current 5 mA at injection", "tool_calls": [], "usage": {}}
        if "describe images for a search index" in system:
            return {"content": "A plot of the beam current over one day.", "tool_calls": [], "usage": {}}
        if "compare two passages" in system:
            return {"content": '{"verdict": "updates", "explanation": "newer procedure"}', "tool_calls": [], "usage": {}}
        return {"content": "ok", "tool_calls": [], "usage": {}}


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        fake = FakeChat()

        async def chat(ref):
            return fake

        c.app.state.manager.services.models.chat = chat
        c.fake = fake
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


@pytest.fixture
def admin(client):
    return Caller(client, user("root", RW + ("admin",)))


def kb(caller, **body):
    response = caller.post("/kb/api/v1/knowledge-bases", json={"name": body.pop("name", "Docs"), **body})
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


def upload(caller, kb_id, data, name="doc.md", **form):
    response = caller.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/file",
                           files={"file": (name, io.BytesIO(data), "application/octet-stream")}, data=form)
    assert response.status_code in (200, 201), response.text
    return response.json()["data"]


def search(caller, query, kb_id, **body):
    response = caller.post("/kb/api/v1/knowledge-search", json={"query": query, "knowledge_base_ids": [kb_id], **body})
    assert response.status_code == 200, response.text
    return response.json()["data"]


def item(caller, knowledge_id):
    return caller.get(f"/kb/api/v1/knowledge/{knowledge_id}").json()["data"]


def runner_of(client):
    return client.app.state.manager.make_runner()


def run(coro_factory):
    from kb_core.orm.base.session import dispose_async_engine

    async def wrapped():
        try:
            return await coro_factory()
        finally:
            await dispose_async_engine()
    return asyncio.run(wrapped())


def sql(statement, **params):
    from kb_core.orm.base.session import get_engine

    with get_engine().begin() as conn:
        return conn.execute(sa.text(statement), params)


# --- versions and replace -------------------------------------------------------------------------
def test_replace_keeps_versions_and_serves_only_new_content(client, alice):
    kb_id = kb(alice, freshness_config={"keep_previous_versions": True})
    first = upload(alice, kb_id, b"# Guide\n\nThe obsolete xylophone procedure.\n", external_id="guide")
    drain(client)
    second = upload(alice, kb_id, b"# Guide\n\nThe modern zeppelin procedure.\n", external_id="guide", mode="replace")
    assert second["id"] == first["id"] and second["outcome"] == "replaced"
    drain(client)
    current = item(alice, first["id"])
    assert current["version"] == 2 and current["parse_status"] == "completed"
    assert search(alice, "zeppelin", kb_id) and not search(alice, "xylophone", kb_id)
    versions = alice.get(f"/kb/api/v1/knowledge/{first['id']}/versions").json()["data"]
    assert versions["current_version"] == 2 and versions["versions"][0]["version"] == 1
    assert versions["versions"][0]["file_kept"] is True
    old = alice.get(f"/kb/api/v1/knowledge/{first['id']}/versions/1/download")
    assert old.status_code == 200 and b"xylophone" in old.content
    listing = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge").json()
    assert listing["total"] == 1                                         # exactly one live document


def test_reprocessing_keeps_a_disabled_item_disabled(client, alice):
    kb_id = kb(alice)
    doc = upload(alice, kb_id, b"# Note\n\nKeep me disabled please.\n")
    drain(client)
    alice.put(f"/kb/api/v1/knowledge/{doc['id']}", json={"enable_status": "disabled"})
    alice.post(f"/kb/api/v1/knowledge/{doc['id']}/reparse")
    drain(client)
    assert item(alice, doc["id"])["enable_status"] == "disabled"
    assert not search(alice, "disabled please", kb_id)


# --- expiry and deprecation ------------------------------------------------------------------------
def test_expiry_task_disables_or_deletes(client, alice):
    keep = kb(alice, name="Keep")
    gone = kb(alice, name="Gone", freshness_config={"delete_on_expiry": True})
    a = upload(alice, keep, b"# A\n\nquasar telemetry\n", expires_at="2001-01-01T00:00:00Z")
    b = upload(alice, gone, b"# B\n\nquasar telemetry\n", expires_at="2001-01-01T00:00:00Z")
    drain(client)
    run(lambda: client.app.state.manager.services.freshness._expire())
    assert item(alice, a["id"])["enable_status"] == "disabled"
    assert alice.get(f"/kb/api/v1/knowledge/{b['id']}").status_code == 404


def test_deprecated_excluded_or_downranked(client, alice):
    exclude = kb(alice, name="Exclude")
    downrank = kb(alice, name="Downrank", freshness_config={"deprecated_mode": "downrank"})
    for kb_id in (exclude, downrank):
        upload(alice, kb_id, b"# Old\n\nThe nebula calibration constant.\n", deprecated="true")
    drain(client)
    assert not search(alice, "nebula calibration", exclude)
    hits = search(alice, "nebula calibration", downrank)
    assert hits and hits[0]["deprecated"] is True


# --- URL refresh ------------------------------------------------------------------------------------
class _Page(http.server.BaseHTTPRequestHandler):
    body = b"<html><body><article><h1>Status</h1><p>The magnet is warm today.</p></article></body></html>"
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


def test_url_refresh_only_reingests_changed_pages(client, alice):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        kb_id = kb(alice)
        url = f"http://127.0.0.1:{server.server_address[1]}/status"
        doc = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/url",
                         json={"url": url, "refresh_interval": 60}).json()["data"]
        drain(client)
        chunks_before = [c["id"] for c in alice.get(f"/kb/api/v1/chunks/{doc['id']}").json()["data"]]

        def due():
            sql("UPDATE kb_knowledges SET last_refreshed_at = now() - interval '2 minutes', "
                "processed_at = now() - interval '2 minutes' WHERE id = :i", i=doc["id"])

        due()
        hits = _Page.hits
        assert run(lambda: client.app.state.manager.services.freshness._refresh_scan()) == 1
        drain(client)
        assert _Page.hits == hits + 1                                     # one fetch
        assert [c["id"] for c in alice.get(f"/kb/api/v1/chunks/{doc['id']}").json()["data"]] == chunks_before
        assert item(alice, doc["id"])["version"] == 1                    # unchanged: no re-ingest

        _Page.body = b"<html><body><article><h1>Status</h1><p>The magnet is cold and quenched.</p></article></body></html>"
        due()
        run(lambda: client.app.state.manager.services.freshness._refresh_scan())
        drain(client)
        assert item(alice, doc["id"])["version"] == 2
        assert search(alice, "quenched", kb_id)
        versions = alice.get(f"/kb/api/v1/knowledge/{doc['id']}/versions").json()["data"]["versions"]
        assert versions[0]["replaced_by"] == "refresh"
    finally:
        server.shutdown()


# --- recency ---------------------------------------------------------------------------------------
def test_recency_boost(client, alice):
    kb_id = kb(alice, freshness_config={"recency_half_life_days": 30, "recency_floor": 0.2})
    upload(alice, kb_id, b"# Old\n\nThe aurora schedule for shifts.\n", name="old.md",
           valid_from="2020-01-01T00:00:00Z")
    upload(alice, kb_id, b"# New\n\nThe aurora schedule for shifts.\n", name="new.md")
    drain(client)
    hits = search(alice, "aurora schedule", kb_id)
    assert hits[0]["knowledge_title"] == "new" and hits[-1]["recency_weight"] == pytest.approx(0.2)


# --- tags ------------------------------------------------------------------------------------------
def test_tags(client, alice):
    kb_id = kb(alice)
    ops = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/tags", json={"name": "ops"}).json()["data"]
    dup = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/tags", json={"name": "ops"})
    assert dup.status_code == 409 and dup.json()["data"]["id"] == ops["id"]
    tagged = upload(alice, kb_id, b"# Tagged\n\nThe pulsar rota.\n", name="t.md", tag_ids=ops["id"])
    plain = upload(alice, kb_id, b"# Plain\n\nThe pulsar rota.\n", name="p.md")
    drain(client)
    assert item(alice, tagged["id"])["tag_ids"] == [ops["id"]]
    listed = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge?tag_ids={ops['id']}").json()["data"]
    assert [i["id"] for i in listed] == [tagged["id"]]
    untagged = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge?tag_ids=__untagged__").json()["data"]
    assert [i["id"] for i in untagged] == [plain["id"]]
    assert {h["knowledge_id"] for h in search(alice, "pulsar rota", kb_id, tag_ids=[ops["id"]])} == {tagged["id"]}
    assert {h["knowledge_id"] for h in search(alice, "pulsar rota", kb_id, tag_ids=[str(ops["seq_id"])])} == {tagged["id"]}
    counts = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/tags").json()["data"]
    assert counts[0]["knowledge_count"] == 1
    # re-tagging needs no re-indexing
    retag = alice.put("/kb/api/v1/knowledge/tags", json={"updates": {plain["id"]: [ops["id"]], tagged["id"]: []}})
    assert retag.status_code == 200, retag.text
    assert {h["knowledge_id"] for h in search(alice, "pulsar rota", kb_id, tag_ids=[ops["id"]])} == {plain["id"]}
    assert alice.delete(f"/kb/api/v1/knowledge-bases/{kb_id}/tags/{ops['id']}").status_code == 400
    forced = alice.delete(f"/kb/api/v1/knowledge-bases/{kb_id}/tags/{ops['id']}?force=true").json()["data"]
    assert forced["detached"] == 1 and item(alice, plain["id"])["tag_ids"] == []


# --- conflicts and staleness -------------------------------------------------------------------------
def test_conflicts_and_review(client, alice):
    kb_id = kb(alice, freshness_config={"conflict_detection": True, "conflict_similarity": 0.9,
                                        "conflict_llm_judge": True}, summary_model_id=chat_model(alice))
    text = "\n\n".join(f"Section {i}: the cryogenic valve {i} must be closed before warm-up." for i in range(6))
    older = upload(alice, kb_id, ("# Procedure\n\n" + text).encode(), name="v1.md")
    drain(client)
    sql("UPDATE kb_knowledges SET created_at = now() - interval '1 day' WHERE id = :i", i=older["id"])
    newer = upload(alice, kb_id, ("# Procedure (new)\n\n" + text + "\n\nAlso log it.").encode(), name="v2.md")
    drain(client)
    assert item(alice, newer["id"])["parse_status"] == "completed"
    conflicts = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/conflicts").json()["data"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["knowledge_id"] == newer["id"] and conflict["other_knowledge_id"] == older["id"]
    assert conflict["verdict"] == "updates" and conflict["matched_chunks"] >= 1
    assert item(alice, older["id"])["superseded_by"] == [newer["id"]]
    stale = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/staleness").json()["data"]
    assert stale["summary"]["possibly_superseded"] == 1
    reviewed = alice.put(f"/kb/api/v1/knowledge-conflicts/{conflict['id']}", json={"deprecate_older": True})
    assert reviewed.json()["data"]["status"] == "resolved"
    assert all(h["knowledge_id"] != older["id"] for h in search(alice, "cryogenic valve", kb_id))
    assert "deprecated" in alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/staleness").json()["data"]["summary"]


# --- housekeeping, reindex ------------------------------------------------------------------------------
def test_housekeeping_is_admin_only_and_dry_run_by_default(client, alice, admin):
    kb_id = kb(alice)
    doc = upload(alice, kb_id, b"# X\n\ntext\n")
    drain(client)
    alice.delete(f"/kb/api/v1/knowledge/{doc['id']}")
    sql("UPDATE kb_knowledges SET deleted_at = now() - interval '60 days' WHERE id = :i", i=doc["id"])
    assert alice.post("/kb/api/v1/system/housekeeping", json={}).status_code == 403
    dry = admin.post("/kb/api/v1/system/housekeeping", json={}).json()["data"]
    assert dry["dry_run"] is True and dry["purged_items"] == 1
    assert sql("SELECT count(*) FROM kb_knowledges WHERE id = :i", i=doc["id"]).scalar() == 1
    real = admin.post("/kb/api/v1/system/housekeeping", json={"dry_run": False}).json()["data"]
    assert real["purged_items"] == 1
    assert sql("SELECT count(*) FROM kb_knowledges WHERE id = :i", i=doc["id"]).scalar() == 0


def test_reindex_with_another_embedding_model(client, alice):
    kb_id = kb(alice)
    upload(alice, kb_id, b"# Doc\n\nThe galaxy rotation curve.\n")
    drain(client)
    model = alice.post("/kb/api/v1/models", json={"name": "hash-32", "type": "Embedding", "parameters": {
        "provider": "hash", "dimension": 32}}).json()["data"]
    response = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/reindex", json={"embedding_model_id": model["id"]})
    assert response.status_code == 200 and response.json()["data"]["queued"] == 1
    drain(client)
    assert sql("SELECT DISTINCT dimension FROM kb_embeddings WHERE knowledge_base_id = :k", k=kb_id).scalars().all() == [32]
    assert search(alice, "galaxy rotation", kb_id, disable_keywords_match=True)


# --- events -------------------------------------------------------------------------------------------
def test_events_reach_only_readers(client, alice):
    manager = client.app.state.manager
    bob = Caller(client, user("bob", RW, org="atlas"))
    kb_id = kb(alice)

    async def scenario():
        mine = manager.events.subscribe(alice.ctx)
        theirs = manager.events.subscribe(bob.ctx)
        upload(alice, kb_id, b"# E\n\nevent text\n")
        await manager.events.relay_once()
        got = []
        while not mine.queue.empty():
            got.append(mine.queue.get_nowait())
        return got, theirs.queue.qsize()

    got, bobs = run(scenario)
    assert got and got[0]["type"] == "knowledge_status" and got[0]["parse_status"] == "pending" and bobs == 0
    assert "_visibility" not in got[0]


# --- enrichment and VLM ---------------------------------------------------------------------------------
def chat_model(caller):
    return caller.post("/kb/api/v1/models", json={"name": "fake-chat", "type": "KnowledgeQA", "parameters": {
        "base_url": "http://127.0.0.1:9/v1"}}).json()["data"]["id"]


def test_summary_and_questions(client, alice):
    model_id = chat_model(alice)
    kb_id = kb(alice, summary_model_id=model_id,
               question_generation_config={"enabled": True, "question_count": 2})
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}", json={"question_generation_config": {"enabled": True,
                                                                                         "question_count": 2}})
    doc = upload(alice, kb_id, b"# Pilot\n\nRestart the pilot after stage-out failures.\n")
    drain(client)
    current = item(alice, doc["id"])
    assert current["parse_status"] == "completed" and current["description"].startswith("How the pilot")
    chunks = alice.get(f"/kb/api/v1/chunks/{doc['id']}").json()["data"]
    assert any(c["chunk_type"] == "summary" for c in chunks)
    text = [c for c in chunks if c["chunk_type"] == "text"][0]
    assert text["metadata"]["generated_questions"][0]["question"].startswith("Which detector")
    hits = search(alice, "hypervelocity flux capacitor", kb_id)
    assert hits and hits[0]["id"] == text["id"]                          # a question hit maps to its chunk


def test_vlm_ocr_and_caption(client, alice):
    vlm = alice.post("/kb/api/v1/models", json={"name": "fake-vlm", "type": "VLLM", "parameters": {
        "base_url": "http://127.0.0.1:9/v1"}}).json()["data"]["id"]
    kb_id = kb(alice)
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}", json={"vlm_config": {"enabled": True, "model_id": vlm}})
    doc = upload(alice, kb_id, b"\x89PNG\r\n\x1a\nfake", name="beam.png")
    drain(client)
    assert item(alice, doc["id"])["parse_status"] == "completed"
    kinds = {c["chunk_type"] for c in alice.get(f"/kb/api/v1/chunks/{doc['id']}").json()["data"]}
    assert {"image_ocr", "image_caption"} <= kinds
    assert search(alice, "beam current injection", kb_id)
