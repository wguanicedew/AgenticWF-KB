"""Redaction: remove sensitive text from an item everywhere it is kept, so nothing brings it back."""

import http.server
import io
import threading

import pytest

from conftest import Caller, drain, make_client, user
from agenticwf.kb.runtime.errors import Invalid
from agenticwf.kb.runtime.redaction import Redactor

RW = ("app:access", "knowledge:read", "knowledge:write")
IBAN = "DE89 3704 0044 0532 0130 00"
DOC = f"""# Travel expenses

## Reimbursement

Refunds for the beamtime travel are paid to the account {IBAN} at Musterbank within 30 days.

## Contacts

Ask the secretariat about the per diem rates.
""".encode()


# --- rules (no database) ------------------------------------------------------------------------------
def test_redactor_rules():
    r = Redactor.from_body({"terms": [IBAN]})
    text, n = r.apply(f"pay to de89 3704  0044\n0532 0130 00 now and {IBAN}")
    assert n == 2 and "3704" not in text and text.count("[REDACTED]") == 2
    assert Redactor.from_body({"terms": ["Musterbank"], "case_sensitive": True}).count("musterbank") == 0
    assert Redactor.from_body({"patterns": [r"DE\d{2}(?:\s?\d{4}){4}\s?\d{2}"]}).count(f"x {IBAN} y") == 1
    for bad in ({}, {"terms": []}, {"patterns": ["a*"]}, {"patterns": ["("]}, {"terms": ["RED"], "replacement": "[RED]"}):
        with pytest.raises(Invalid):
            Redactor.from_body(bad)
    previews = r.previews(f"Refunds are paid to the account {IBAN} at Musterbank.")
    assert previews and "3704" not in previews[0] and "[REDACTED] at Musterbank" in previews[0]
    assert r.scrub({"a": [IBAN, {"b": IBAN}], "n": 3}) == {"a": ["[REDACTED]", {"b": "[REDACTED]"}], "n": 3}


# --- end to end ---------------------------------------------------------------------------------------
@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def grep(caller, kb, text):
    response = caller.post("/kb/api/v1/knowledge-grep", json={"pattern": text, "regex": False,
                                                               "knowledge_base_ids": [kb]})
    assert response.status_code == 200, response.text
    return response.json()["data"]["matches"]


def chunks_text(caller, item_id):
    return " ".join(c["content"] for c in caller.get(f"/kb/api/v1/chunks/{item_id}?page_size=100").json()["data"])


def test_redact_a_file_for_good(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Admin"}).json()["data"]["id"]
    item = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file",
                      files={"file": ("travel.md", io.BytesIO(DOC), "text/markdown")}).json()["data"]
    drain(client)
    assert grep(alice, kb, "0532 0130")

    preview = alice.post(f"/kb/api/v1/knowledge/{item['id']}/redact", json={"terms": [IBAN], "dry_run": True})
    assert preview.status_code == 200, preview.text
    preview = preview.json()["data"]
    assert preview["dry_run"] and preview["matches"] >= 2          # the parsed text and its chunk
    assert all("3704" not in p for p in preview["previews"])
    assert grep(alice, kb, "0532 0130")                             # a dry run changes nothing

    done = alice.post(f"/kb/api/v1/knowledge/{item['id']}/redact", json={"terms": [IBAN]}).json()["data"]
    assert done["matches"] == preview["matches"]
    # gone at once from search, the chunks and the stored text, before re-processing
    assert not grep(alice, kb, "0532 0130")
    assert "0532" not in chunks_text(alice, item["id"])
    content = alice.get(f"/kb/api/v1/knowledge/{item['id']}/content").json()["data"]["content"]
    assert "0532" not in content and "[REDACTED] at Musterbank" in content
    drain(client)

    after = alice.get(f"/kb/api/v1/knowledge/{item['id']}").json()["data"]
    assert after["parse_status"] == "completed" and after["metadata"]["redacted"]["by"] == "alice"
    assert after["file_name"] == "travel.redacted.md"
    download = alice.get(f"/kb/api/v1/knowledge/{item['id']}/download")
    assert download.status_code == 200 and b"0532" not in download.content and b"[REDACTED]" in download.content
    assert "0532" not in chunks_text(alice, item["id"])
    hits = alice.post("/kb/api/v1/knowledge-search", json={"query": "refund account Musterbank",
                                                          "knowledge_base_ids": [kb]}).json()["data"]
    assert hits and all("0532" not in h["content"] for h in hits)

    # nothing on disk keeps it, and a reparse cannot bring it back
    from agenticwf.kb.runtime.storage import owner_scope

    storage = client.app.state.manager.services.storage
    keys = storage.keys(owner_scope(after["owner_subject"]))
    assert keys and all(b"0532" not in storage.read(k) for k in keys)
    alice.post(f"/kb/api/v1/knowledge/{item['id']}/reparse")
    drain(client)
    assert "0532" not in chunks_text(alice, item["id"]) and not grep(alice, kb, "0532 0130")

    # a new original ends the redacted state
    replaced = alice.put(f"/kb/api/v1/knowledge/{item['id']}/file",
                         files={"file": ("travel.md", io.BytesIO(b"# Travel\n\nNo bank details any more.\n"),
                                         "text/markdown")}).json()["data"]
    assert "redacted" not in replaced["metadata"]


def test_redact_url_and_manual_items_and_the_whole_kb(client, alice):
    class Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"<html><head><title>Payments</title></head><body><p>Our account is {IBAN}, " \
                   f"please quote the invoice number.</p></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Admin"}).json()["data"]["id"]
        url = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/url",
                         json={"url": f"http://127.0.0.1:{server.server_address[1]}/pay", "refresh_interval": 3600}
                         ).json()["data"]
        manual = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={
            "title": "Bank", "content": f"Salary goes to {IBAN}.", "status": "publish"}).json()["data"]
        clean = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={
            "title": "Clean", "content": "Nothing to see.", "status": "publish"}).json()["data"]
        drain(client)

        preview = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/redact",
                             json={"patterns": [r"DE\d{2}(?:\s?\d{4}){4}\s?\d{2}"], "dry_run": True}).json()["data"]
        assert {i["knowledge_id"] for i in preview["items"]} == {url["id"], manual["id"]}
        done = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/redact",
                          json={"patterns": [r"DE\d{2}(?:\s?\d{4}){4}\s?\d{2}"]}).json()["data"]
        assert done["matches"] == preview["matches"] and not grep(alice, kb, "0532 0130")
        drain(client)

        item = alice.get(f"/kb/api/v1/knowledge/{manual['id']}").json()["data"]
        assert item["parse_status"] == "completed" and "0532" not in chunks_text(alice, manual["id"])
        page = alice.get(f"/kb/api/v1/knowledge/{url['id']}").json()["data"]
        assert page["refresh_interval"] == 0 and page["metadata"]["redacted"]
        # a refresh reads the redacted text, not the live page
        alice.post(f"/kb/api/v1/knowledge/{url['id']}/reparse")
        drain(client)
        assert "0532" not in chunks_text(alice, url["id"]) and "invoice number" in chunks_text(alice, url["id"])
        assert alice.get(f"/kb/api/v1/knowledge/{clean['id']}").json()["data"]["version"] == 1   # untouched
    finally:
        server.shutdown()


def test_redaction_needs_the_right_to_manage(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Admin", "visibility": "groups",
                                                        "groups": ["org:eic"]}).json()["data"]["id"]
    item = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={
        "title": "Bank", "content": f"Salary goes to {IBAN}.", "status": "publish", "visibility": "groups",
        "groups": ["org:eic"]}).json()["data"]
    drain(client)
    bob = Caller(client, user("bob", RW, org="eic"))
    assert bob.post(f"/kb/api/v1/knowledge/{item['id']}/redact", json={"terms": [IBAN]}).status_code == 403
    nothing = bob.post(f"/kb/api/v1/knowledge-bases/{kb}/redact", json={"terms": [IBAN]})
    assert nothing.status_code == 200, nothing.text
    nothing = nothing.json()["data"]
    assert nothing["matches"] == 0 and grep(alice, kb, "0532 0130")
    assert alice.post(f"/kb/api/v1/knowledge/{item['id']}/redact", json={"terms": []}).status_code == 400


def test_redaction_reaches_wiki_pages(client, alice):
    import sqlalchemy as sa

    from agenticwf.kb.core.orm.base.session import get_engine
    from agenticwf.kb.core.orm.models.ext import WikiPage, WikiPageRevision
    from agenticwf.kb.core.orm.models.kb import SOURCE_WIKI_PAGE, Embedding

    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Admin"}).json()["data"]["id"]
    with get_engine().begin() as conn:
        conn.execute(sa.insert(WikiPage.__table__).values(
            id="p1", knowledge_base_id=kb, slug="payments", title="Payments", summary=f"Account {IBAN}",
            content=f"Refunds go to {IBAN}.", page_type="entity", status="published", aliases=[], source_refs=[],
            source_ids=[], chunk_refs=[], in_links=[], out_links=[]))
        conn.execute(sa.insert(WikiPageRevision.__table__).values(
            id="r1", page_id="p1", version=1, title="Payments", summary="", content=f"Old: {IBAN}",
            page_type="entity", status="published", aliases=[], edit_source="pipeline", editor_id=""))
        conn.execute(sa.insert(Embedding.__table__).values(
            source_id="p1:0", source_type=SOURCE_WIKI_PAGE, chunk_id="wiki:p1:0", knowledge_id="wiki:p1",
            knowledge_base_id=kb, owner_subject="alice", visibility="all", groups=[], content=f"Refunds go to {IBAN}.",
            dimension=0, is_enabled=True))
    preview = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/redact", json={"terms": [IBAN], "dry_run": True}).json()
    assert preview["data"]["wiki"] == {"pages": 1, "matches": 4}
    alice.post(f"/kb/api/v1/knowledge-bases/{kb}/redact", json={"terms": [IBAN]})
    with get_engine().begin() as conn:
        texts = [*conn.execute(sa.select(WikiPage.summary, WikiPage.content)).one(),
                 conn.execute(sa.select(WikiPageRevision.content)).scalar_one(),
                 conn.execute(sa.select(Embedding.content).where(Embedding.knowledge_id == "wiki:p1")).scalar_one()]
    assert all("0532" not in t and "[REDACTED]" in t for t in texts)
