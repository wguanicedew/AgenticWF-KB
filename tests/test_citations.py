"""Source citations (docs/ingestion.md, Source citations): built at injection (supplied > the
document's own metadata > defaults), carried by search results with section and page, and
named by every agentic answer."""

import io
import json

import pytest

from conftest import Caller, drain, make_client, user
from kb_runtime import citation
from kb_runtime.ingestion.parsers import parse
from test_agentic import ScriptedChat, call, chat_model

RW = ("app:access", "knowledge:read", "knowledge:write")


def pdf(pages: list[str], info: dict | None = None) -> bytes:
    """A PDF with one text line per page and an Info dictionary, written by hand."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode())
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if info:
        objects.append(("<< " + " ".join(f"/{k} ({v})" for k, v in info.items()) + " >>").encode())
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    trailer = f"<< /Size {len(objects) + 1} /Root 1 0 R" + (f" /Info {len(objects)} 0 R" if info else "") + " >>"
    out.write(b"trailer\n" + trailer.encode() + b"\nstartxref\n%d\n%%%%EOF\n" % xref)
    return out.getvalue()


# --- units ------------------------------------------------------------------------------------------
def test_build_and_format():
    built = citation.build({"doi": "10.1000/xyz"}, {"title": "Pilot manual", "authors": ["A. Smith", "B. Jones"],
                                                    "published": "D:20250314120000Z"},
                           {"title": "pilot", "url": None})
    assert built["title"] == "Pilot manual" and built["published"] == "2025-03-14"
    assert built["text"] == "A. Smith, B. Jones (2025). Pilot manual. doi:10.1000/xyz"
    web = citation.build({}, {"title": "Status", "publisher": "CERN"},
                         {"title": "https://x.org", "url": "https://x.org/s", "accessed": "2026-10-17"})
    assert web["text"] == "Status. CERN. https://x.org/s (accessed 2026-10-17)"
    custom = citation.build({"text": "My own reference", "authors": "Ann; Bob"}, {"title": "T"}, {})
    assert custom["text"] == "My own reference" and custom["authors"] == ["Ann", "Bob"]
    assert citation.locate(built, "Setup > Restart", 3).endswith("[§ Setup > Restart, p. 3]")
    assert citation.locate(built, None, 3, 4).endswith("[pp. 3–4]")
    with pytest.raises(ValueError):
        citation.normalise_supplied({"published": "last tuesday"})


def test_pdf_metadata_and_pages():
    result = parse(pdf(["First page about calibration", "Second page about cryostat"],
                       {"Title": "Detector notes", "Author": "C. Expert", "CreationDate": "D:20240102"}), "n.pdf")
    assert result.metadata["title"] == "Detector notes" and result.metadata["authors"] == ["C. Expert"]
    assert result.metadata["published"] == "D:20240102"
    second = result.markdown.index("Second page")
    assert result.metadata["pages"][0] == 0 and result.metadata["pages"][1] == second


# --- end to end -----------------------------------------------------------------------------------------
@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


def ok(response, status=200):
    assert response.status_code in (status, 201), response.text
    return response.json()["data"]


def test_citations_from_injection_to_answer(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs", "chunking_config": {"chunk_size": 60,
                                                                                              "chunk_overlap": 0}}))["id"]
    # supplied at injection (flat fields), merged over the PDF's own metadata
    doc = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file",
                        files={"file": ("notes.pdf", io.BytesIO(pdf(["Calibration constants are in the database",
                                                                     "The cryostat alarm threshold is 1.2 bar"],
                                                                    {"Title": "Detector notes", "Author": "C. Expert",
                                                                     "CreationDate": "D:20240102"})), "application/pdf")},
                        data={"source_url": "https://edms.cern.ch/doc/42", "version": "v3"}))
    manual = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={
        "title": "Shift guide", "content": "# Shift guide\n\n## Alarms\n\nCall the cryo expert for cryostat alarms.",
        "citation": {"authors": ["Run Coordination"], "published": "2026-01-15", "publisher": "EIC"}}))
    drain(client)
    item = ok(alice.get(f"/kb/api/v1/knowledge/{doc['id']}"))
    c = item["citation"]
    assert c["title"] == "Detector notes" and c["authors"] == ["C. Expert"] and c["published"] == "2024-01-02"
    assert c["url"] == "https://edms.cern.ch/doc/42" and c["version"] == "v3"
    assert c["text"] == "C. Expert (2024). Detector notes. v3. https://edms.cern.ch/doc/42"
    hits = ok(alice.post("/kb/api/v1/knowledge-search", json={"query": "cryostat alarm threshold",
                                                              "knowledge_base_ids": [kb]}))
    pdf_hit = [h for h in hits if h["knowledge_id"] == doc["id"]][0]
    assert pdf_hit["citation"]["page"] == 2 and pdf_hit["citation_text"].endswith("[p. 2]")
    guide_hits = ok(alice.post("/kb/api/v1/knowledge-search", json={"query": "call the cryo expert",
                                                                     "knowledge_base_ids": [kb]}))
    guide_hit = [h for h in guide_hits if h["knowledge_id"] == manual["id"]][0]
    assert guide_hit["citation_text"].startswith("Run Coordination (2026). Shift guide. EIC")
    assert "§ " in guide_hit["citation_text"]
    # changing the citation later; null resets it to what the document says
    ok(alice.put(f"/kb/api/v1/knowledge/{doc['id']}", json={"citation": {"citation_text": "EDMS 42, rev. 3"}}))
    assert ok(alice.get(f"/kb/api/v1/knowledge/{doc['id']}"))["citation"]["text"] == "EDMS 42, rev. 3"
    ok(alice.put(f"/kb/api/v1/knowledge/{doc['id']}", json={"citation": None}))
    assert ok(alice.get(f"/kb/api/v1/knowledge/{doc['id']}"))["citation"]["text"].startswith("C. Expert (2024)")

    # an answer names its sources: inline (n, source, url) and in the footer and `sources`
    model = chat_model(alice)
    fake = ScriptedChat([{"tool_calls": [call("search_knowledge", queries=["cryostat alarm threshold"])]},
                         {"content": 'The threshold is 1.2 bar <ref id="c1"/>.'}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "cryostat alarm threshold?",
                                                            "knowledge_base_ids": [kb], "model_id": model}))
    first = data["sources"][0]
    assert first["n"] == 1 and first["cited"] is True and first["citation_text"]
    assert ' n="1" source="' in data["answer"] and "\n\nSources:\n[1] " in data["answer"]
    assert data["evidence"][0]["citation_text"]
    # an answer whose model cited nothing still lists what it was given
    fake2 = ScriptedChat([{"tool_calls": [call("grep_knowledge", pattern="cryostat")]},
                          {"content": "The threshold is 1.2 bar."}])

    async def chat2(ref):
        return fake2

    client.app.state.manager.services.models.chat = chat2
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "cryostat alarm threshold?",
                                                            "knowledge_base_ids": [kb], "model_id": model}))
    assert data["sources"] and all(not s["cited"] for s in data["sources"])
    assert all(s["citation_text"] for s in data["sources"])         # grep results got their citation
    assert "(consulted, not cited inline)" in data["answer"]


def test_url_items_cite_the_page(client, monkeypatch):
    import http.server
    import threading

    class Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = (b"<html><head><title>Status board</title><meta name='author' content='Ops Team'>"
                    b"<meta property='article:published_time' content='2026-09-01'></head><body><article>"
                    b"<h1>Status board</h1><p>The magnet is warm and the hall is open for access.</p>"
                    b"</article></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        alice = Caller(client, user("alice", RW, org="eic"))
        kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "Web"}))["id"]
        url = f"http://127.0.0.1:{server.server_address[1]}/status"
        item = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/url", json={"url": url}))
        drain(client)
        c = ok(alice.get(f"/kb/api/v1/knowledge/{item['id']}"))["citation"]
        assert c["title"] == "Status board" and c["url"] == url and c["accessed"]
        assert c["text"].endswith(f"{url} (accessed {c['accessed']})")
    finally:
        server.shutdown()
