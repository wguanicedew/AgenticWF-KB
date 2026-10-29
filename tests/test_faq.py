"""FAQ knowledge bases (docs/faq.md) and knowledge-grep (docs/agentic-search.md)."""

import io

import pytest

from conftest import Caller, drain, make_client, user
from agenticwf.kb.runtime.faq import rules

RW = ("app:access", "knowledge:read", "knowledge:write")


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def faq_kb(caller, **cfg):
    body = {"name": "Help", "type": "faq", **({"faq_config": cfg} if cfg else {})}
    response = caller.post("/kb/api/v1/knowledge-bases", json=body)
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


def entries(caller, kb_id, items, **body):
    response = caller.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries", json={"entries": items, **body})
    assert response.status_code == 200, response.text
    return response.json()["data"]


ENTRY = {"standard_question": "How do I reset my grid certificate?",
         "similar_questions": ["grid certificate reset", "renew certificate"],
         "negative_questions": ["how do I reset my password"],
         "answers": ["Open the CA portal and request a new certificate."]}


# --- rules -------------------------------------------------------------------------------------------
def test_normalise_and_clean_rules():
    assert rules.normalise("  How do I, RESET?! ") == rules.normalise("how do i reset")
    fields, warnings = rules.clean_entry({"standard_question": "Q one?", "answers": ["a"],
                                          "similar_questions": ["q ONE", "q two"],
                                          "negative_questions": ["Q two!", "q three"]})
    assert fields["similar_questions"] == ["q two"]          # equal to the standard question: dropped
    assert fields["negative_questions"] == ["q three"] and warnings
    with pytest.raises(Exception):
        rules.clean_entry({"standard_question": "x", "answers": []})
    merged, _ = rules.clean_entry({"similar_questions": ["new"], "answers": ["b"]},
                                  base={"standard_question": "S", "similar_questions": ["old"], "answers": ["a"]},
                                  merge=True)
    assert merged["similar_questions"] == ["old", "new"] and merged["answers"] == ["b"]


def test_index_texts_per_config():
    fields = {"standard_question": "S?", "similar_questions": ["A?", "B?"], "answers": ["yes"]}
    combined = rules.index_texts("c1", fields, {"index_mode": "question_answer", "question_index_mode": "combined"})
    assert combined == [("c1", 0, "S?\nA?\nB?\nyes")]
    separate = rules.index_texts("c1", fields, {"index_mode": "question_only", "question_index_mode": "separate"})
    assert [r[0] for r in separate] == ["c1", "c1:s0", "c1:s1"] and separate[1] == ("c1:s0", 1, "A?")


def test_priority_and_exact_marker():
    items = [{"id": "d", "chunk_type": "text", "score": 0.95, "knowledge_id": "k", "chunk_index": 0},
             {"id": "f", "chunk_type": "faq", "score": 0.91, "knowledge_id": "k2", "chunk_index": 0},
             {"id": "g", "chunk_type": "faq", "score": 0.5, "knowledge_id": "k2", "chunk_index": 1}]
    out = rules.apply_priority(items, 2, {"priority_enabled": True, "score_boost": 1.2,
                                          "direct_answer_threshold": 0.9})
    assert [i["id"] for i in out] == ["f", "d"] and out[0]["match"] == "exact" and out[0]["score"] == 0.91


# --- API ---------------------------------------------------------------------------------------------
def test_faq_upsert_rules_and_reports(client, alice):
    kb_id = faq_kb(alice)
    dry = entries(alice, kb_id, [ENTRY], dry_run=True)
    assert dry["created"] == 1 and dry["dry_run"]
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries").json()["total"] == 0
    first = entries(alice, kb_id, [ENTRY])
    assert first["created"] == 1 and first["entries"][0]["id"]
    entry_id = first["entries"][0]["id"]
    again = entries(alice, kb_id, [ENTRY])
    assert again["unchanged"] == 1
    clash = entries(alice, kb_id, [{"standard_question": "Renew certificate", "answers": ["x"]},
                                   {"standard_question": "Where is the CA?", "similar_questions": ["grid certificate reset"],
                                    "answers": ["CERN CA"]}])
    assert clash["failed"][0]["index"] == 0 and clash["created"] == 1
    assert clash["warnings"][0]["index"] == 1
    appended = entries(alice, kb_id, [{"standard_question": ENTRY["standard_question"], "similar_questions": ["reset cert"],
                                       "answers": ["Use the new portal."]}])
    assert appended["updated"] == 1
    entry = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries/{entry_id}").json()["data"]
    assert entry["answers"] == ["Use the new portal."] and "reset cert" in entry["similar_questions"]
    assert entry["version"] == 2
    one = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entry", json=ENTRY)
    assert one.status_code == 400
    replaced = entries(alice, kb_id, [ENTRY], mode="replace")
    assert replaced["deleted"] == 1 and replaced["updated"] == 1
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries").json()["total"] == 1


def test_faq_search_negative_questions_and_priority(client, alice):
    kb_id = faq_kb(alice)
    entries(alice, kb_id, [ENTRY])
    hits = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/search",
                      json={"query_text": "grid certificate reset"}).json()["data"]
    assert hits and hits[0]["matched_question"] == "grid certificate reset"
    assert hits[0]["answers"] == ENTRY["answers"]
    # a negative question never returns the entry, on every search path
    for path, body in ((f"/kb/api/v1/knowledge-bases/{kb_id}/faq/search", {"query_text": "How do I reset my password?"}),
                       ("/kb/api/v1/knowledge-search", {"query": "How do I reset my password?",
                                                        "knowledge_base_ids": [kb_id]})):
        assert alice.post(path, json=body).json()["data"] == []
    mixed = alice.post("/kb/api/v1/knowledge-search", json={"query": "How do I reset my grid certificate?",
                                                            "knowledge_base_ids": [kb_id]}).json()["data"]
    assert mixed[0]["chunk_type"] == "faq" and mixed[0]["match"] == "exact"


def test_faq_fields_tags_delete_and_reindex(client, alice):
    kb_id = faq_kb(alice)
    tag = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/tags", json={"name": "grid"}).json()["data"]
    made = entries(alice, kb_id, [ENTRY, {"standard_question": "What is Rucio?", "answers": ["Data management."],
                                          "tag_id": tag["id"]}])["entries"]
    first, second = made[0]["id"], made[1]["id"]
    tagged = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries?tag_id={tag['seq_id']}").json()["data"]
    assert [e["id"] for e in tagged] == [second]
    untagged = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries?tag_id=__untagged__").json()["data"]
    assert [e["id"] for e in untagged] == [first]
    search = {"query": "Rucio", "knowledge_base_ids": [kb_id], "tag_ids": [tag["id"]]}
    assert alice.post("/kb/api/v1/knowledge-search", json=search).json()["data"]
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries/tags", json={"updates": {str(second): None}})
    assert alice.post("/kb/api/v1/knowledge-search", json=search).json()["data"] == []
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries/fields",
              json={"by_id": {str(first): {"is_enabled": False}}})
    assert alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/search",
                      json={"query_text": "grid certificate reset"}).json()["data"] == []
    response = alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}",
                         json={"faq_config": {"question_index_mode": "separate", "index_mode": "question_only"}})
    assert response.json()["data"]["faq_config"]["question_index_mode"] == "separate"
    from conftest import sql_rows
    rows = sql_rows("SELECT source_id, is_enabled FROM kb_embeddings WHERE knowledge_base_id = :k ORDER BY source_id", k=kb_id)
    assert len(rows) == 4 and sum(1 for _, enabled in rows if not enabled) == 3    # first: standard + 2 similar
    alice.request("DELETE", f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries", json={"ids": [first]})
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries").json()["total"] == 1


def test_faq_import_export_round_trip(client, alice):
    kb_id = faq_kb(alice)
    csv = ("问题,相似问题,反例问题,机器人回答,分类,是否全部回复(TRUE/FALSE)\n"
           "How do I submit a job?,submit job##job submission,,Use the CLI.##Or the web UI.,ops,TRUE\n"
           "What is PanDA?,,,A workload manager.,,FALSE\n").encode("utf-8-sig")
    response = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/import",
                          files={"file": ("faq.csv", io.BytesIO(csv), "text/csv")}, data={"mode": "append"})
    report = response.json()["data"]
    assert report["created"] == 2 and not report["failed"], report
    first = report["entries"][0]
    assert first["similar_questions"] == ["submit job", "job submission"] and first["tag_id"]
    assert report["entries"][1]["answer_strategy"] == "random"
    exported = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries/export")
    assert exported.content.startswith("﻿".encode())
    again = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/import",
                       files={"file": ("faq.csv", io.BytesIO(exported.content), "text/csv")},
                       data={"mode": "replace"}).json()["data"]
    assert again["unchanged"] == 2 and again["created"] == again["updated"] == again["deleted"] == 0


def test_documents_refused_and_other_owner_404(client, alice):
    kb_id = faq_kb(alice)
    upload = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/file",
                        files={"file": ("a.md", io.BytesIO(b"# A\n\nx\n"), "text/markdown")})
    assert upload.status_code == 400
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries").status_code == 404
    assert bob.post(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries", json={"entries": [ENTRY]}).status_code == 404


# --- grep --------------------------------------------------------------------------------------------
def test_grep(client, alice):
    kb_id = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    doc = alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/file",
                     files={"file": ("errors.md", io.BytesIO(
                         b"# Errors\n\nIntro line.\nError E-1042: stage-out failed (retry_count=3).\nNext line.\n"),
                         "text/markdown")}).json()["data"]
    drain(client)

    def grep(**body):
        response = alice.post("/kb/api/v1/knowledge-grep", json={"knowledge_base_ids": [kb_id], **body})
        assert response.status_code == 200, response.text
        return response.json()["data"]

    found = grep(pattern=r"E-\d{4}")
    assert found["mode"] == "regex" and found["matches"][0]["knowledge_id"] == doc["id"]
    assert "Intro line." in found["matches"][0]["lines"][0]["text"]
    assert grep(pattern="retry_count=3", regex=False)["matches"]
    assert not grep(pattern="error e-1042", case_sensitive=True)["matches"]
    assert grep(pattern="error e-1042")["matches"]
    fallback = grep(pattern="(retry_count=3")                 # invalid regex: literal
    assert fallback["mode"] == "literal" and fallback["matches"]
    alice.put(f"/kb/api/v1/knowledge/{doc['id']}", json={"enable_status": "disabled"})
    assert not grep(pattern="E-1042")["matches"]
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.post("/kb/api/v1/knowledge-grep", json={"pattern": "E-1042"}).json()["data"]["matches"] == []


# --- the management page -------------------------------------------------------------------------------
def test_management_page(client, alice):
    page = alice.get("/kb/")
    assert page.status_code == 200 and '<meta name="kb-base" content="/kb">' in page.text
    assert "{{" not in page.text and "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "Not signed in" in client.get("/kb/").text


def test_faq_feature_switch(client, alice, monkeypatch):
    kb_id = faq_kb(alice)
    from agenticwf.kb.runtime import features
    monkeypatch.setattr(features, "enabled", lambda name: name != "faq")
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/faq/entries").status_code == 404
    assert alice.post("/kb/api/v1/knowledge-bases", json={"name": "x", "type": "faq"}).status_code == 400
