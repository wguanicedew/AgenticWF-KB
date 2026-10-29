"""The LLM wiki (docs/llm-wiki.md): text helpers, the ingest pipeline with a scripted LLM
(creation, merging, dedup, truncation, retraction), finalize, revisions, the API, search,
global search, communities, access and the wiki preset."""

import asyncio
import io
import json
import re

import pytest

from conftest import Caller, drain, make_client, sql_rows, user
from agenticwf.kb.runtime.wiki import text as wt

RW = ("app:access", "knowledge:read", "knowledge:write")


# --- text helpers ------------------------------------------------------------------------------------
def test_slugs_handles_and_links():
    assert wt.page_slug("entity", "Rucio DMS (v2)") == "entity/rucio-dms-v2"
    assert wt.normalise_slug("Entity/Café Crème") == "entity/cafe-creme"
    assert wt.normalise_slug("data placement") == "concept/data-placement"
    assert wt.slugify("数据 管理") == "数据-管理"
    handles = wt.Handles(["entity/rucio", "concept/x"])
    assert handles.restore("See [[ref-1|Rucio]] and [[ref-9|Gone]] and [[ref-2]].") == \
        "See [[entity/rucio|Rucio]] and Gone and [[concept/x]]."
    text, dead = wt.strip_dead_links("[[entity/a|A]] [[entity/b|B]] [[community/l0-x|C]]", {"entity/a"})
    assert text == "[[entity/a|A]] B [[community/l0-x|C]]" and dead == ["entity/b"]
    linked = wt.inject_cross_links("# Rucio\nRucio moves data with FTS. FTS again. 云 here.", "entity/rucio",
                                   [("entity/fts", ["FTS"]), ("entity/rucio", ["Rucio"]), ("entity/yun", ["云"])])
    assert linked == "# Rucio\nRucio moves data with [[entity/fts|FTS]]. FTS again. 云 here."
    assert wt.split_summary("SUMMARY: Short.\n\n# Body\ntext") == ("Short.", "# Body\ntext")
    assert wt.replace_links("[[a|A]] [[b]]", {"a": "c", "b": None}) == "[[c|A]] b"


# --- the scripted LLM ------------------------------------------------------------------------------------
class WikiLLM:
    """Answers each wiki pass from its prompt. Candidates come from the document title."""

    CANDIDATES = {
        "rucio": [{"slug": "entity/rucio", "title": "Rucio", "aliases": ["Rucio DMS"]},
                  {"slug": "concept/data-placement", "title": "Data placement"}],
        "fts": [{"slug": "entity/fts", "title": "FTS", "aliases": ["File Transfer Service"]},
                {"slug": "entity/rucio", "title": "Rucio"}],
        "panda": [{"slug": "entity/panda", "title": "PanDA"}, {"slug": "entity/rucio-dms", "title": "Rucio DMS"},
                  {"slug": "concept/long-page", "title": "Long page"}],
        "twins a": [{"slug": "concept/heartbeat", "title": "Heartbeat"}],
        "twins b": [{"slug": "entity/heartbeat", "title": "Heartbeat"}],
    }

    def __init__(self):
        self.calls = []

    async def complete(self, messages, max_tokens=None, temperature=0.2, tools=None, **kw):
        system, user_text = messages[0]["content"], messages[-1]["content"]
        self.calls.append(system[:40])
        if "You plan wiki pages" in system:
            title = re.search(r"Document: (.*)", user_text).group(1).strip().lower()
            return self._json({"candidates": self.CANDIDATES.get(title, [])})
        if "You file new wiki pages" in system:
            slugs = re.findall(r"^(\S+?):", user_text.split("New pages:")[1], re.MULTILINE)
            if any("heartbeat" in s for s in slugs):        # a small model echoing the page as its folder
                return self._json({"folders": {s: "Concept/Heartbeat" for s in slugs}})
            return self._json({"folders": {s: "Systems/Data" for s in slugs}})
        if "You link passages" in system:
            refs = re.findall(r"^(ref-\d+):", user_text, re.MULTILINE)
            labels = re.findall(r"\[(c\d{3})\]", user_text)
            relations = [{"source": refs[0], "relation": "uses", "target": refs[1], "chunks": labels[:1]}] \
                if len(refs) > 1 else []
            return self._json({"citations": {r: labels for r in refs}, "new_slugs": [], "relations": relations})
        if "You decide whether newly proposed" in system:
            merges = {"entity/rucio-dms": "entity/rucio"} if "entity/rucio-dms" in user_text else {}
            return self._json({"merges": merges})
        if "You write one wiki page" in system or "Your page was cut off" in user_text:
            first_user = messages[1]["content"]
            title = re.search(r"Page: (.*?) \(", first_user).group(1)
            if title == "Long page":
                return {"content": "SUMMARY: cut", "tool_calls": [], "usage": {}, "finish_reason": "length"}
            labels = re.findall(r"\[(c\d{3})\]", first_user)
            refs = [r for r in re.findall(r"^(ref-\d+):", first_user.split("Other pages you may link:")[1],
                                          re.MULTILINE)]
            self_ref = re.search(r"\((ref-\d+),", first_user).group(1)
            keep = " ".join(re.findall(r"\[src:\d+\]", first_user.split("Existing page:")[1].split("Other pages")[0]))
            link = f" Related: [[{refs[0]}|Other page]]." if refs else ""
            body = (f"# {title}\n\n{title} is covered in the sources [{labels[0]}].{link} "
                    f"Self [[{self_ref}|me]]. {keep} An invented citation [src:999999].")
            return {"content": f"SUMMARY: About {title}.\n\n{body}", "tool_calls": [], "usage": {},
                    "finish_reason": "stop"}
        if "You write the page of one community" in system:
            return {"content": "SUMMARY: A community.\n\n## Key entities\nSome members.", "tool_calls": [],
                    "usage": {}, "finish_reason": "stop"}
        if "You answer a question from one community summary" in system:
            slug = re.search(r"Summary \((.*?)\)", user_text).group(1)
            return self._json({"answer": f"{slug} is relevant", "score": 80 if "summary/" in slug else 10})
        if "You combine partial answers" in system:
            slugs = re.findall(r"\[\[(.*?)\|", user_text)
            return {"content": "Overall: " + ", ".join(f"[[{s}|page]]" for s in slugs), "tool_calls": [],
                    "usage": {}, "finish_reason": "stop"}
        return {"content": "{}", "tool_calls": [], "usage": {}, "finish_reason": "stop"}

    @staticmethod
    def _json(value):
        return {"content": json.dumps(value), "tool_calls": [], "usage": {}, "finish_reason": "stop"}


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        llm = WikiLLM()

        async def chat(ref):
            return llm

        c.app.state.manager.services.models.chat = chat
        c.llm = llm
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def ok(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def wiki_kb(alice, visibility="private", groups=()):
    model = ok(alice.post("/kb/api/v1/models", json={"name": "wiki-chat", "type": "KnowledgeQA", "parameters": {
        "base_url": "http://127.0.0.1:9/v1", "supports_tools": True}}), 201)["id"]
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "Ops", "visibility": visibility, "groups": list(groups), "default_item_visibility": "all",
        "indexing_strategy": {"wiki_enabled": True}, "wiki_config": {"synthesis_model_id": model}}), 201)
    return kb["id"], model


def upload(alice, kb_id, title, text):
    return ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/manual",
                         json={"title": title, "content": text, "visibility": "all"}), 201)


def page(alice, kb_id, slug, status=200):
    response = alice.get(f"/kb/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug}")
    assert response.status_code == status, response.text
    return response.json()["data"] if status == 200 else None


W = "/kb/api/v1/knowledgebase/{}/wiki"


# --- the pipeline -------------------------------------------------------------------------------------------
def test_ingest_merge_dedup_truncation_and_retraction(client, alice):
    kb_id, _ = wiki_kb(alice)
    doc1 = upload(alice, kb_id, "Rucio", "# Rucio\n\nRucio places data on storage sites.\n\nIt uses rules.")
    drain(client)
    assert ok(alice.get(f"/kb/api/v1/knowledge/{doc1['id']}"))["parse_status"] == "completed"
    rucio = page(alice, kb_id, "entity/rucio")
    assert rucio["summary"] == "About Rucio." and re.search(r"\[src:\d+\]", rucio["content"])
    assert "[[concept/data-placement|" in rucio["content"] and "ref-" not in rucio["content"]
    assert "[[entity/rucio|" not in rucio["content"]                      # no self-links
    assert rucio["source_refs"] == [f"{doc1['id']}|Rucio"] and rucio["category_path"] == "Systems/Data"
    assert rucio["aliases"] == ["Rucio DMS"]
    assert page(alice, kb_id, f"summary/{doc1['id']}")["page_type"] == "summary"
    index = ok(alice.get(W.format(kb_id) + "/index"))
    assert "## Systems/Data" in index["content"] and "[[entity/rucio|Rucio]]" in index["content"]
    relations = ok(alice.get(W.format(kb_id) + "/relations"))
    assert relations and relations[0]["relation"] == "uses" and relations[0]["chunk_refs"]

    quality = ok(alice.get(W.format(kb_id) + "/quality"))
    assert quality["checks"] == {"every_page_has_a_source": True, "no_dead_links": True, "citations_resolve": True,
                                 "every_relation_cites_a_chunk": True, "settled": True}
    assert quality["document_coverage"] == 1.0 and quality["cited_paragraphs"] > 0
    assert "[src:999999]" not in rucio["content"]
    doc2 = upload(alice, kb_id, "FTS", "# FTS\n\nFTS transfers files for Rucio.")
    drain(client)
    rucio = page(alice, kb_id, "entity/rucio")
    assert set(rucio["source_ids"]) == {doc1["id"], doc2["id"]} and rucio["version"] >= 2
    assert len(re.findall(r"\[src:\d+\]", rucio["content"])) >= 2            # old citations kept, new added
    history = ok(alice.get(W.format(kb_id) + "/revisions/entity/rucio"))
    assert history["revisions"] and history["current_version"] == rucio["version"]

    doc3 = upload(alice, kb_id, "PanDA", "# PanDA\n\nPanDA schedules jobs and asks Rucio DMS for data.")
    drain(client)
    assert page(alice, kb_id, "entity/rucio-dms", 404) is None                 # deduplicated into entity/rucio
    assert doc3["id"] in page(alice, kb_id, "entity/rucio")["source_ids"]
    assert page(alice, kb_id, "concept/long-page", 404) is None                # truncated 4×: never written
    assert page(alice, kb_id, "entity/panda")

    alice.delete(f"/kb/api/v1/knowledge/{doc1['id']}")
    drain(client)
    rucio = page(alice, kb_id, "entity/rucio")
    assert doc1["id"] not in rucio["source_ids"]
    placement = alice.get(W.format(kb_id) + "/pages?status=archived").json()["data"]
    assert "concept/data-placement" in [p["slug"] for p in placement]       # only doc1 cited it
    assert all(doc1["id"] not in r["chunk_refs"][0] for r in ok(alice.get(W.format(kb_id) + "/relations")))


def test_same_batch_twins_and_echoed_folders(client, alice):
    kb_id, _ = wiki_kb(alice)
    kb = client.app.state.manager.services.wiki
    upload(alice, kb_id, "Twins A", "# A\n\nThe heartbeat is sent every 30 minutes.")
    upload(alice, kb_id, "Twins B", "# B\n\nThe heartbeat tells PanDA the pilot is alive.")
    drain(client)
    slugs = [p["slug"] for p in alice.get(W.format(kb_id) + "/pages?page_size=200").json()["data"]]
    assert ("concept/heartbeat" in slugs) != ("entity/heartbeat" in slugs)        # one page, not twins
    heartbeat = page(alice, kb_id, [s for s in slugs if s.endswith("/heartbeat")][0])
    assert len(heartbeat["source_ids"]) == 2 and heartbeat["category_path"] == "Concepts"
    assert kb is not None


def test_private_items_never_reach_the_wiki(client, alice):
    """In a shared KB a private item has fewer readers than the wiki; it never reaches a page.
    (In a private KB the owner is the only reader, so the owner's items do feed it.)"""
    kb_id, _ = wiki_kb(alice, "groups", ["org:eic"])
    ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/knowledge/manual",
                  json={"title": "Rucio", "content": "# Rucio\n\nsecret", "visibility": "private"}), 201)
    drain(client)
    assert ok(alice.get(W.format(kb_id) + "/stats"))["pages"] == 0


def test_pending_op_claims_are_disjoint(client, alice):
    kb_id, _ = wiki_kb(alice)
    for i in range(6):
        sql_rows("INSERT INTO kb_wiki_pending_ops (knowledge_base_id, knowledge_id, op, payload, fail_count, created_at) "
                 "VALUES (:k, :i, 'ingest', '{}', 0, now()) RETURNING id", k=kb_id, i=f"doc-{i}")
    pipeline = client.app.state.manager.services.wiki_pipeline
    from agenticwf.kb.core.orm.base.session import dispose_async_engine

    async def both():
        try:
            return await asyncio.gather(pipeline._claim(kb_id, 4), pipeline._claim(kb_id, 4))
        finally:
            await dispose_async_engine()
    first, second = asyncio.run(both())
    ids_a, ids_b = {o["id"] for o in first}, {o["id"] for o in second}
    assert not ids_a & ids_b and len(ids_a | ids_b) == 6


# --- the API -----------------------------------------------------------------------------------------------
def test_api_pages_revisions_lint_search_and_access(client, alice):
    kb_id, _ = wiki_kb(alice)
    upload(alice, kb_id, "Rucio", "# Rucio\n\nRucio places data on storage sites.")
    drain(client)
    base = W.format(kb_id)
    created = ok(alice.post(base + "/pages", json={"title": "Rucio vs FTS", "page_type": "comparison",
                                                   "content": "Compare [[entity/rucio|Rucio]] with FTS.",
                                                   "folder": "Comparisons"}), 201)
    assert created["slug"] == "comparison/rucio-vs-fts" and created["category_path"] == "Comparisons"
    assert alice.post(base + "/pages", json={"title": "x", "page_type": "index", "content": "x"}).status_code == 400
    before = page(alice, kb_id, "entity/rucio")
    renamed = ok(alice.put(base + "/pages/entity/rucio", json={"slug": "entity/rucio-dms", "content": "Edited."}))
    assert renamed["slug"] == "entity/rucio-dms" and renamed["last_edit_source"] == "user"
    assert "[[entity/rucio-dms|Rucio]]" in page(alice, kb_id, "comparison/rucio-vs-fts")["content"]
    reverted = ok(alice.post(base + "/revert", json={"slug": "entity/rucio-dms", "version": before["version"]}))
    assert reverted["last_edit_source"] == "revert" and reverted["content"] == before["content"]
    assert alice.post(base + "/revert", json={"slug": "entity/rucio-dms",
                                              "version": reverted["version"]}).status_code == 400
    old = ok(alice.get(base + f"/revisions/entity/rucio-dms?version={renamed['version']}"))
    assert old["content"] == "Edited."
    issue = ok(alice.post(base + "/issues", json={"slug": "entity/rucio-dms", "issue_type": "out_of_date",
                                                  "description": "old"}), 201)
    assert ok(alice.put(base + f"/issues/{issue['id']}/status", json={"status": "resolved"}))["status"] == "resolved"
    lint = ok(alice.get(base + "/lint"))
    assert {"slug": "comparison/rucio-vs-fts", "check": "no_summary"} in lint["findings"]
    found = ok(alice.get(base + "/search?q=%5ERucio%24"))
    assert [p["slug"] for p in found][0] == "entity/rucio-dms"                # a title hit ranks first
    assert ok(alice.get(base + "/search?q=(unclosed"))  is not None            # invalid regex: literal
    graph = ok(alice.get(base + "/graph?slug=entity/rucio-dms"))
    assert {e["source"] for e in graph["edges"]} >= {"comparison/rucio-vs-fts"}
    hits = ok(alice.post("/kb/api/v1/knowledge-search", json={"query": "Rucio vs FTS compare",
                                                              "knowledge_base_ids": [kb_id]}))
    assert any(h["chunk_type"] == "wiki_page" and h["wiki_slug"] == "comparison/rucio-vs-fts" for h in hits)
    ok(alice.delete(base + "/pages/comparison/rucio-vs-fts"))
    assert alice.get(base + "/pages/comparison/rucio-vs-fts").status_code == 404
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.get(base + "/pages/entity/rucio-dms").status_code == 404
    assert bob.get(base + "/stats").status_code == 404


def test_revision_retention(client, alice):
    kb_id, _ = wiki_kb(alice)
    base = W.format(kb_id)
    ok(alice.post(base + "/pages", json={"title": "Notes", "content": "v1"}), 201)
    kb = client.app.state.manager.services.wiki
    kb_cfg = {"revision_soft_cap": 3, "revision_hard_cap": 5}
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}", json={"wiki_config": {**kb_cfg}})
    for i in range(8):
        ok(alice.put(base + "/pages/synthesis/notes", json={"content": f"v{i + 2}"}))
    revisions = ok(alice.get(base + "/revisions/synthesis/notes"))["revisions"]
    assert len(revisions) == 5                       # user edits: only the hard cap prunes them
    assert kb is not None


def test_global_search_and_communities(client, alice):
    kb_id, _ = wiki_kb(alice)
    upload(alice, kb_id, "Rucio", "# Rucio\n\nRucio places data on storage sites.")
    upload(alice, kb_id, "FTS", "# FTS\n\nFTS transfers files for Rucio.")
    drain(client)
    communities = ok(alice.get(W.format(kb_id) + "/communities"))
    result = ok(alice.post(W.format(kb_id) + "/global-search", json={"query": "What does the corpus cover?"}))
    level0 = {c["page_slug"] for c in communities if c["level"] == 0}
    assert result["answer"].startswith("Overall: [[") and set(result["sources"]) <= level0
    assert communities and all(c["page_slug"].startswith("community/") for c in communities)
    community = page(alice, kb_id, communities[0]["page_slug"])
    assert community["page_type"] == "community" and community["summary"] == "A community."


def test_wiki_preset_reads_pages_and_cites_sources(client, alice):
    from test_agentic import ScriptedChat, call

    kb_id, model = wiki_kb(alice)
    upload(alice, kb_id, "Rucio", "# Rucio\n\nRucio places data on storage sites.")
    drain(client)
    fake = ScriptedChat([{"tool_calls": [call("wiki_search", query="rucio")]},
                         {"tool_calls": [call("wiki_read_page", slugs=["p1"])]},
                         {"content": 'Rucio places data <ref id="c1"/>, see <ref id="p1"/>.'}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "what does rucio do", "preset": "wiki",
                                                            "knowledge_base_ids": [kb_id], "model_id": model}))
    assert "wiki_search" in fake.calls[0]["messages"][0]["content"] and fake.calls[0]["temperature"] == 0.5
    read = [m["content"] for m in fake.calls[2]["messages"] if m["role"] == "tool"][1]
    assert '<wiki_page id="p1" slug="entity/rucio"' in read and '<source id="c1" doc="d1"' in read
    assert '<wiki slug="entity/rucio"' in data["answer"] and "<kb doc=" in data["answer"]
