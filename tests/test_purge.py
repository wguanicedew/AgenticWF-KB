"""Deleting for good: a deleted item keeps no content, `purge` erases rows at once, housekeeping
purges deleted knowledge bases after the retention period."""

import asyncio
import io

import pytest
import sqlalchemy as sa

from conftest import Caller, drain, make_client, user
from agenticwf.kb.core.orm.base.session import dispose_async_engine, get_engine

RW = ("app:access", "knowledge:read", "knowledge:write")
DOC = b"# Expenses\n\nRefunds go to the account DE89 3704 0044 0532 0130 00.\n"


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def rows(sql, **params):
    with get_engine().connect() as conn:
        return conn.execute(sa.text(sql), params).all()


def setup_kb(client, alice):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Admin"}).json()["data"]["id"]
    tag = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/tags", json={"name": "money"}).json()["data"]["id"]
    item = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file", data={"tag_ids": tag},
                      files={"file": ("expenses.md", io.BytesIO(DOC), "text/markdown")}).json()["data"]
    manual = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual", json={
        "title": "Bank", "content": "Salary goes to DE89 3704 0044 0532 0130 00.", "status": "publish"}).json()["data"]
    drain(client)
    return kb, item, manual


def test_a_deleted_item_keeps_no_content_and_purge_erases_it(client, alice):
    kb, item, manual = setup_kb(client, alice)
    assert alice.delete(f"/kb/api/v1/knowledge/{manual['id']}").json()["data"]["purged"] is False
    (row,) = rows("SELECT deleted_at, description, metadata::text, citation::text FROM kb_knowledges WHERE id = :i",
                  i=manual["id"])
    assert row[0] is not None and row[1] == "" and row[2] == "{}" and row[3] == "{}"
    assert not rows("SELECT 1 FROM kb_chunks WHERE knowledge_id = :i", i=manual["id"])

    assert alice.delete(f"/kb/api/v1/knowledge/{item['id']}?purge=true").json()["data"]["purged"] is True
    assert not rows("SELECT 1 FROM kb_knowledges WHERE id = :i", i=item["id"])
    assert not rows("SELECT 1 FROM kb_knowledge_tag_relations WHERE knowledge_id = :i", i=item["id"])
    assert alice.get(f"/kb/api/v1/knowledge/{item['id']}").status_code == 404
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb}/knowledge").json()["data"] == []


def test_purging_a_knowledge_base_erases_every_row(client, alice):
    from agenticwf.kb.core.orm.models.ext import WikiPage

    kb, item, _ = setup_kb(client, alice)
    with get_engine().begin() as conn:
        conn.execute(sa.insert(WikiPage.__table__).values(
            id="p1", knowledge_base_id=kb, slug="money", title="Money", summary="", content="DE89 3704",
            page_type="entity", status="published", aliases=[], source_refs=[], source_ids=[item["id"]],
            chunk_refs=[], in_links=[], out_links=[]))
    other = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Other"}).json()["data"]["id"]
    response = alice.delete(f"/kb/api/v1/knowledge-bases/{kb}?purge=true")
    assert response.status_code == 200 and response.json()["data"]["purged"] is True
    for table in ("kb_knowledges", "kb_knowledge_tags", "kb_wiki_pages", "kb_embeddings", "kb_chunks", "kb_sites"):
        assert not rows(f"SELECT 1 FROM {table} WHERE knowledge_base_id = :k", k=kb), table
    assert not rows("SELECT 1 FROM kb_knowledge_bases WHERE id = :k", k=kb)
    assert rows("SELECT 1 FROM kb_knowledge_bases WHERE id = :k", k=other)     # nothing else


def test_housekeeping_purges_deleted_knowledge_bases(client, alice):
    kb, item, _ = setup_kb(client, alice)
    alice.delete(f"/kb/api/v1/knowledge-bases/{kb}")
    assert rows("SELECT 1 FROM kb_knowledge_bases WHERE id = :k", k=kb)          # soft-deleted, kept for now
    assert rows("SELECT description FROM kb_knowledges WHERE id = :i", i=item["id"])[0][0] == ""
    freshness = client.app.state.manager.services.freshness

    async def run(**kw):
        try:
            return await freshness.housekeeping(**kw)
        finally:
            await dispose_async_engine()

    assert asyncio.run(run(dry_run=False))["purged_knowledge_bases"] == 0     # within purge_after_days
    report = asyncio.run(run(dry_run=False, retention_days=0))
    assert report["purged_knowledge_bases"] == 1
    assert not rows("SELECT 1 FROM kb_knowledge_bases WHERE id = :k", k=kb)
    assert not rows("SELECT 1 FROM kb_knowledges WHERE knowledge_base_id = :k", k=kb)


def test_purging_a_site_and_its_pages(client, alice, monkeypatch):
    from agenticwf.kb.core.orm.models.kb import KnowledgeSite, Knowledge, new_id

    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Web"}).json()["data"]["id"]
    site_id, page_id = new_id(), new_id()
    with get_engine().begin() as conn:
        conn.execute(sa.insert(KnowledgeSite.__table__).values(
            id=site_id, knowledge_base_id=kb, owner_subject="alice", url="https://h.org/docs/", title="Docs",
            config={}, status="completed", error_message="", stats={}, refresh_interval=0, visibility="private",
            groups=[], created_at=sa.func.now(), updated_at=sa.func.now()))
        conn.execute(sa.insert(Knowledge.__table__).values(
            id=page_id, knowledge_base_id=kb, owner_subject="alice", type="url", source="https://h.org/docs/",
            title="Docs", visibility="private", groups=[], site_id=site_id, parse_status="completed",
            created_at=sa.func.now(), updated_at=sa.func.now()))
    result = alice.delete(f"/kb/api/v1/sites/{site_id}?purge=true").json()["data"]
    assert result["pages_deleted"] == 1
    assert not rows("SELECT 1 FROM kb_sites WHERE id = :s", s=site_id)
    assert not rows("SELECT 1 FROM kb_knowledges WHERE id = :p", p=page_id)
