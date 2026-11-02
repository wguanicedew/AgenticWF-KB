"""Sharing requests: owners ask to share a KB more widely; an admin or the groups' admins decide."""

import dataclasses
import io

import pytest

from conftest import Caller, drain, make_client, user
from agenticwf.kb.runtime.services import sharing

RW = ("app:access", "knowledge:read", "knowledge:write")
DOC = b"# Beamline\n\nThe AID2E optimiser runs every night.\n"


def member(subject, orgs, managed=(), scopes=RW):
    """A user in authsvc groups (`orgs`: id -> name), administering `managed` of them."""
    return dataclasses.replace(user(subject, scopes), organizations=dict(orgs),
                               managed_organizations=frozenset(managed))


# --- the rules (no database) ----------------------------------------------------------------------
def test_who_shares_directly_and_who_approves():
    owner = member("owner", {"eic": "EIC"})
    head = member("head", {"eic": "EIC", "atlas": "ATLAS"}, managed=["atlas"])
    admin = user("root", RW + ("admin",))
    assert sharing.may_share_directly(owner, "groups", ["org:eic"])          # a group they are in
    assert not sharing.may_share_directly(owner, "groups", ["org:atlas"])    # one they are not
    assert not sharing.may_share_directly(owner, "all", [])                  # everyone: admins only
    assert sharing.may_share_directly(admin, "all", [])

    assert sharing.can_approve(head, "groups", ["org:atlas"])
    assert not sharing.can_approve(head, "groups", ["org:atlas", "org:eic"])  # every group asked for
    assert not sharing.can_approve(head, "all", [])
    assert not sharing.can_approve(head, "groups", ["role:shifter"])          # not an authsvc group
    assert sharing.can_approve(admin, "all", []) and sharing.can_approve(admin, "groups", ["role:shifter"])


# --- the flow -------------------------------------------------------------------------------------
@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


def test_request_approve_and_decline(client):
    owner = Caller(client, member("owner", {"eic": "EIC"}))
    head = Caller(client, member("head", {"atlas": "ATLAS"}, managed=["atlas"]))
    reader = Caller(client, member("reader", {"atlas": "ATLAS"}))
    admin = Caller(client, user("root", RW + ("admin",)))
    kb = owner.post("/kb/api/v1/knowledge-bases", json={"name": "AID2E"}).json()["data"]["id"]
    owner.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file",
               files={"file": ("notes.md", io.BytesIO(DOC), "text/markdown")})
    drain(client)

    # directly: refused, with the way out; a request for what they may do themselves: refused too
    refused = owner.put(f"/kb/api/v1/knowledge-bases/{kb}", json={"visibility": "all"})
    assert refused.status_code == 403 and "sharing request" in refused.text
    direct = owner.post(f"/kb/api/v1/knowledge-bases/{kb}/share-requests", json={"visibility": "groups", "groups": ["org:eic"]})
    assert direct.status_code == 409

    # with a group they are not in: that group's admin approves; documents follow
    asked = owner.post(f"/kb/api/v1/knowledge-bases/{kb}/share-requests",
                       json={"visibility": "groups", "groups": ["org:atlas"], "note": "for the ATLAS study"})
    assert asked.status_code == 201, asked.text
    request = asked.json()["data"]
    assert request["status"] == "pending"
    assert [r["id"] for r in owner.get("/kb/api/v1/share-requests").json()["data"]] == [request["id"]]
    assert [r["id"] for r in head.get("/kb/api/v1/share-requests?scope=approve").json()["data"]] == [request["id"]]
    assert reader.get("/kb/api/v1/share-requests?scope=approve").json()["data"] == []
    assert reader.post(f"/kb/api/v1/share-requests/{request['id']}/approve").status_code == 404
    assert owner.post(f"/kb/api/v1/share-requests/{request['id']}/approve").status_code == 403   # not their own
    approved = head.post(f"/kb/api/v1/share-requests/{request['id']}/approve", json={"note": "ok"}).json()["data"]
    assert approved["status"] == "approved" and approved["applied_to_items"]["documents"] == 1
    [doc] = reader.get(f"/kb/api/v1/knowledge-bases/{kb}/knowledge").json()["data"]
    assert doc["visibility"] == "groups"
    assert head.post(f"/kb/api/v1/share-requests/{request['id']}/approve").status_code == 409

    # with everyone: only an admin decides; declining changes nothing; a newer request replaces the pending one
    first = owner.post(f"/kb/api/v1/knowledge-bases/{kb}/share-requests", json={"visibility": "all"}).json()["data"]
    second = owner.post(f"/kb/api/v1/knowledge-bases/{kb}/share-requests", json={"visibility": "all"}).json()["data"]
    mine = {r["id"]: r["status"] for r in owner.get("/kb/api/v1/share-requests").json()["data"]}
    assert mine[first["id"]] == "cancelled" and mine[second["id"]] == "pending"
    assert head.get("/kb/api/v1/share-requests?scope=approve").json()["data"] == []
    declined = admin.post(f"/kb/api/v1/share-requests/{second['id']}/decline", json={"note": "not public"}).json()["data"]
    assert declined["status"] == "declined" and declined["current_visibility"] == "groups"

    third = owner.post(f"/kb/api/v1/knowledge-bases/{kb}/share-requests", json={"visibility": "all"}).json()["data"]
    assert owner.delete(f"/kb/api/v1/share-requests/{third['id']}").json()["data"]["status"] == "cancelled"
    assert admin.get("/kb/api/v1/share-requests?scope=approve").json()["data"] == []
    assert admin.get("/kb/api/v1/me").json()["data"]["approves_shares"]
    assert head.get("/kb/api/v1/me").json()["data"]["approves_shares"]
    assert not reader.get("/kb/api/v1/me").json()["data"]["approves_shares"]
