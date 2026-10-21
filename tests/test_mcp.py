"""The MCP endpoint (/kb/mcp): JSON-RPC over Streamable HTTP, tools under the caller's principal."""

import io
import json

import pytest

from conftest import Caller, drain, make_client, user

RW = ("app:access", "knowledge:read", "knowledge:write")


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


def rpc(caller, method, params=None, id_=1):
    response = caller.post("/kb/mcp", json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})
    assert response.status_code == 200, response.text
    return response.json()


def test_mcp_tools(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
    alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file",
               files={"file": ("e.md", io.BytesIO(b"# Errors\n\nError E-2201: the copy tool failed.\n"), "text/markdown")})
    drain(client)
    init = rpc(alice, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "t", "version": "1"}})["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "agenticwf-kb"
    note = alice.post("/kb/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert note.status_code == 202
    names = {t["name"] for t in rpc(alice, "tools/list")["result"]["tools"]}
    assert {"kb_search", "kb_read", "kb_grep", "records_query", "wiki_search"} <= names
    found = rpc(alice, "tools/call", {"name": "kb_search", "arguments": {"query": "copy tool failed"}})["result"]
    hits = json.loads(found["content"][0]["text"])
    assert not found["isError"] and hits[0]["title"] == "e"
    read = rpc(alice, "tools/call", {"name": "kb_read", "arguments": {"knowledge_id": hits[0]["knowledge_id"]}})
    assert "E-2201" in read["result"]["structuredContent"]["content"]
    grep = rpc(alice, "tools/call", {"name": "kb_grep", "arguments": {"pattern": r"E-\d+"}})["result"]
    assert grep["structuredContent"]["matches"]
    bad = rpc(alice, "tools/call", {"name": "kb_read", "arguments": {"knowledge_id": "nope"}})["result"]
    assert bad["isError"] and "not found" in bad["content"][0]["text"]
    assert rpc(alice, "nope")["error"]["code"] == -32601
    # another user sees none of alice's knowledge
    bob = Caller(client, user("bob", RW, org="atlas"))
    other = rpc(bob, "tools/call", {"name": "kb_search", "arguments": {"query": "copy tool failed"}})["result"]
    assert json.loads(other["content"][0]["text"]) == []
    assert client.post("/kb/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
