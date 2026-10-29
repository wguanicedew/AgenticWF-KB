"""
The MCP server endpoint (docs/agent-integration.md, MCP): `POST /kb/mcp`, Streamable HTTP,
stateless JSON-RPC 2.0. It exposes search, read, grep, records and wiki as tools to MCP clients
other than the agent (which keeps its native tools).

The request has passed the mode's front like any API call (the standalone token, authsvc's
policy, the agent's socket), so every tool runs under the request's principal through its
KBThread, with the same data-level filtering as the REST API. Tools are read-only and need
`knowledge:read`.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from agenticwf.kb.api.deps import SCOPE_READ, KBThread, thread_for
from agenticwf.kb.runtime.errors import RuntimeError_
from agenticwf.kb.runtime.thread.thread import SearchOptions, feature_enabled

router = APIRouter()
READ = thread_for(SCOPE_READ)
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER = {"name": "agenticwf-kb", "version": "0.3"}

KB_IDS = {"type": "array", "items": {"type": "string"}, "description": "knowledge base ids or <owner>:<slug>"}

TOOLS: dict[str, dict] = {
    "kb_list_knowledge_bases": {"description": "The knowledge bases you can read.",
                                "inputSchema": {"type": "object", "properties": {}}},
    "kb_search": {"description": "Hybrid (semantic + keyword) search over knowledge bases; returns passages with ids.",
                  "inputSchema": {"type": "object", "properties": {
                      "query": {"type": "string"}, "knowledge_base_ids": KB_IDS,
                      "top_k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query"]}},
    "kb_read": {"description": "Read a document's text page by page.",
                "inputSchema": {"type": "object", "properties": {
                    "knowledge_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 200, "maximum": 20000}}, "required": ["knowledge_id"]}},
    "kb_grep": {"description": "Exact text (regex or literal) over passages: identifiers, error codes, versions.",
                "inputSchema": {"type": "object", "properties": {
                    "pattern": {"type": "string"}, "regex": {"type": "boolean"}, "case_sensitive": {"type": "boolean"},
                    "knowledge_base_ids": KB_IDS, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                    "required": ["pattern"]}},
}
RECORD_TOOLS = {
    "records_query": {"description": "Query structured records with the record query language "
                                     "(docs/structured-records.md).",
                      "inputSchema": {"type": "object", "properties": {
                          "record_type": {"type": "string"}, "filter": {"type": "object"}, "sort": {"type": "array"},
                          "fields": {"type": "array"}, "limit": {"type": "integer"}, "knowledge_base_ids": KB_IDS},
                          "required": ["record_type"]}},
    "records_aggregate": {"description": "Group and aggregate structured records (count, avg, p90, best, …).",
                          "inputSchema": {"type": "object", "properties": {
                              "record_type": {"type": "string"}, "filter": {"type": "object"},
                              "group_by": {"type": "array"}, "metrics": {"type": "array"}, "having": {"type": "object"},
                              "scope": {"type": "string"}, "knowledge_base_ids": KB_IDS},
                              "required": ["record_type", "metrics"]}},
}
WIKI_TOOLS = {
    "wiki_search": {"description": "Find pages of a knowledge base's wiki.",
                    "inputSchema": {"type": "object", "properties": {
                        "knowledge_base_id": {"type": "string"}, "query": {"type": "string"}},
                        "required": ["knowledge_base_id", "query"]}},
    "wiki_read_page": {"description": "Read one wiki page by slug.",
                       "inputSchema": {"type": "object", "properties": {
                           "knowledge_base_id": {"type": "string"}, "slug": {"type": "string"}},
                           "required": ["knowledge_base_id", "slug"]}},
}


def available_tools() -> dict[str, dict]:
    tools = dict(TOOLS)
    if feature_enabled("records"):
        tools.update(RECORD_TOOLS)
    if feature_enabled("wiki"):
        tools.update(WIKI_TOOLS)
    return tools


async def call_tool(thread: KBThread, name: str, args: dict) -> Any:
    if name == "kb_list_knowledge_bases":
        items, _ = await thread.list_kbs(1, 100)
        return [{"id": kb["id"], "ref": kb["ref"], "name": kb["name"], "type": kb["type"],
                 "description": kb["description"]} for kb in items]
    if name == "kb_search":
        hits = await thread.search(str(args.get("query") or ""), args.get("knowledge_base_ids") or None,
                                   SearchOptions(top_k=int(args.get("top_k") or 10)))
        return [{"id": h["id"], "knowledge_id": h["knowledge_id"], "title": h["knowledge_title"],
                 "knowledge_base_id": h["knowledge_base_id"], "chunk_type": h["chunk_type"], "score": round(h["score"], 4),
                 "content": h["content"], "citation": h.get("citation_text"),
                 **({"match": h["match"]} if h.get("match") else {})} for h in hits]
    if name == "kb_read":
        return await thread.knowledge_content(str(args["knowledge_id"]), int(args.get("offset") or 0),
                                              int(args.get("limit") or 8000))
    if name == "kb_grep":
        return await thread.grep(str(args.get("pattern") or ""), args.get("knowledge_base_ids") or None,
                                 regex=args.get("regex", True) is not False,
                                 case_sensitive=bool(args.get("case_sensitive")), limit=int(args.get("limit") or 20))
    if name == "records_query":
        return await thread.query_records(args)
    if name == "records_aggregate":
        return await thread.aggregate_records(args)
    if name == "wiki_search":
        return await thread.wiki_search(str(args["knowledge_base_id"]), str(args.get("query") or ""))
    if name == "wiki_read_page":
        return await thread.wiki_page(str(args["knowledge_base_id"]), str(args["slug"]))
    raise KeyError(name)


def _result(id_, result) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


async def handle(thread: KBThread, message: Any) -> dict | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(message.get("id") if isinstance(message, dict) else None, -32600, "invalid request")
    method, id_, params = message["method"], message.get("id"), message.get("params") or {}
    if id_ is None:                          # a notification (e.g. notifications/initialized): no answer
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(id_, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": SERVER,
                             "instructions": "Read-only access to the AgenticWF knowledge bases you can read."})
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": [{"name": n, **spec} for n, spec in available_tools().items()]})
    if method == "tools/call":
        name = params.get("name")
        if name not in available_tools():
            return _error(id_, -32602, f"unknown tool {name!r}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _error(id_, -32602, "arguments must be an object")
        try:
            data = await call_tool(thread, name, args)
        except RuntimeError_ as exc:          # the KB's own errors are tool errors the client can show
            return _result(id_, {"content": [{"type": "text", "text": f"Error: {exc.message}"}], "isError": True})
        except (KeyError, TypeError, ValueError) as exc:
            return _result(id_, {"content": [{"type": "text", "text": f"Error: bad arguments ({exc})"}],
                                 "isError": True})
        text = json.dumps(data, default=str, ensure_ascii=False)
        structured = data if isinstance(data, dict) else {"items": data}
        return _result(id_, {"content": [{"type": "text", "text": text}], "structuredContent": structured,
                             "isError": False})
    return _error(id_, -32601, f"method {method!r} not found")


@router.post("/mcp")
async def mcp(request: Request, thread: KBThread = Depends(READ)):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(_error(None, -32700, "parse error"), status_code=400)
    if isinstance(body, list):                 # a batch
        answers = [a for a in [await handle(thread, m) for m in body] if a is not None]
        return JSONResponse(answers) if answers else Response(status_code=202)
    answer = await handle(thread, body)
    return JSONResponse(answer) if answer is not None else Response(status_code=202)


@router.get("/mcp")
async def mcp_stream():
    """No server-initiated stream: the endpoint is stateless."""
    return Response(status_code=405, headers={"Allow": "POST"})
