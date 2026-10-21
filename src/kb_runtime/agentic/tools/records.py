"""
Records read tools (docs/agentic-search.md, Records read tools; docs/structured-records.md):
describe_record_types, query_records, aggregate_records, find_similar_records, and the read-only
plugin operations (e.g. suggest_initial_points). Offered when a records KB is in scope.

Records come back with `rN` handles; numbers in an answer are copied from this output.
"""

from __future__ import annotations

import json
import re

from kb_runtime.agentic.tools import Tool, ToolContext, ToolError, attr, int_arg, list_arg
from kb_runtime.errors import Invalid, NotFound

MAX_ROWS = 50
MAX_AGG_ROWS = 200


def record_payload(record: dict, score=None) -> dict:
    facets = record.get("facets") or {}
    measures = record.get("measures") or {}
    lines = [f"{record.get('name')} ({record.get('record_type')})"]
    lines += [f"{k}: {v}" for k, v in facets.items()] + [f"{k}: {v}" for k, v in measures.items()]
    return {"kind": "record", "record_id": record["id"], "record_type": record.get("record_type"),
            "name": record.get("name"), "knowledge_base_id": record.get("knowledge_base_id"),
            "content": "\n".join(lines), "score": score,
            "citation_text": f"{record.get('name')} ({record.get('record_type')})"}


def record_handle(tc: ToolContext, record: dict, score=None) -> str:
    return tc.handles.issue("r", record["id"], record_payload(record, score))


def _records_scope(tc: ToolContext, handles: list[str] | None) -> list[str]:
    # an argument that is no handle at all (small models copy the "bN" placeholder) means "all in
    # scope"; that never widens past the request's scope
    handles = [h for h in handles or [] if tc.handles.lookup(h) is not None]
    ids = tc.kb_ids(handles)
    records = [i for i in ids if tc.scope.kbs[i].type == "records"]
    if not records:
        raise ToolError("no records knowledge base is in scope")
    return records


def _structured(value):
    """Small models send objects as JSON strings, sometimes in Python syntax: parse them."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except ValueError:
        import ast

        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ToolError(f"could not parse {text[:80]!r} as JSON")


async def _resolve_type(tc: ToolContext, kb_ids: list[str], name) -> tuple[str, "object"]:
    """The record type to use: the one named, or, when exactly one type is in scope and the name
    is unknown, that one (small models invent names). The error lists the types in scope."""
    from kb_core.orm.base.session import async_session_scope

    records = tc.services.records
    async with async_session_scope(commit=False) as session:
        in_scope = sorted(await records._kb_types(session, kb_ids))
        try:
            if name:
                td = await records.typedef(session, tc.auth, str(name), kb_ids)
                return td.qualified, td
        except (NotFound, Invalid):
            pass
        if len(in_scope) == 1:
            td = await records.typedef(session, tc.auth, in_scope[0], kb_ids)
            return td.qualified, td
    raise ToolError(f"unknown record type {name!r}; the types in scope are: {', '.join(in_scope) or 'none'}")


def _fix_fields(td, value, key: str | None = None):
    """Map a bare field name to the one declared path ending with it (`accuracy` →
    `metrics.accuracy`); ambiguous or unknown names are left for the query's own error."""
    if isinstance(value, list):
        return [_fix_fields(td, v, key) for v in value]
    if isinstance(value, dict):
        return {k: _fix_fields(td, v, k) for k, v in value.items()}
    if key in ("field", None) and isinstance(value, str) and not td.has_field(value) \
            and not td.has_field(value, "items") and re.fullmatch(r"[\w.]+", value):
        matches = {i.path for (_, _), i in td.fields.items() if i.path.split(".")[-1] == value.split(".")[-1]}
        if len(matches) == 1:
            return matches.pop()
    return value


async def _prepare(tc: ToolContext, args: dict, kb_ids: list[str], keys: tuple[str, ...]) -> dict:
    record_type, td = await _resolve_type(tc, kb_ids, args.get("record_type"))
    body = {"knowledge_base_ids": kb_ids, "record_type": record_type}
    for key in keys:
        if args.get(key) not in (None, "", [], {}):
            value = _structured(args[key])
            if key in ("fields", "group_by") and isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            body[key] = _fix_fields(td, value, None if key in ("fields", "group_by") else key)
    return body


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)[:200]
    return str(value)


async def describe_record_types(tc: ToolContext, args: dict) -> str:
    records = tc.services.records
    kb_ids = _records_scope(tc, list_arg(args, "knowledge_base_ids"))
    wanted = list_arg(args, "record_types")
    from kb_core.orm.base.session import async_session_scope

    async with async_session_scope(commit=False) as session:
        names = sorted(await records._kb_types(session, kb_ids))
        if wanted:
            names = [n for n in names if n in wanted or n.rpartition(":")[2] in wanted]
        defs = []
        for name in names:
            try:
                defs.append(await records.typedef(session, tc.auth, name, kb_ids))
            except (NotFound, Invalid):
                continue
    if not defs:
        return "No record types in scope."
    ops = {}
    for qualified, operation in records.read_only_operations():
        ops.setdefault(qualified, []).append(operation.tool_name)
    out = []
    for td in defs:
        stats = {(s["field"], s["scope"]): s for s in await records.stats(tc.auth, td.qualified)}
        r = td.roles
        lines = [f'<record_type name="{attr(td.qualified)}" version="{td.version}" title="{attr(td.title)}">']
        if td.description:
            lines.append(td.description)
        lines.append(f"id: {r['id']}; name: {r['name'] or r['id']}; time: {r['time'] or '-'}")
        if r["text"]:
            lines.append("text (use op=text): " + ", ".join(r["text"]))
        for path in r["facets"]:
            s = stats.get((path, "record"))
            top = ", ".join(_fmt(t["value"]) for t in (s or {}).get("top_values", [])[:10])
            lines.append(f"facet {path}: {top or '(no values yet)'}")
        for path, spec in r["measures"].items():
            s = stats.get((path, "record")) or {}
            lines.append(f"measure {path}{' [' + spec['unit'] + ']' if spec.get('unit') else ''}"
                         f"{' ' + spec['direction'] if spec.get('direction') else ''}: "
                         f"range {_fmt(s.get('min'))} .. {_fmt(s.get('max'))}, n={s.get('count', 0)}")
        if r["features"]:
            lines.append("features: " + ", ".join(r["features"]))
        if r["items"]:
            items = r["items"]
            lines.append(f"items {items['path']} (filter/aggregate with scope=items): measures "
                         + ", ".join(f"{p}{' ' + v['direction'] if v.get('direction') else ''}"
                                     for p, v in items["measures"].items())
                         + (f"; features {', '.join(items['features'])}" if items["features"] else ""))
        other = sorted(i.path for (scope, _), i in td.fields.items() if scope == "record")
        lines.append("all fields: " + ", ".join(other)[:1500])
        if ops.get(td.qualified):
            lines.append("operations: " + ", ".join(ops[td.qualified]))
        lines.append("</record_type>")
        out.append("\n".join(lines))
    return tc.clip("\n".join(out))


async def query_records(tc: ToolContext, args: dict) -> str:
    kb_ids = _records_scope(tc, list_arg(args, "knowledge_base_ids"))
    body = await _prepare(tc, args, kb_ids, ("filter", "sort", "fields"))
    body["limit"] = int_arg(args, "limit", 20, 1, MAX_ROWS)
    try:
        result = await tc.services.records.query(tc.auth, body)
    except (Invalid, NotFound) as exc:
        raise ToolError(str(exc))
    rows = result["records"]
    out = [f'<records type="{attr(result["record_type"])}" total="{result["total"]}" returned="{len(rows)}">']
    for row in rows:
        handle = record_handle(tc, {**row, "facets": {}, "measures": row["fields"]})
        fields = "; ".join(f"{k}={_fmt(v)}" for k, v in row["fields"].items() if v is not None)
        out.append(f'<record id="{handle}" name="{attr(row["name"])}" time="{(row["event_time"] or "")[:19]}">'
                   f'{attr(fields)}</record>')
    out.append("</records>")
    if result["total"] > len(rows):
        out.append("[more records match: narrow the filter or sort]")
    return tc.clip("\n".join(out))


async def aggregate_records(tc: ToolContext, args: dict) -> str:
    kb_ids = _records_scope(tc, list_arg(args, "knowledge_base_ids"))
    body = await _prepare(tc, args, kb_ids, ("filter", "group_by", "metrics", "having", "sort"))
    for key in ("scope", "timezone"):
        if args.get(key):
            body[key] = args[key]
    body["limit"] = int_arg(args, "limit", 50, 1, MAX_AGG_ROWS)
    try:
        result = await tc.services.records.aggregate(tc.auth, body)
    except (Invalid, NotFound) as exc:
        raise ToolError(str(exc))
    columns = result["columns"]
    if not result["rows"]:
        return f"No rows for {result['record_type']} with this filter."
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in result["rows"]:
        lines.append("| " + " | ".join(_fmt(row[c]) if row[c] is not None else "" for c in columns) + " |")
    lines.append("Copy numbers exactly from this table. To cite records, find them with query_records.")
    return tc.clip("\n".join(lines))


async def find_similar_records(tc: ToolContext, args: dict) -> str:
    kb_ids = _records_scope(tc, list_arg(args, "knowledge_base_ids"))
    body = {"knowledge_base_ids": kb_ids, "filter": args.get("filter"), "limit": int_arg(args, "limit", 10, 1, MAX_ROWS)}
    if args.get("record_id"):
        handle = tc.handles.get(str(args["record_id"]), ("r",))
        body["record_id"] = handle.key
    else:
        if not args.get("record_type") or not isinstance(args.get("probe"), dict):
            raise ToolError("give record_id (rN) or record_type and probe")
        body.update(record_type=args["record_type"], probe=args["probe"])
    try:
        result = await tc.services.records.similar(tc.auth, body)
    except (Invalid, NotFound) as exc:
        raise ToolError(str(exc))
    out = [f'<similar type="{attr(result["record_type"])}">']
    for row in result["records"]:
        handle = record_handle(tc, {**row, "record_type": result["record_type"]}, row["score"])
        parts = ", ".join(f"{k}={v:.2f}" for k, v in row["breakdown"].items() if v is not None)
        facts = "; ".join(f"{k}={_fmt(v)}" for k, v in {**row["facets"], **row["measures"]}.items())
        out.append(f'<record id="{handle}" name="{attr(row["name"])}" similarity="{row["score"]:.3f}" '
                   f'parts="{parts}">{attr(facts)}</record>')
    out.append("</similar>")
    return tc.clip("\n".join(out))


def operation_tool(qualified: str, operation) -> Tool:
    """A read-only plugin operation as a tool; KB arguments are bN handles inside the scope."""
    async def run(tc: ToolContext, args: dict) -> str:
        from kb_runtime.records.plugins import OpEnv

        body = dict(args)
        body["knowledge_base_ids"] = _records_scope(tc, list_arg(args, "knowledge_base_ids"))
        try:
            result = await operation.run(OpEnv(tc.auth, tc.services, tc.services.records), body)
        except (Invalid, NotFound) as exc:
            raise ToolError(str(exc))
        for job in result.get("source_jobs") or []:
            job["handle"] = record_handle(tc, {"id": job["record_id"], "record_type": qualified,
                                               "name": job.get("record_name"), "knowledge_base_id":
                                               job["knowledge_base_id"]}, job.get("similarity"))
        for point in result.get("points") or []:
            source = point.get("source") or {}
            if source.get("record_id"):
                source["handle"] = record_handle(tc, {"id": source["record_id"], "record_type": qualified,
                                                      "name": source.get("record_name"),
                                                      "knowledge_base_id": source.get("knowledge_base_id")})
                for key in ("record_id", "knowledge_base_id"):
                    source.pop(key, None)
        for job in result.get("source_jobs") or []:
            job.pop("record_id", None)
            job.pop("knowledge_base_id", None)
        return tc.clip(json.dumps(result, default=str, indent=1) + "\nCopy numbers exactly from this output; "
                       "cite source jobs by their handle.")

    params = json.loads(json.dumps(operation.parameters))
    params.setdefault("properties", {})["knowledge_base_ids"] = {"type": "array", "items": {"type": "string"},
                                                                 "description": "bN handles"}
    return Tool(operation.tool_name, operation.description, params, run, group="records")


FILTER = {"type": "object", "description": "{and|or|not: [...]} or {field, op, value[, scope: items]}; ops eq ne lt "
                                           "lte gt gte between in nin contains exists text near"}
KBS = {"type": "array", "items": {"type": "string"}, "description": "bN handles of records KBs; default: all in scope"}

TOOLS = [
    Tool("describe_record_types", "The record types in scope: fields, roles, units, value ranges, top facet values "
         "and operations. Call it before writing a query.",
         {"type": "object", "properties": {"record_types": {"type": "array", "items": {"type": "string"}},
                                           "knowledge_base_ids": KBS}}, describe_record_types, group="records"),
    Tool("query_records", "Filter, sort and list records of one type. Returns rows with rN handles.",
         {"type": "object", "properties": {
             "record_type": {"type": "string"}, "filter": FILTER,
             "sort": {"type": "array", "items": {"type": "object", "properties": {
                 "field": {"type": "string"}, "dir": {"type": "string", "enum": ["asc", "desc"]}}}},
             "fields": {"type": "array", "items": {"type": "string"}},
             "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS}, "knowledge_base_ids": KBS},
          "required": ["record_type"]}, query_records, group="records"),
    Tool("aggregate_records", "Group and aggregate records of one type (count, sum, avg, min, max, p50, p90, p99, "
         "stddev, count_distinct, best). Returns a table; copy its numbers exactly.",
         {"type": "object", "properties": {
             "record_type": {"type": "string"}, "filter": FILTER,
             "group_by": {"type": "array", "items": {}, "description": "fields, or {field, bucket: hour|day|week|month|quarter|year}"},
             "metrics": {"type": "array", "items": {"type": "object", "properties": {
                 "fn": {"type": "string"}, "field": {"type": "string"}}}},
             "having": {"type": "object"}, "sort": {"type": "array", "items": {"type": "object"}},
             "scope": {"type": "string", "enum": ["record", "items"]}, "timezone": {"type": "string"},
             "limit": {"type": "integer", "minimum": 1, "maximum": MAX_AGG_ROWS}, "knowledge_base_ids": KBS},
          "required": ["record_type", "metrics"]}, aggregate_records, group="records"),
    Tool("find_similar_records", "Records most similar to one record (rN) or to a partial record (probe), with a "
         "text / facets / features breakdown.",
         {"type": "object", "properties": {
             "record_id": {"type": "string", "description": "rN"}, "record_type": {"type": "string"},
             "probe": {"type": "object"}, "filter": FILTER, "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS},
             "knowledge_base_ids": KBS}}, find_similar_records, group="records"),
]
