"""
The record query and aggregate languages (docs/structured-records.md, Query language,
Aggregates), compiled to parametrised SQL.

Every field is checked against the type (aliases resolved) before it reaches SQL, and is then
addressed only as a bound JSON path (`data #>> :p`); values are bound parameters. Nothing from
the caller is ever spliced into the SQL text except names from fixed whitelists (operators,
functions, time buckets, sort directions).

`evaluate()` is the Python reference for filters: the property test checks that the compiled
SQL returns the same rows.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from typing import Any

from kb_runtime.errors import Invalid
from kb_runtime.records.schema import FieldInfo, TypeDef, get_path, has_path, parse_time

OPS = ("eq", "ne", "lt", "lte", "gt", "gte", "between", "in", "nin", "contains", "exists", "text", "near")
FNS = ("count", "sum", "avg", "min", "max", "p50", "p90", "p99", "stddev", "count_distinct", "best")
BUCKETS = ("hour", "day", "week", "month", "quarter", "year")
MAX_DEPTH = 8
MAX_LEAVES = 64


@dataclass
class Compiler:
    """Builds one SQL statement's WHERE / expressions, collecting bound parameters."""

    typedef: TypeDef
    params: dict = field(default_factory=dict)
    item_alias: str | None = None          # "i" when items rows are joined (aggregate scope items)
    _n: int = 0
    _leaves: int = 0

    def bind(self, value) -> str:
        self._n += 1
        name = f"p{self._n}"
        self.params[name] = value
        return f":{name}"

    # -- fields ---------------------------------------------------------------------------------------
    def resolve(self, path: str, scope: str = "record") -> FieldInfo:
        return self.typedef.field(path, scope)

    def expr(self, info: FieldInfo, alias: str | None = None) -> str:
        """The SQL expression of a field, typed by the schema."""
        if info.scope == "record" and info.path == self.typedef.roles["time"]:
            return "r.event_time"
        table = alias or ("r" if info.scope == "record" else (self.item_alias or "i"))
        path = self.bind(info.path.split("."))
        if info.type in ("array", "object"):
            return f"({table}.data #> CAST({path} AS text[]))"
        raw = f"({table}.data #>> CAST({path} AS text[]))"
        if info.numeric:
            return f"(CASE WHEN jsonb_typeof({table}.data #> CAST({path} AS text[])) = 'number' THEN {raw}::float8 END)"
        if info.type == "boolean":
            return f"(CASE WHEN jsonb_typeof({table}.data #> CAST({path} AS text[])) = 'boolean' THEN {raw}::boolean END)"
        if info.temporal:
            return f"({raw})::timestamptz"
        return raw

    def value(self, info: FieldInfo, value):
        if value is None:
            raise Invalid(f"{info.path}: a value is required")
        if info.numeric:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise Invalid(f"{info.path}: {value!r} is not a number")
            return float(value)
        if info.type == "boolean":
            if not isinstance(value, bool):
                raise Invalid(f"{info.path}: {value!r} is not true or false")
            return value
        if info.temporal or (info.scope == "record" and info.path == self.typedef.roles["time"]):
            return parse_time(value)
        if info.type in ("array", "object"):
            return value
        return str(value)

    # -- filters -------------------------------------------------------------------------------------
    def where(self, node: dict | None, depth: int = 0, in_items: bool = False) -> str:
        if not node:
            return "TRUE"
        if depth > MAX_DEPTH:
            raise Invalid("the filter is nested too deeply")
        if not isinstance(node, dict):
            raise Invalid("a filter is an object")
        scope = node.get("scope", "record")
        if scope not in ("record", "items"):
            raise Invalid("scope must be record or items")
        if scope == "items" and not in_items:
            if not self.typedef.roles["items"]:
                raise Invalid(f"record type {self.typedef.name} has no items")
            if self.item_alias:                  # items already joined: filter the joined row
                return self.where({k: v for k, v in node.items() if k != "scope"}, depth, True)
            inner = self.where({k: v for k, v in node.items() if k != "scope"}, depth + 1, True)
            return f"EXISTS (SELECT 1 FROM kb_record_items i WHERE i.record_id = r.id AND {inner})"
        for key in ("and", "or"):
            if key in node:
                parts = node[key]
                if not isinstance(parts, list) or not parts:
                    raise Invalid(f"{key} takes a non-empty list")
                joined = f" {key.upper()} ".join(self.where(p, depth + 1, in_items) for p in parts)
                return f"({joined})"
        if "not" in node:
            return f"(NOT COALESCE({self.where(node['not'], depth + 1, in_items)}, FALSE))"
        return self.leaf(node, in_items)

    def leaf(self, node: dict, in_items: bool) -> str:
        self._leaves += 1
        if self._leaves > MAX_LEAVES:
            raise Invalid(f"at most {MAX_LEAVES} conditions per filter")
        op = node.get("op")
        if op not in OPS:
            raise Invalid(f"op must be one of {', '.join(OPS)}")
        scope = "items" if in_items else "record"
        info = self.resolve(node.get("field"), scope)
        alias = (self.item_alias or "i") if in_items else "r"
        expr = self.expr(info, alias)
        if info.type == "string" and not info.temporal and op in ("lt", "lte", "gt", "gte", "between"):
            expr = f'({expr} COLLATE "C")'           # code-point order, as the reference evaluator
        value = node.get("value")
        if op == "exists":
            present = value is None or bool(value)
            path = self.bind(info.path.split("."))
            check = f"({alias}.data #> CAST({path} AS text[])) IS NOT NULL" if not (
                scope == "record" and info.path == self.typedef.roles["time"]) else "r.event_time IS NOT NULL"
            return check if present else f"NOT ({check})"
        if op == "text":
            if scope != "record" or info.path not in self.typedef.roles["text"]:
                raise Invalid("text matches only text-role fields")
            q = self.bind(str(value or ""))
            return (f"r.id IN (SELECT source_id FROM kb_embeddings WHERE id @@@ paradedb.match('content', {q}) "
                    f"AND source_type = 5 AND knowledge_base_id = ANY(:kbs))")
        if info.type in ("array", "object") and op not in ("contains",):
            raise Invalid(f"{info.path}: only contains and exists apply to {info.type} fields")
        if op in ("eq", "ne", "lt", "lte", "gt", "gte"):
            if op in ("lt", "lte", "gt", "gte") and info.type == "boolean":
                raise Invalid(f"{info.path}: comparisons do not apply to booleans")
            sql_op = {"eq": "=", "ne": "IS DISTINCT FROM", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[op]
            return f"({expr} {sql_op} {self.bind(self.value(info, value))})"
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise Invalid("between takes [low, high]")
            low, high = (self.bind(self.value(info, v)) for v in value)
            return f"({expr} BETWEEN {low} AND {high})"
        if op in ("in", "nin"):
            if not isinstance(value, list) or not value:
                raise Invalid(f"{op} takes a non-empty list")
            values = [self.value(info, v) for v in value]
            cast = "float8[]" if info.numeric else "boolean[]" if info.type == "boolean" else \
                "timestamptz[]" if info.temporal else "text[]"
            arr = self.bind(values)
            if op == "in":
                return f"({expr} = ANY(CAST({arr} AS {cast})))"
            return f"({expr} IS NULL OR NOT ({expr} = ANY(CAST({arr} AS {cast}))))"
        if op == "contains":
            if info.type == "array":
                return f"({expr} @> jsonb_build_array(CAST({self.bind(json.dumps(value))} AS jsonb)))"
            if info.type != "string" or info.temporal:
                raise Invalid(f"{info.path}: contains applies to strings and arrays")
            return f"(strpos(lower({expr}), lower({self.bind(str(value))})) > 0)"
        if op == "near":
            if not info.numeric:
                raise Invalid(f"{info.path}: near applies to numbers")
            if not isinstance(value, list) or len(value) != 2:
                raise Invalid("near takes [value, tolerance]")
            centre, tolerance = (self.bind(self.value(info, v)) for v in value)
            return f"(abs({expr} - {centre}) <= {tolerance})"
        raise Invalid(f"unsupported op {op}")


# --- query ------------------------------------------------------------------------------------------
BASE_WHERE = ("r.knowledge_base_id = ANY(:kbs) AND r.record_type = ANY(:types) AND r.deleted_at IS NULL "
              "AND (r.valid_from IS NULL OR r.valid_from <= now()) AND (r.valid_until IS NULL OR r.valid_until > now()) "
              "AND (r.expires_at IS NULL OR r.expires_at > now()) AND (r.deprecated_at IS NULL OR :with_deprecated)")


def compile_query(typedef: TypeDef, body: dict, max_rows: int) -> tuple[str, str, dict, list[str]]:
    """(select SQL, count SQL, params, projected fields)."""
    c = Compiler(typedef)
    where = c.where(body.get("filter"))
    order = []
    for item in body.get("sort") or []:
        info = c.resolve(item.get("field") if isinstance(item, dict) else item)
        direction = (item.get("dir") or "asc").lower() if isinstance(item, dict) else "asc"
        if direction not in ("asc", "desc"):
            raise Invalid("sort dir is asc or desc")
        order.append(f"{c.expr(info)} {direction.upper()} NULLS LAST")
    order.append("r.event_time DESC NULLS LAST")
    order.append("r.id")
    limit = int(body.get("limit") or 20)
    if not 1 <= limit <= max_rows:
        raise Invalid(f"limit must be between 1 and {max_rows}")
    offset = max(0, int(body.get("offset") or 0))
    fields = body.get("fields") or None
    if fields is not None:
        fields = [c.resolve(f).path for f in fields]
    sql = (f"SELECT r.* FROM kb_records r WHERE {BASE_WHERE} AND {where} ORDER BY {', '.join(order)} "
           f"LIMIT {limit} OFFSET {offset}")
    count = f"SELECT count(*) FROM kb_records r WHERE {BASE_WHERE} AND {where}"
    return sql, count, c.params, fields


# --- aggregate ----------------------------------------------------------------------------------------
def metric_name(metric: dict) -> str:
    fn = metric.get("fn")
    return "count" if fn == "count" and not metric.get("field") else f"{fn}({metric.get('field')})"


def compile_aggregate(typedef: TypeDef, body: dict, max_rows: int) -> tuple[str, dict, list[str], list[str]]:
    """(SQL, params, group column names, metric column names)."""
    scope = body.get("scope") or "record"
    if scope not in ("record", "items"):
        raise Invalid("scope must be record or items")
    if scope == "items" and not typedef.roles["items"]:
        raise Invalid(f"record type {typedef.name} has no items")
    c = Compiler(typedef, item_alias="i" if scope == "items" else None)
    timezone = body.get("timezone") or "UTC"
    _check_timezone(timezone)

    def field_info(path: str) -> FieldInfo:
        if scope == "items" and typedef.has_field(path, "items"):
            return c.resolve(path, "items")
        return c.resolve(path, "record")

    groups, group_names = [], []
    for g in body.get("group_by") or []:
        if isinstance(g, str):
            info = field_info(g)
            if info.type in ("array", "object"):
                raise Invalid(f"{info.path}: cannot group by an {info.type}")
            groups.append(c.expr(info))
            group_names.append(info.path)
        elif isinstance(g, dict):
            info = field_info(g.get("field"))
            bucket = g.get("bucket")
            if bucket not in BUCKETS:
                raise Invalid(f"bucket must be one of {', '.join(BUCKETS)}")
            expr = c.expr(info)
            if not (info.temporal or expr == "r.event_time"):
                raise Invalid(f"{info.path}: time buckets need a time field")
            groups.append(f"date_trunc('{bucket}', {expr} AT TIME ZONE {c.bind(timezone)})")
            group_names.append(f"{info.path}:{bucket}")
        else:
            raise Invalid("group_by takes field names or {field, bucket}")
    if len(groups) > 4:
        raise Invalid("at most 4 group_by entries")

    metrics = body.get("metrics") or [{"fn": "count"}]
    selects, names, by_name = [], [], {}
    for metric in metrics:
        fn = metric.get("fn") if isinstance(metric, dict) else None
        if fn not in FNS:
            raise Invalid(f"fn must be one of {', '.join(FNS)}")
        name = metric_name(metric)
        if fn == "count" and not metric.get("field"):
            sql = "count(*)"
        else:
            info = field_info(metric.get("field"))
            expr = c.expr(info)
            if fn == "count":
                sql = f"count({expr})"
            elif fn == "count_distinct":
                sql = f"count(DISTINCT {expr})"
            else:
                if not info.numeric:
                    raise Invalid(f"{fn} needs a numeric field, {info.path} is {info.type}")
                if fn in ("p50", "p90", "p99"):
                    sql = f"percentile_cont({int(fn[1:]) / 100}) WITHIN GROUP (ORDER BY {expr})"
                elif fn == "best":
                    measure = typedef.measure(info.path, info.scope) or {}
                    if measure.get("direction") not in ("maximize", "minimize"):
                        raise Invalid(f"best needs a measure with a direction; {info.path} has none")
                    sql = f"{'max' if measure['direction'] == 'maximize' else 'min'}({expr})"
                elif fn == "stddev":
                    sql = f"stddev_samp({expr})"
                else:
                    sql = f"{fn}({expr})"
        selects.append(sql)
        names.append(name)
        by_name[name] = sql

    having = _having(c, body.get("having"), by_name) if body.get("having") else "TRUE"
    order = []
    columns = {n: i + 1 for i, n in enumerate(group_names + names)}
    for item in body.get("sort") or []:
        key = item.get("field") if isinstance(item, dict) else item
        direction = (item.get("dir") or "asc").lower() if isinstance(item, dict) else "asc"
        if key not in columns or direction not in ("asc", "desc"):
            raise Invalid(f"sort by a group or metric column: {', '.join(columns)}")
        order.append(f"{columns[key]} {direction.upper()} NULLS LAST")
    order += [str(i + 1) for i in range(len(groups))]
    limit = int(body.get("limit") or 50)
    if not 1 <= limit <= max_rows:
        raise Invalid(f"limit must be between 1 and {max_rows}")
    joined = "JOIN kb_record_items i ON i.record_id = r.id" if scope == "items" else ""
    where = c.where(body.get("filter"))
    select = ", ".join([*(f"{g}" for g in groups), *selects])
    group_sql = f"GROUP BY {', '.join(str(i + 1) for i in range(len(groups)))}" if groups else ""
    sql = (f"SELECT {select} FROM kb_records r {joined} WHERE {BASE_WHERE} AND {where} {group_sql} "
           f"HAVING {having} {('ORDER BY ' + ', '.join(order)) if order else ''} LIMIT {limit}")
    return sql, c.params, group_names, names


def _having(c: Compiler, node, by_name: dict, depth: int = 0) -> str:
    if depth > MAX_DEPTH or not isinstance(node, dict):
        raise Invalid("having is a filter over metric names")
    for key in ("and", "or"):
        if key in node:
            return "(" + f" {key.upper()} ".join(_having(c, p, by_name, depth + 1) for p in node[key]) + ")"
    if "not" in node:
        return f"(NOT {_having(c, node['not'], by_name, depth + 1)})"
    name, op = node.get("field"), node.get("op")
    if name not in by_name:
        raise Invalid(f"having uses metric names: {', '.join(by_name)}")
    ops = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
    if op == "between":
        low, high = node["value"]
        return f"({by_name[name]} BETWEEN {c.bind(float(low))} AND {c.bind(float(high))})"
    if op not in ops:
        raise Invalid("having ops: eq ne lt lte gt gte between")
    return f"({by_name[name]} {ops[op]} {c.bind(float(node['value']))})"


def _check_timezone(name: str) -> None:
    import zoneinfo

    try:
        zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise Invalid(f"unknown timezone {name!r}")


# --- the Python reference evaluator --------------------------------------------------------------------
_ABSENT = object()


def _field_value(typedef: TypeDef, info: FieldInfo, record: dict, event_time):
    if info.scope == "record" and info.path == typedef.roles["time"]:
        return event_time
    value = get_path(record, info.path, _ABSENT)
    if value is _ABSENT or value is None:
        return None
    if info.numeric:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if info.type == "boolean":
        return value if isinstance(value, bool) else None
    if info.temporal:
        return parse_time(value)
    if info.type in ("array", "object"):
        return value
    return value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)


def evaluate(typedef: TypeDef, node: dict | None, record: dict, event_time=None, items: list[dict] | None = None,
             in_items: bool = False) -> bool:
    """Does `record` (canonical data) match the filter? Mirrors the compiled SQL (SQL NULL → no
    match), except `text`, which needs the BM25 index."""
    result = _eval(typedef, node, record, event_time, items or [], in_items)
    return bool(result)


def _eval(typedef, node, record, event_time, items, in_items):
    if not node:
        return True
    scope = node.get("scope", "record")
    if scope == "items" and not in_items:
        inner = {k: v for k, v in node.items() if k != "scope"}
        return any(_eval(typedef, inner, item, None, [], True) for item in items)
    if "and" in node:
        values = [_eval(typedef, p, record, event_time, items, in_items) for p in node["and"]]
        if any(v is False for v in values):
            return False
        return None if any(v is None for v in values) else True
    if "or" in node:
        values = [_eval(typedef, p, record, event_time, items, in_items) for p in node["or"]]
        if any(v is True for v in values):
            return True
        return None if any(v is None for v in values) else False
    if "not" in node:
        return not bool(_eval(typedef, node["not"], record, event_time, items, in_items))
    info = typedef.field(node["field"], "items" if in_items else "record")
    op, value = node["op"], node.get("value")
    actual = _field_value(typedef, info, record, None if in_items else event_time)
    if op == "exists":
        present = (event_time is not None) if (not in_items and info.path == typedef.roles["time"]) else \
            has_path(record, info.path) and get_path(record, info.path) is not None
        return present if (value is None or value) else not present
    convert = Compiler(typedef).value
    if op in ("eq", "ne", "lt", "lte", "gt", "gte"):
        target = convert(info, value)
        if op == "ne":
            return actual != target
        if actual is None:
            return None
        return {"eq": actual == target, "lt": actual < target, "lte": actual <= target, "gt": actual > target,
                "gte": actual >= target}[op]
    if op == "between":
        low, high = (convert(info, v) for v in value)
        return None if actual is None else low <= actual <= high
    if op in ("in", "nin"):
        values = [convert(info, v) for v in value]
        if op == "in":
            return None if actual is None else actual in values
        return actual is None or actual not in values
    if op == "contains":
        if info.type == "array":
            return None if actual is None else value in actual
        return None if actual is None else str(value).lower() in actual.lower()
    if op == "near":
        centre, tolerance = (convert(info, v) for v in value)
        return None if actual is None else abs(actual - centre) <= tolerance
    raise Invalid(f"the reference evaluator does not support {op}")


def json_safe(value: Any):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value
