"""
Record types (docs/structured-records.md, Record types and field roles): JSON Schema plus field
roles. Pure functions over a TypeDef; no database.

    TypeDef.from_row / from_body     a type's schema, roles, aliases, similarity
    fields_of(schema, items_path)    declared dotted paths with their JSON types (record and items)
    canonicalise / validate          aliases → canonical paths; JSON Schema + strict extra keys
    extract                          role values for the columns (id, name, text, time, facets, …)
    classify_change                  additive (same version) or breaking (new version)
    infer                            a proposed type from sample records
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from kb_runtime.errors import Invalid

ROLE_KEYS = ("id", "name", "text", "time", "facets", "measures", "features", "items")
DIRECTIONS = ("maximize", "minimize")
SCALARS = ("string", "number", "integer", "boolean")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# --- paths ------------------------------------------------------------------------------------------
_MISSING = object()


def get_path(obj: Any, path: str, default=None):
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def has_path(obj: Any, path: str) -> bool:
    return get_path(obj, path, _MISSING) is not _MISSING


def set_path(obj: dict, path: str, value) -> None:
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def pop_path(obj: dict, path: str):
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        current = current.get(part) if isinstance(current, dict) else None
        if current is None:
            return _MISSING
    return current.pop(parts[-1], _MISSING) if isinstance(current, dict) else _MISSING


# --- schema fields -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldInfo:
    path: str
    type: str                     # string | number | integer | boolean | array | object
    format: str = ""
    log: bool = False
    scope: str = "record"         # record | items

    @property
    def numeric(self) -> bool:
        return self.type in ("number", "integer")

    @property
    def temporal(self) -> bool:
        return self.type == "string" and self.format in ("date-time", "date")


def _type_of(node: dict) -> str:
    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    return t or ("object" if "properties" in node else "string")


def fields_of(schema: dict, items_path: str | None = None) -> dict[tuple[str, str], FieldInfo]:
    """{(scope, path): FieldInfo} for every declared property, nested objects included. The
    items array's object properties are listed with scope `items`, relative to the item."""
    out: dict[tuple[str, str], FieldInfo] = {}

    def walk(node: dict, prefix: str, scope: str):
        for name, child in (node.get("properties") or {}).items():
            if not isinstance(child, dict):
                continue
            path = f"{prefix}{name}"
            kind = _type_of(child)
            out[(scope, path)] = FieldInfo(path, kind, child.get("format", ""), bool(child.get("x-log")), scope)
            if kind == "object":
                walk(child, path + ".", scope)
            elif kind == "array" and scope == "record" and path == items_path:
                item = child.get("items") or {}
                if isinstance(item, dict):
                    walk(item, "", "items")

    walk(schema or {}, "", "record")
    return out


def dynamic_of(schema: dict, items_path: str | None = None) -> dict[tuple[str, str], FieldInfo]:
    """{(scope, object path): FieldInfo of its keys} for objects whose `additionalProperties` is a
    scalar schema: any key under them is a field of that type (e.g. HPO `params.<name>`)."""
    out: dict[tuple[str, str], FieldInfo] = {}

    def walk(node: dict, prefix: str, scope: str):
        extra = node.get("additionalProperties")
        if prefix and isinstance(extra, dict) and _type_of(extra) in SCALARS:
            out[(scope, prefix[:-1])] = FieldInfo("", _type_of(extra), extra.get("format", ""),
                                                  bool(extra.get("x-log")), scope)
        for name, child in (node.get("properties") or {}).items():
            if not isinstance(child, dict):
                continue
            kind = _type_of(child)
            if kind == "object":
                walk(child, f"{prefix}{name}.", scope)
            elif kind == "array" and scope == "record" and f"{prefix}{name}" == items_path:
                item = child.get("items") or {}
                if isinstance(item, dict):
                    walk(item, "", "items")

    walk(schema or {}, "", "record")
    return out


_KEY = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# --- the type definition ---------------------------------------------------------------------------------
def normalise_roles(roles: dict) -> dict:
    roles = dict(roles or {})
    unknown = set(roles) - set(ROLE_KEYS)
    if unknown:
        raise Invalid(f"unknown roles: {', '.join(sorted(unknown))}")

    def as_list(value):
        if value in (None, ""):
            return []
        return [value] if isinstance(value, str) else list(value)

    def as_measures(value):
        if isinstance(value, list):
            value = {v: {} for v in value}
        out = {}
        for path, spec in (value or {}).items():
            spec = dict(spec or {})
            if spec.get("direction") and spec["direction"] not in DIRECTIONS:
                raise Invalid(f"measure {path}: direction must be maximize or minimize")
            out[path] = {k: v for k, v in spec.items() if k in ("unit", "direction")}
        return out

    items = roles.get("items")
    if isinstance(items, str):
        items = {"path": items}
    if items:
        items = {"path": items["path"], "measures": as_measures(items.get("measures")),
                 "features": as_list(items.get("features"))}
    return {"id": roles.get("id") or "", "name": roles.get("name") or "", "text": as_list(roles.get("text")),
            "time": roles.get("time") or "", "facets": as_list(roles.get("facets")),
            "measures": as_measures(roles.get("measures")), "features": as_list(roles.get("features")),
            "items": items or None}


@dataclass
class TypeDef:
    name: str                          # bare name
    qualified: str                     # <owner>:<name>
    version: int
    schema: dict
    roles: dict
    aliases: dict = field(default_factory=dict)
    similarity: dict = field(default_factory=dict)
    title: str = ""
    description: str = ""
    plugin: str | None = None
    status: str = "active"
    id: str = ""

    def __post_init__(self):
        self.roles = normalise_roles(self.roles)
        items = self.roles["items"]
        self.fields = fields_of(self.schema, items["path"] if items else None)
        self.dynamic = dynamic_of(self.schema, items["path"] if items else None)
        self._alias = {}
        for canonical, names in (self.aliases or {}).items():
            for alias in [names] if isinstance(names, str) else names:
                self._alias[alias] = canonical

    @classmethod
    def from_row(cls, row) -> "TypeDef":
        return cls(name=row.name, qualified=f"{row.owner_subject}:{row.name}", version=row.version,
                   schema=row.schema or {}, roles=row.roles or {}, aliases=row.aliases or {},
                   similarity=row.similarity or {}, title=row.title, description=row.description, plugin=row.plugin,
                   status=row.status, id=row.id)

    # -- fields ------------------------------------------------------------------------------------
    def canonical(self, path: str) -> str:
        return self._alias.get(path, path)

    def field(self, path: str, scope: str = "record") -> FieldInfo:
        """The declared field (aliases resolved), or Invalid. Only declared fields reach SQL."""
        if not isinstance(path, str) or not path:
            raise Invalid("a field path is required")
        path = self.canonical(path)
        info = self.fields.get((scope, path))
        if info is None and "." in path:
            parent, _, key = path.rpartition(".")
            template = self.dynamic.get((scope, parent))
            if template is not None and _KEY.match(key):
                info = FieldInfo(path, template.type, template.format, template.log, scope)
        if info is None:
            where = "item " if scope == "items" else ""
            raise Invalid(f"unknown {where}field {path!r} for record type {self.name}")
        return info

    def has_field(self, path: str, scope: str = "record") -> bool:
        try:
            self.field(path, scope)
            return True
        except Invalid:
            return False

    def measure(self, path: str, scope: str = "record") -> dict | None:
        path = self.canonical(path)
        if scope == "items":
            items = self.roles["items"]
            return (items or {}).get("measures", {}).get(path)
        return self.roles["measures"].get(path)

    def check_roles(self) -> None:
        r = self.roles
        for role in ("id", "name", "time"):
            if r[role] and ("record", r[role]) not in self.fields:
                raise Invalid(f"role {role}: {r[role]!r} is not a declared field")
        if not r["id"]:
            raise Invalid("the id role is required")
        for role in ("text", "facets", "features"):
            for path in r[role]:
                if ("record", path) not in self.fields:
                    raise Invalid(f"role {role}: {path!r} is not a declared field")
        for path in r["measures"]:
            info = self.fields.get(("record", path))
            if info is None or not info.numeric:
                raise Invalid(f"role measures: {path!r} is not a declared numeric field")
        if r["time"] and not (self.fields[("record", r["time"])].temporal or self.fields[("record", r["time"])].numeric):
            raise Invalid("role time: the field must be a date-time string or an epoch number")
        items = r["items"]
        if items:
            info = self.fields.get(("record", items["path"]))
            if info is None or info.type != "array":
                raise Invalid(f"role items: {items['path']!r} is not a declared array")
            for path in list(items["measures"]) + items["features"]:
                if ("items", path) not in self.fields:
                    raise Invalid(f"role items: {path!r} is not a declared item field")

    # -- records ---------------------------------------------------------------------------------
    def canonicalise(self, record: dict) -> dict:
        record = copy.deepcopy(record)
        for alias, canonical in self._alias.items():
            value = pop_path(record, alias)
            if value is not _MISSING and not has_path(record, canonical):
                set_path(record, canonical, value)
        return record

    def validate(self, record: dict, strict: bool = True) -> list[str]:
        import jsonschema

        validator = jsonschema.Draft202012Validator(self.schema, format_checker=jsonschema.FormatChecker())
        errors = [f"{'.'.join(str(p) for p in e.absolute_path) or '(record)'}: {e.message}"
                  for e in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))[:20]]
        if strict:
            errors += [f"{path}: not in the schema" for path in undeclared(self.schema, record)]
        if not errors and self.roles["id"] and get_path(record, self.roles["id"]) in (None, ""):
            errors.append(f"{self.roles['id']}: the id is required")
        return errors

    def extract(self, record: dict) -> dict:
        r = self.roles
        external_id = str(get_path(record, r["id"]))
        name = get_path(record, r["name"]) if r["name"] else None
        texts = [str(get_path(record, p)) for p in r["text"] if get_path(record, p) not in (None, "")]
        event_time = parse_time(get_path(record, r["time"])) if r["time"] else None

        def pick(paths, source):
            return {p: get_path(source, p) for p in paths if get_path(source, p) is not None}

        items = []
        if r["items"]:
            for item in get_path(record, r["items"]["path"]) or []:
                if isinstance(item, dict):
                    items.append({"data": item, "measures": pick(r["items"]["measures"], item),
                                  "features": pick(r["items"]["features"], item)})
        return {"external_id": external_id, "name": str(name) if name not in (None, "") else external_id,
                "text": "\n".join(texts), "event_time": event_time, "facets": pick(r["facets"], record),
                "measures": {p: float(v) for p, v in pick(r["measures"], record).items() if _is_number(v)},
                "features": pick(r["features"], record), "items": items}

    def summary(self) -> dict:
        return {"name": self.qualified, "version": self.version, "title": self.title, "description": self.description,
                "plugin": self.plugin, "status": self.status, "roles": self.roles,
                "fields": [{"path": i.path, "type": i.type, "scope": i.scope,
                            **({"format": i.format} if i.format else {}), **({"log": True} if i.log else {})}
                           for i in self.fields.values()]}


def undeclared(schema: dict, record: Any, prefix: str = "") -> list[str]:
    """Keys of `record` not declared where the schema lists properties (strict mode)."""
    out = []
    if not isinstance(record, dict) or not isinstance(schema, dict):
        return out
    props = schema.get("properties")
    if props is None:
        return out
    for key, value in record.items():
        if key not in props:
            if schema.get("additionalProperties") in (None, False):
                out.append(prefix + key)
            continue
        child = props[key] if isinstance(props[key], dict) else {}
        if isinstance(value, dict):
            out += undeclared(child, value, prefix + key + ".")
        elif isinstance(value, list) and isinstance(child.get("items"), dict):
            for item in value:
                out += [p for p in undeclared(child["items"], item, prefix + key + "[].") if p not in out]
    return out


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_time(value) -> datetime.datetime | None:
    if value in (None, ""):
        return None
    if _is_number(value):
        number = float(value)
        if number > 1e12:                 # epoch milliseconds
            number /= 1000
        return datetime.datetime.fromtimestamp(number, datetime.timezone.utc)
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        raise Invalid(f"not an ISO-8601 time: {value!r}")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def coerce(value: str, info: FieldInfo | None):
    """A CSV/XLSX cell to the schema's type (strings stay strings when the field is unknown)."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    kind = info.type if info else "string"
    try:
        if kind == "integer":
            return int(float(text))
        if kind == "number":
            return float(text)
        if kind == "boolean":
            return text.lower() in ("1", "true", "yes", "y")
        if kind in ("array", "object"):
            if text[:1] in "[{":
                return json.loads(text)
            return [part.strip() for part in text.split("##")] if kind == "array" else text
    except ValueError:
        return text                       # left for the schema to reject
    return text


# --- schema evolution ----------------------------------------------------------------------------------
def classify_change(old: TypeDef, new: TypeDef) -> list[str]:
    """Reasons the change is breaking (empty = additive)."""
    reasons = []
    for key, info in old.fields.items():
        other = new.fields.get(key)
        if other is None:
            reasons.append(f"field {info.path} removed")
        elif other.type != info.type and not {other.type, info.type} <= {"number", "integer"}:
            reasons.append(f"field {info.path} changed type {info.type} → {other.type}")
    old_required = set(_required(old.schema))
    for path in set(_required(new.schema)) - old_required:
        reasons.append(f"field {path} became required")
    if old.roles["id"] != new.roles["id"]:
        reasons.append("the id role changed")
    if (old.roles["items"] or {}).get("path") != (new.roles["items"] or {}).get("path"):
        reasons.append("the items role changed")
    return reasons


def _required(schema: dict, prefix: str = "") -> list[str]:
    out = [prefix + r for r in schema.get("required") or []]
    for name, child in (schema.get("properties") or {}).items():
        if isinstance(child, dict) and _type_of(child) == "object" and name in (schema.get("required") or []):
            out += _required(child, prefix + name + ".")
    return out


def check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise Invalid("a record type name is lowercase letters, digits and _, starting with a letter (≤ 64)")
    return name


# --- inference ------------------------------------------------------------------------------------------
_TIME_NAME = re.compile(r"(^|_)(at|time|date|timestamp|ts)$|^(time|date|timestamp)")
_MINIMIZE = re.compile(r"loss|error|err$|latency|duration|time_s|seconds|hours|cost|regret|rmse|mae|mse")
_MAXIMIZE = re.compile(r"acc|f1|auc|score|throughput|precision|recall|map$|bleu|rouge|reward")
_ID_NAME = re.compile(r"(^|_)id$")
_FEATURE_PARENTS = ("params", "config", "settings", "hyperparameters", "hparams")


def _json_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _looks_time(name: str, values: list) -> bool:
    if not _TIME_NAME.search(name.split(".")[-1].lower()):
        return False
    for v in values:
        if _is_number(v):
            if not 1e8 < float(v) < 1e13:
                return False
        else:
            try:
                datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                return False
    return bool(values)


def infer(samples: list[dict], name: str = "inferred") -> dict:
    """A proposed type (schema + roles) from sample records (docs/structured-records.md, Schema
    inference). Nothing is stored."""
    samples = [s for s in samples if isinstance(s, dict)]
    if not samples:
        raise Invalid("give at least one sample record")
    values: dict[str, list] = {}
    order: list[str] = []
    item_samples: dict[str, list[dict]] = {}

    def walk(obj: dict, prefix: str):
        for key, value in obj.items():
            path = prefix + key
            if path not in values:
                values[path] = []
                order.append(path)
            values[path].append(value)
            if isinstance(value, dict):
                walk(value, path + ".")
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                item_samples.setdefault(path, []).extend(value)

    for sample in samples:
        walk(sample, "")
    total = len(samples)

    def node_for(path: str, vals: list) -> dict:
        types = {_json_type(v) for v in vals if v is not None}
        if types <= {"integer", "number"} and types:
            node = {"type": "integer" if types == {"integer"} else "number"}
        elif len(types) == 1:
            node = {"type": types.pop()}
        else:
            node = {"type": "string"}
        if node["type"] == "string" and _looks_time(path, vals):
            node["format"] = "date-time"
        return node

    schema: dict = {"type": "object", "properties": {}, "required": []}
    nodes: dict[str, dict] = {}
    for path in order:
        vals = [v for v in values[path] if v is not None]
        node = node_for(path, vals)
        parent_path, _, key = path.rpartition(".")
        parent = schema if not parent_path else nodes.get(parent_path)
        if parent is None or parent.get("type") != "object":
            continue
        parent.setdefault("properties", {})[key] = node
        nodes[path] = node
        present = len(values[path])
        if not parent_path and present >= 0.95 * total:
            schema["required"].append(key)

    roles: dict = {"text": [], "facets": [], "measures": {}, "features": []}
    items_path = max(item_samples, key=lambda p: len(item_samples[p]), default=None)
    for path in order:
        node = nodes.get(path)
        if node is None or (items_path and (path == items_path or path.startswith(items_path + "."))):
            continue
        vals = [v for v in values[path] if v is not None]
        leaf = path.split(".")[-1].lower()
        parent = path.split(".")[0].lower() if "." in path else ""
        kind = node["type"]
        if kind == "string" and node.get("format") == "date-time":
            roles.setdefault("time", path)
            continue
        if kind == "string":
            distinct = len({str(v) for v in vals})
            avg = sum(len(str(v)) for v in vals) / max(1, len(vals))
            if not roles.get("id") and _ID_NAME.search(leaf) and distinct == len(vals) == total:
                roles["id"] = path
            elif not roles.get("name") and not parent and (leaf in ("name", "title") or leaf.endswith("_name")):
                roles["name"] = path
            elif avg > 80:
                roles["text"].append(path)
            elif parent in _FEATURE_PARENTS:
                roles["features"].append(path)
            elif distinct <= 50 and distinct <= max(1, 0.2 * len(vals)):
                roles["facets"].append(path)
        elif kind in ("number", "integer"):
            if _ID_NAME.search(leaf):
                continue
            if parent in _FEATURE_PARENTS:
                roles["features"].append(path)
            else:
                direction = "minimize" if _MINIMIZE.search(leaf) else "maximize" if _MAXIMIZE.search(leaf) else None
                roles["measures"][path] = {"direction": direction} if direction else {}
        elif kind == "boolean":
            roles["features" if parent in _FEATURE_PARENTS else "facets"].append(path)
    if items_path:
        rows = item_samples[items_path]
        item_schema = infer(rows, "items")["schema"] if rows else {"type": "object"}
        item_schema.pop("required", None)
        nodes[items_path].clear()
        nodes[items_path].update({"type": "array", "items": item_schema})
        item_fields = fields_of({"type": "object", "properties": item_schema.get("properties", {})})
        measures, features = {}, []
        for (_, path), info in item_fields.items():
            leaf = path.split(".")[-1].lower()
            if info.numeric and path.split(".")[0] not in _FEATURE_PARENTS and not _ID_NAME.search(leaf):
                direction = "minimize" if _MINIMIZE.search(leaf) else "maximize" if _MAXIMIZE.search(leaf) else None
                measures[path] = {"direction": direction} if direction else {}
            elif info.type in SCALARS:
                features.append(path)
        roles["items"] = {"path": items_path, "measures": measures, "features": features}
    if not roles.get("id"):
        candidates = [p for p in order if p in nodes and nodes[p]["type"] == "string" and "." not in p
                      and len({str(v) for v in values[p]}) == total]
        if candidates:
            roles["id"] = candidates[0]
    roles = {k: v for k, v in roles.items() if v not in ([], {}, None)}
    return {"name": name, "schema": schema, "roles": roles, "suggested_by": "rules"}
