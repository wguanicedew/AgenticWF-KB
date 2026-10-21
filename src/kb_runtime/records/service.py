"""
RecordService: record types and the records of `records` knowledge bases
(docs/structured-records.md).

Types are owned rows (owner, visibility, groups), named `<owner>:<name>`; a bare name resolves
among the caller's own types, then types shared with the caller's groups, then `system`. Records
carry no owner: they follow their KB. The definitions of the types a readable KB uses are
readable through that KB.

Tasks: `records:index` (text roles → kb_embeddings, postprocess), `records:file` (a JSON / JSONL /
CSV / XLSX upload), `records:migrate` (a breaking change), `maint:record_stats` (schedule).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json

import anyio
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope, get_engine
from kb_core.orm.models.ext import Record, RecordFieldStat, RecordItem, RecordType, RecordVersion
from kb_core.orm.models.indexes import ensure_vector_index
from kb_core.orm.models.kb import SOURCE_RECORD, Embedding, Knowledge, KnowledgeBase, new_id, utcnow
from kb_runtime import access, features
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound
from kb_runtime.records import plugins as plugin_registry
from kb_runtime.records import query as rq
from kb_runtime.records import similarity as sim
from kb_runtime.records.schema import TypeDef, check_name, classify_change, coerce, get_path, infer, parse_time, set_path
from kb_runtime.tasks import queue
from kb_runtime.tasks.runner import HandlerSpec, ScheduleSpec, TaskContext

LOG = get_logger(__name__)

SYSTEM = access.SYSTEM_OWNER
ENVELOPE = ("_valid_from", "_valid_until", "_expires_at", "_deprecated", "record_type")
LIMITS = {"statement_timeout": 5, "max_rows": 1000, "max_items": 100_000, "max_record_bytes": 5 * 1024 * 1024,
          "max_batch": 10_000, "max_infer_samples": 1000, "stats_interval": 3600}
CANDIDATES = 2000


def limit(name: str) -> int:
    if config_has_option("records", name):
        return int(config_get("records", name))
    return LIMITS[name]


def schedules() -> tuple[ScheduleSpec, ...]:
    return (ScheduleSpec("record_stats", limit("stats_interval"), "maint:record_stats"),)


def type_dict(row: RecordType) -> dict:
    td = TypeDef.from_row(row)
    return {"id": row.id, "name": td.qualified, "bare_name": row.name, "version": row.version, "title": row.title,
            "description": row.description, "plugin": row.plugin, "status": row.status,
            "owner_subject": row.owner_subject, "visibility": row.visibility, "groups": list(row.groups or []),
            "schema": row.schema, "roles": td.roles, "similarity": row.similarity, "aliases": row.aliases,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}


def record_dict(row: Record, items: list | None = None, fields: list[str] | None = None) -> dict:
    data = {"id": row.id, "knowledge_base_id": row.knowledge_base_id, "record_type": row.record_type,
            "type_version": row.type_version, "external_id": row.external_id, "name": row.name,
            "event_time": row.event_time.isoformat() if row.event_time else None, "facets": row.facets,
            "measures": row.measures, "version": row.version,
            "deprecated": row.deprecated_at is not None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}
    if fields is not None:
        data["fields"] = {f: get_path(row.data, f) for f in fields}
    else:
        data["data"] = row.data
    if items is not None:
        data["items"] = [{"item_index": i.item_index, "data": i.data, "derived": i.derived} for i in items]
    return data


def _cell(value):
    return rq.json_safe(value)


class RecordService:
    def __init__(self, models, kbs, storage):
        self.models = models
        self.kbs = kbs
        self.storage = storage
        self.plugins = plugin_registry.enabled()

    # == types =========================================================================================
    async def sync_plugin_types(self) -> list[str]:
        """Plugin types as rows owned by `system` with visibility `all` (as system models)."""
        names = []
        async with async_session_scope() as session:
            for plugin in self.plugins.values():
                for definition in plugin.types:
                    td = TypeDef(name=definition["name"], qualified=f"{SYSTEM}:{definition['name']}", version=1,
                                 schema=definition["schema"], roles=definition["roles"],
                                 aliases=definition.get("aliases") or {})
                    td.check_roles()
                    row = (await session.execute(sa.select(RecordType).where(
                        RecordType.owner_subject == SYSTEM, RecordType.name == definition["name"])
                        .order_by(RecordType.version.desc()).limit(1))).scalar_one_or_none()
                    if row is None:
                        row = RecordType(owner_subject=SYSTEM, visibility="all", groups=[], name=definition["name"],
                                         version=1)
                        session.add(row)
                    row.title, row.description = definition.get("title", ""), definition.get("description", "")
                    row.plugin, row.schema, row.roles = plugin.name, definition["schema"], definition["roles"]
                    row.similarity, row.aliases = definition.get("similarity") or {}, definition.get("aliases") or {}
                    row.status = "active"
                    names.append(f"{SYSTEM}:{definition['name']}")
        return names

    @staticmethod
    def _split(name: str) -> tuple[str | None, str, int | None]:
        """(owner or None, bare name, version or None) of `[owner:]name[@version]`."""
        version = None
        if "@" in name:
            name, _, v = name.rpartition("@")
            version = int(v) if v.isdigit() else None
        owner, sep, bare = name.rpartition(":")
        return (owner if sep else None), bare, version

    async def _kb_types(self, session, kb_ids: list[str]) -> set[str]:
        """Qualified types used by these KBs (allowed or present), so readable through them."""
        if not kb_ids:
            return set()
        out = set()
        for (cfg,) in (await session.execute(sa.select(KnowledgeBase.records_config).where(
                KnowledgeBase.id.in_(kb_ids)))).all():
            out |= set((cfg or {}).get("allowed_types") or [])
            if (cfg or {}).get("default_type"):
                out.add(cfg["default_type"])
        out |= set((await session.execute(sa.select(Record.record_type).where(
            Record.knowledge_base_id.in_(kb_ids), Record.deleted_at.is_(None)).distinct())).scalars())
        return out

    async def _readable_kb_ids(self, session, ctx: AuthContext, refs: list[str] | None = None,
                               records_only: bool = True) -> list[str]:
        ids = await self.kbs.readable_ids(session, ctx, refs)
        if not ids or not records_only:
            return ids
        kinds = dict((await session.execute(sa.select(KnowledgeBase.id, KnowledgeBase.type).where(
            KnowledgeBase.id.in_(ids)))).all())
        if refs:
            wrong = [i for i in ids if kinds.get(i) != "records"]
            if wrong:
                raise Invalid("not a records knowledge base")
        return [i for i in ids if kinds.get(i) == "records"]

    async def resolve(self, session, ctx: AuthContext, name: str, kb_ids: list[str] | None = None,
                      version: int | None = None) -> RecordType:
        """The type `name` (bare or qualified, optional @version) the caller can read: the latest
        version unless one is asked for. Unreadable = NotFound."""
        owner, bare, at = self._split(str(name or ""))
        version = version or at
        query = sa.select(RecordType).where(RecordType.name == bare)
        if owner is not None:
            query = query.where(RecordType.owner_subject == owner)
        if version:
            query = query.where(RecordType.version == version)
        rows = list((await session.execute(query.order_by(RecordType.version.desc()))).scalars())
        through_kbs = await self._kb_types(session, kb_ids or [])
        readable = [r for r in rows if access.can_see(ctx, r) or f"{r.owner_subject}:{r.name}" in through_kbs]
        if not readable:
            raise NotFound(f"record type {name} not found")
        if owner is None:
            # own first, then shared with the caller's groups, then system
            def level(row):
                if row.owner_subject == ctx.subject:
                    return 0
                if row.owner_subject == SYSTEM:
                    return 2
                return 1
            best = min(level(r) for r in readable)
            readable = [r for r in readable if level(r) == best]
            owners = {r.owner_subject for r in readable}
            if len(owners) > 1:
                raise Invalid(f"record type name {bare!r} is ambiguous; use one of "
                              + ", ".join(sorted(f"{o}:{bare}" for o in owners)))
        return readable[0]          # highest version first

    async def typedef(self, session, ctx, name, kb_ids=None, version=None) -> TypeDef:
        return TypeDef.from_row(await self.resolve(session, ctx, name, kb_ids, version))

    async def list_types(self, ctx: AuthContext, all_versions: bool = False) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb_ids = await self._readable_kb_ids(session, ctx)
            through = await self._kb_types(session, kb_ids)
            rows = (await session.execute(sa.select(RecordType).order_by(
                RecordType.owner_subject, RecordType.name, RecordType.version.desc()))).scalars().all()
        out, seen = [], set()
        for row in rows:
            key = (row.owner_subject, row.name)
            if not (access.can_see(ctx, row) or f"{row.owner_subject}:{row.name}" in through):
                continue
            if key in seen and not all_versions:
                continue
            seen.add(key)
            out.append(type_dict(row))
        return out

    async def get_type(self, ctx: AuthContext, name: str) -> dict:
        async with async_session_scope(commit=False) as session:
            kb_ids = await self._readable_kb_ids(session, ctx)
            return type_dict(await self.resolve(session, ctx, name, kb_ids))

    def _definition(self, body: dict, name: str) -> TypeDef:
        if not isinstance(body.get("schema"), dict) or body["schema"].get("type", "object") != "object":
            raise Invalid("schema must be a JSON Schema object with type object")
        import jsonschema

        try:
            jsonschema.Draft202012Validator.check_schema(body["schema"])
        except jsonschema.SchemaError as exc:
            raise Invalid(f"invalid JSON Schema: {exc.message}")
        td = TypeDef(name=name, qualified=name, version=1, schema=body["schema"], roles=body.get("roles") or {},
                     aliases=body.get("aliases") or {}, similarity=body.get("similarity") or {})
        td.check_roles()
        return td

    async def register(self, ctx: AuthContext, body: dict) -> dict:
        name = check_name(body.get("name"))
        self._definition(body, name)
        visibility = body.get("visibility") or "private"
        groups = access.check_share(ctx, visibility, body.get("groups") or [])
        owner = SYSTEM if body.get("system") else ctx.subject
        if owner == SYSTEM and not access.is_admin(ctx):
            raise Forbidden("only an admin registers system record types")
        if visibility == "all" and not access.is_admin(ctx):
            raise Forbidden("only an admin shares a record type with everyone")
        async with async_session_scope() as session:
            exists = (await session.execute(sa.select(RecordType.id).where(
                RecordType.owner_subject == owner, RecordType.name == name))).first()
            if exists:
                raise Conflict(f"you already have a record type {name!r}; change it with PUT")
            row = RecordType(owner_subject=owner, owner_org=ctx.organization_id, visibility=visibility, groups=groups,
                             name=name, version=1, title=body.get("title") or "", description=body.get("description") or "",
                             schema=body["schema"], roles=body.get("roles") or {}, similarity=body.get("similarity") or {},
                             aliases=body.get("aliases") or {})
            session.add(row)
            await session.flush()
            return type_dict(row)

    async def _owned(self, session, ctx, name) -> RecordType:
        row = await self.resolve(session, ctx, name)
        if not access.can_manage(ctx, row):
            raise Forbidden("only the type's owner changes it")
        if row.plugin and not access.is_admin(ctx):
            raise Forbidden("plugin types change with the plugin")
        return row

    async def update_type(self, ctx: AuthContext, name: str, body: dict, new_version: bool = False) -> dict:
        async with async_session_scope() as session:
            row = await self._owned(session, ctx, name)
            old = TypeDef.from_row(row)
            merged = {"schema": body.get("schema", row.schema), "roles": body.get("roles", row.roles),
                      "aliases": body.get("aliases", row.aliases), "similarity": body.get("similarity", row.similarity)}
            new = self._definition(merged, row.name)
            reasons = classify_change(old, new)
            if reasons and not new_version:
                raise Conflict("breaking change: " + "; ".join(reasons) + " (send new_version=true)",
                               {"breaking": reasons})
            if reasons:
                row = RecordType(owner_subject=row.owner_subject, owner_org=row.owner_org, visibility=row.visibility,
                                 groups=list(row.groups or []), name=row.name, version=row.version + 1,
                                 title=body.get("title", row.title), description=body.get("description", row.description),
                                 plugin=row.plugin, **merged)
                session.add(row)
            else:
                for key, value in merged.items():
                    setattr(row, key, value)
                row.title = body.get("title", row.title)
                row.description = body.get("description", row.description)
            if "visibility" in body or "groups" in body:
                visibility = body.get("visibility") or row.visibility
                if visibility == "all" and not access.is_admin(ctx):
                    raise Forbidden("only an admin shares a record type with everyone")
                row.groups = access.check_share(ctx, visibility, body.get("groups", row.groups) or [])
                row.visibility = visibility
            row.updated_at = utcnow()
            await session.flush()
            return {**type_dict(row), "breaking": reasons}

    async def deprecate_type(self, ctx: AuthContext, name: str) -> dict:
        async with async_session_scope() as session:
            row = await self._owned(session, ctx, name)
            await session.execute(sa.update(RecordType).where(
                RecordType.owner_subject == row.owner_subject, RecordType.name == row.name).values(
                status="deprecated", updated_at=utcnow()))
            return {"name": f"{row.owner_subject}:{row.name}", "status": "deprecated"}

    async def infer_type(self, ctx: AuthContext, body: dict, file_name: str | None = None,
                         data: bytes | None = None) -> dict:
        samples = body.get("samples") or body.get("records") or []
        if data is not None:
            samples = parse_file(file_name or "", data, None, {})
        if not isinstance(samples, list) or not 1 <= len(samples) <= limit("max_infer_samples"):
            raise Invalid(f"give 1 to {limit('max_infer_samples')} sample records")
        proposal = infer(samples, body.get("name") or "inferred")
        if body.get("llm"):
            proposal = await self._llm_infer(proposal, samples)
        return proposal

    async def _llm_infer(self, proposal: dict, samples: list) -> dict:
        from kb_runtime.prompts import render

        model_id = await self.models.default_id("KnowledgeQA")
        if not model_id:
            return {**proposal, "llm_note": "no chat model configured"}
        chat = await self.models.chat(model_id)
        reply = await chat.complete([
            {"role": "system", "content": render("records", "infer_system")},
            {"role": "user", "content": render("records", "infer_user", proposal=json.dumps(proposal)[:6000],
                                               samples=json.dumps(samples[:5], default=str)[:6000])}],
            max_tokens=800, temperature=0.2)
        try:
            from kb_runtime.agentic.engine import _json_object

            suggestion = _json_object(reply["content"])
        except ValueError:
            return proposal
        out = dict(proposal)
        for key in ("title", "description"):
            if isinstance(suggestion.get(key), str):
                out[key] = suggestion[key]
        if isinstance(suggestion.get("roles"), dict):
            out["suggested_roles"] = suggestion["roles"]
        out["suggested_by"] = "llm"
        return out

    async def stats(self, ctx: AuthContext, name: str, kb_ref: str | None = None) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb_ids = await self._readable_kb_ids(session, ctx, [kb_ref] if kb_ref else None)
            td = await self.typedef(session, ctx, name, kb_ids)
            rows = (await session.execute(sa.select(RecordFieldStat).where(
                RecordFieldStat.knowledge_base_id.in_(kb_ids or ["-"]),
                RecordFieldStat.record_type == td.qualified))).scalars().all()
        return merge_stats(rows)

    async def operation(self, ctx: AuthContext, name: str, op: str, body: dict, services) -> dict:
        async with async_session_scope(commit=False) as session:
            kb_ids = await self._readable_kb_ids(session, ctx)
            row = await self.resolve(session, ctx, name, kb_ids)
        plugin = self.plugins.get(row.plugin or "")
        operation = plugin.operations.get(op) if plugin else None
        if operation is None:
            raise NotFound(f"record type {name} has no operation {op!r}")
        if not operation.read_only and "knowledge:write" not in ctx.scopes:
            raise Forbidden("missing scopes: knowledge:write")
        return await operation.run(plugin_registry.OpEnv(ctx, services, self), body)

    def read_only_operations(self) -> list[tuple[str, plugin_registry.Operation]]:
        out = []
        for plugin in self.plugins.values():
            for definition in plugin.types:
                for operation in plugin.operations.values():
                    if operation.read_only and operation.tool_name:
                        out.append((f"{SYSTEM}:{definition['name']}", operation))
        return out

    # == records =========================================================================================
    async def _records_kb(self, session, ctx, kb_ref, write=False, for_update=False) -> KnowledgeBase:
        if not features.enabled("records"):
            raise NotFound("records are switched off in this deployment")
        kb = await self.kbs.load(session, ctx, kb_ref, for_update=for_update)
        if kb.type != "records":
            raise Invalid("not a records knowledge base")
        if write and not access.can_contribute(ctx, kb):
            raise Forbidden("you may not add records to this knowledge base")
        return kb

    async def check_records_config(self, ctx: AuthContext, cfg: dict) -> dict:
        """Resolve and store the type names qualified (KB create / update)."""
        cfg = dict(cfg or {})
        async with async_session_scope(commit=False) as session:
            allowed = []
            for name in cfg.get("allowed_types") or []:
                row = await self.resolve(session, ctx, name)
                allowed.append(f"{row.owner_subject}:{row.name}")
            cfg["allowed_types"] = allowed
            if cfg.get("default_type"):
                row = await self.resolve(session, ctx, cfg["default_type"])
                cfg["default_type"] = f"{row.owner_subject}:{row.name}"
        cfg.setdefault("strict", True)
        cfg.setdefault("keep_versions", True)
        return cfg

    async def upsert(self, ctx: AuthContext, kb_ref: str, body: dict, knowledge_id: str | None = None) -> dict:
        if isinstance(body.get("records"), list):
            raw_records = body["records"]
        elif isinstance(body.get("record"), dict):
            raw_records = [body["record"]]
        else:
            raw_records = [{k: v for k, v in body.items() if k not in ("atomic",)}]
        if len(raw_records) > limit("max_batch"):
            raise Invalid(f"at most {limit('max_batch')} records per batch")
        atomic = bool(body.get("atomic"))
        outcomes, changed, touched_types = [], [], set()
        async with async_session_scope() as session:
            kb = await self._records_kb(session, ctx, kb_ref, write=True, for_update=True)
            cfg = kb.records_config or {}
            types: dict[str, TypeDef] = {}
            for index, raw in enumerate(raw_records):
                try:
                    if not isinstance(raw, dict):
                        raise Invalid("a record is an object")
                    record = dict(raw)
                    name = record.pop("record_type", None) or body.get("record_type") or cfg.get("default_type")
                    if not name:
                        raise Invalid("record_type is required (or the KB's default_type)")
                    if name not in types:
                        row = await self.resolve(session, ctx, name, [kb.id])
                        if row.status != "active":
                            raise Invalid(f"record type {name} is deprecated")
                        types[name] = TypeDef.from_row(row)
                    td = types[name]
                    if cfg.get("allowed_types") and td.qualified not in cfg["allowed_types"]:
                        raise Invalid(f"record type {td.qualified} is not allowed in this knowledge base")
                    outcome, row_id = await self._upsert_one(session, kb, td, record, cfg, ctx, knowledge_id)
                    outcomes.append({"index": index, "outcome": outcome, "id": row_id})
                    if outcome in ("created", "updated"):
                        changed.append(row_id)
                        touched_types.add(td.qualified)
                except (Invalid, NotFound) as exc:
                    outcomes.append({"index": index, "outcome": "invalid",
                                     "errors": exc.errors if isinstance(exc, _RecordErrors) else [str(exc)]})
            invalid = [o for o in outcomes if o["outcome"] == "invalid"]
            if atomic and invalid:
                await session.rollback()
                return {"outcomes": outcomes, "atomic": True, "written": False, **_counts(outcomes)}
            if changed:
                await queue.enqueue(session, "records:index", {"knowledge_base_id": kb.id, "record_ids": changed},
                                    owner_subject=kb.owner_subject, owner_org=kb.owner_org, knowledge_base_id=kb.id,
                                    queue="postprocess", max_retry=3)
            kb_id = kb.id
        for qualified in touched_types:
            await self.recompute_stats(kb_id, qualified)
        return {"outcomes": outcomes, **_counts(outcomes)}

    async def _upsert_one(self, session, kb, td: TypeDef, record: dict, cfg: dict, ctx, knowledge_id):
        envelope = {k: record.pop(k) for k in list(record) if k in ENVELOPE}
        record = td.canonicalise(record)
        plugin = self.plugins.get(td.plugin or "")
        if plugin:
            record = plugin.normalize(record, cfg)
        errors = td.validate(record, strict=cfg.get("strict", True))
        if plugin:
            errors += plugin.validate(record, cfg)
        if errors:
            raise _RecordErrors(errors)
        derived_items: list[dict] = []
        if plugin:
            record, derived_items = plugin.derive(record)
        size = len(json.dumps(record, default=str))
        if size > limit("max_record_bytes"):
            raise Invalid(f"the record is larger than {limit('max_record_bytes')} bytes")
        ex = td.extract(record)
        if len(ex["items"]) > limit("max_items"):
            raise Invalid(f"at most {limit('max_items')} items per record")
        fresh = {"valid_from": parse_time(envelope.get("_valid_from")),
                 "valid_until": parse_time(envelope.get("_valid_until")),
                 "expires_at": parse_time(envelope.get("_expires_at"))}
        deprecated = str(envelope.get("_deprecated", "")).lower() in ("1", "true", "yes")
        digest = hashlib.sha256(json.dumps([record, {k: str(v) for k, v in fresh.items()}, deprecated],
                                           sort_keys=True, default=str).encode()).hexdigest()
        row = (await session.execute(sa.select(Record).where(
            Record.knowledge_base_id == kb.id, Record.record_type == td.qualified,
            Record.external_id == ex["external_id"], Record.deleted_at.is_(None)).with_for_update())).scalar_one_or_none()
        if row is not None and row.content_hash == digest and row.type_version == td.version:
            return "unchanged", row.id
        if row is None:
            row = Record(id=new_id(), knowledge_base_id=kb.id, record_type=td.qualified, external_id=ex["external_id"],
                         version=1, created_by=ctx.subject)
            session.add(row)
            outcome = "created"
        else:
            if cfg.get("keep_versions", True):
                session.add(RecordVersion(record_id=row.id, version=row.version, type_version=row.type_version,
                                          data=row.data))
            row.version += 1
            await session.execute(sa.delete(RecordItem).where(RecordItem.record_id == row.id))
            outcome = "updated"
        row.type_version, row.name, row.text, row.data = td.version, ex["name"][:512], ex["text"], record
        row.event_time = ex["event_time"]
        row.facets, row.measures, row.features = ex["facets"], ex["measures"], ex["features"]
        row.content_hash, row.knowledge_id = digest, knowledge_id or row.knowledge_id
        row.valid_from, row.valid_until, row.expires_at = fresh["valid_from"], fresh["valid_until"], fresh["expires_at"]
        row.deprecated_at = (row.deprecated_at or utcnow()) if deprecated else None
        row.updated_at = utcnow()
        await session.flush()
        for i, item in enumerate(ex["items"]):
            session.add(RecordItem(record_id=row.id, item_index=i, data=item["data"], measures=item["measures"],
                                   features=item["features"], derived=derived_items[i] if i < len(derived_items) else {}))
        return outcome, row.id

    async def list_records(self, ctx: AuthContext, kb_ref: str, record_type: str | None = None,
                           facets: dict | None = None, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
        async with async_session_scope(commit=False) as session:
            kb = await self._records_kb(session, ctx, kb_ref)
            query = sa.select(Record).where(Record.knowledge_base_id == kb.id, Record.deleted_at.is_(None))
            if record_type:
                td = await self.typedef(session, ctx, record_type, [kb.id])
                query = query.where(Record.record_type == td.qualified)
            for path, value in (facets or {}).items():
                query = query.where(Record.facets[path].astext == str(value))
            total = (await session.execute(sa.select(sa.func.count()).select_from(query.subquery()))).scalar_one()
            rows = (await session.execute(query.order_by(Record.event_time.desc().nullslast(), Record.id)
                                          .offset((page - 1) * page_size).limit(page_size))).scalars().all()
            return [record_dict(r) for r in rows], total

    async def _readable_record(self, session, ctx, record_id) -> Record:
        row = await session.get(Record, record_id)
        if row is None or row.deleted_at is not None:
            raise NotFound("record not found")
        kb = await session.get(KnowledgeBase, row.knowledge_base_id)
        if kb is None or kb.deleted_at is not None or not access.can_see(ctx, kb):
            raise NotFound("record not found")
        return row

    async def get_record(self, ctx: AuthContext, record_id: str, version: int | None = None) -> dict:
        async with async_session_scope(commit=False) as session:
            row = await self._readable_record(session, ctx, record_id)
            items = (await session.execute(sa.select(RecordItem).where(RecordItem.record_id == row.id)
                                           .order_by(RecordItem.item_index))).scalars().all()
            out = record_dict(row, list(items))
            if version and version != row.version:
                old = (await session.execute(sa.select(RecordVersion).where(
                    RecordVersion.record_id == row.id, RecordVersion.version == version))).scalar_one_or_none()
                if old is None:
                    raise NotFound(f"version {version} is not kept")
                out = {**out, "version": old.version, "type_version": old.type_version, "data": old.data,
                       "items": None, "replaced_at": old.replaced_at.isoformat()}
            out["versions"] = list((await session.execute(sa.select(RecordVersion.version).where(
                RecordVersion.record_id == row.id).order_by(RecordVersion.version))).scalars())
            return out

    async def delete_record(self, ctx: AuthContext, record_id: str) -> None:
        async with async_session_scope() as session:
            row = await self._readable_record(session, ctx, record_id)
            kb = await session.get(KnowledgeBase, row.knowledge_base_id)
            if not access.can_contribute(ctx, kb):
                raise Forbidden("you may not delete records of this knowledge base")
            row.deleted_at = utcnow()
            await session.execute(sa.delete(Embedding).where(Embedding.source_type == SOURCE_RECORD,
                                                             Embedding.source_id == row.id))
            kb_id, qualified = kb.id, row.record_type
        await self.recompute_stats(kb_id, qualified)

    # == query, aggregate, similar ===========================================================================
    async def _scope(self, session, ctx, body) -> tuple[list[str], TypeDef]:
        kb_ids = await self._readable_kb_ids(session, ctx, body.get("knowledge_base_ids") or None)
        if not body.get("record_type"):
            raise Invalid("record_type is required")
        td = await self.typedef(session, ctx, body["record_type"], kb_ids)
        return kb_ids, td

    async def _types_of(self, session, td: TypeDef) -> list[str]:
        return [td.qualified]

    async def _run(self, session, sql: str, params: dict):
        await session.execute(sa.text(f"SET LOCAL statement_timeout = '{int(limit('statement_timeout'))}s'"))
        try:
            return await session.execute(sa.text(sql), params)
        except sa.exc.DBAPIError as exc:
            text = str(getattr(exc, "orig", exc)).lower()
            if "statement timeout" in text or "canceling statement" in text:
                raise Invalid("the query took too long; narrow the filter") from None
            raise

    async def query(self, ctx: AuthContext, body: dict) -> dict:
        async with async_session_scope(commit=False) as session:
            kb_ids, td = await self._scope(session, ctx, body)
            sql, count_sql, params, fields = rq.compile_query(td, body, limit("max_rows"))
            params.update(kbs=kb_ids, types=[td.qualified], with_deprecated=bool(body.get("include_deprecated")))
            rows = (await self._run(session, sql, params)).mappings().all()
            total = (await self._run(session, count_sql, params)).scalar_one()
        default_fields = fields or [p for p in ([td.roles["name"]] if td.roles["name"] else []) + td.roles["facets"]
                                    + list(td.roles["measures"]) if p]
        out = []
        for row in rows:
            out.append({"id": row["id"], "knowledge_base_id": row["knowledge_base_id"], "record_type": row["record_type"],
                        "external_id": row["external_id"], "name": row["name"],
                        "event_time": row["event_time"].isoformat() if row["event_time"] else None,
                        "version": row["version"], "fields": {f: get_path(row["data"], f) for f in default_fields}})
        return {"record_type": td.qualified, "total": total, "records": out}

    async def aggregate(self, ctx: AuthContext, body: dict) -> dict:
        async with async_session_scope(commit=False) as session:
            kb_ids, td = await self._scope(session, ctx, body)
            sql, params, groups, metrics = rq.compile_aggregate(td, body, limit("max_rows"))
            params.update(kbs=kb_ids, types=[td.qualified], with_deprecated=bool(body.get("include_deprecated")))
            rows = (await self._run(session, sql, params)).all()
        columns = groups + metrics
        data = [{c: _cell(v) for c, v in zip(columns, row)} for row in rows]
        return {"record_type": td.qualified, "columns": columns, "groups": groups, "metrics": metrics, "rows": data}

    async def similar(self, ctx: AuthContext, body: dict) -> dict:
        async with async_session_scope(commit=False) as session:
            if body.get("record_id"):
                source = await self._readable_record(session, ctx, body["record_id"])
                body = {**body, "record_type": body.get("record_type") or source.record_type}
            else:
                source = None
            kb_ids, td = await self._scope(session, ctx, body)
            if source is not None:
                probe_facets, probe_features, probe_text = source.facets, source.features, source.text
                exclude = source.id
            else:
                probe = td.canonicalise(dict(body.get("probe") or {}))
                if not probe:
                    raise Invalid("give record_id or a probe")
                probe_facets = {p: get_path(probe, p) for p in td.roles["facets"] if get_path(probe, p) is not None}
                probe_features = {p: get_path(probe, p) for p in td.roles["features"] if get_path(probe, p) is not None}
                probe_text = "\n".join(str(get_path(probe, p)) for p in td.roles["text"] if get_path(probe, p))
                exclude = None
            c = rq.Compiler(td)
            where = c.where(body.get("filter"))
            params = {**c.params, "kbs": kb_ids, "types": [td.qualified], "with_deprecated": False}
            rows = (await self._run(session, f"SELECT r.id, r.knowledge_base_id, r.name, r.external_id, r.facets, "
                                             f"r.features, r.measures FROM kb_records r WHERE {rq.BASE_WHERE} AND {where} "
                                             f"LIMIT {CANDIDATES}", params)).mappings().all()
            rows = [r for r in rows if r["id"] != exclude]
            stats = merge_stats((await session.execute(sa.select(RecordFieldStat).where(
                RecordFieldStat.knowledge_base_id.in_(kb_ids or ["-"]),
                RecordFieldStat.record_type == td.qualified, RecordFieldStat.scope == "record"))).scalars().all())
            text_scores = await self._text_similarity(session, kb_ids, [r["id"] for r in rows], probe_text,
                                                      source.id if source is not None else None)
        ranges = {s["field"]: (s["min"], s["max"]) for s in stats if s["min"] is not None}
        logs = {i.path for i in td.fields.values() if i.log}
        weights = (td.similarity or {}).get("weights")
        scored = []
        for r in rows:
            parts = {"text": text_scores.get(r["id"]) if probe_text else None,
                     "facets": sim.facet_similarity(probe_facets, r["facets"] or {}),
                     "features": sim.features_similarity(probe_features, r["features"] or {}, ranges, logs)}
            scored.append({"id": r["id"], "knowledge_base_id": r["knowledge_base_id"], "name": r["name"],
                           "external_id": r["external_id"], "facets": r["facets"], "measures": r["measures"],
                           "score": round(sim.combine(parts, weights), 4),
                           "breakdown": {k: (round(v, 4) if v is not None else None) for k, v in parts.items()}})
        scored.sort(key=lambda s: (-s["score"], s["id"]))
        top = max(1, min(int(body.get("limit") or 10), 50))
        return {"record_type": td.qualified, "records": scored[:top]}

    async def _text_similarity(self, session, kb_ids, record_ids, probe_text, source_id) -> dict[str, float]:
        if not record_ids or not (probe_text or source_id):
            return {}
        if source_id:
            row = (await session.execute(sa.select(Embedding.embedding, Embedding.dimension).where(
                Embedding.source_type == SOURCE_RECORD, Embedding.source_id == source_id,
                Embedding.dimension > 0))).first()
            if row is None:
                return {}
            vector, dim = list(row[0].to_list() if hasattr(row[0], "to_list") else row[0]), row[1]
        else:
            model_id = (await session.execute(sa.select(KnowledgeBase.embedding_model_id).where(
                KnowledgeBase.id.in_(kb_ids), KnowledgeBase.embedding_model_id.isnot(None)).limit(1))).scalar()
            if not model_id:
                return {}
            embedder = await self.models.embedder(model_id)
            vector, dim = (await embedder.embed([probe_text]))[0], embedder.dimension
        q = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
        rows = await session.execute(sa.text(
            f"SELECT source_id, 1 - (embedding::halfvec({int(dim)}) <=> CAST(:q AS halfvec({int(dim)}))) "
            f"FROM kb_embeddings WHERE source_type = {SOURCE_RECORD} AND dimension = {int(dim)} "
            "AND source_id = ANY(:ids)"), {"q": q, "ids": record_ids})
        return {sid: max(0.0, float(score)) for sid, score in rows.all() if score is not None}

    # == tasks ===============================================================================================
    def handlers(self) -> dict[str, HandlerSpec]:
        return {"records:index": HandlerSpec(self._index_task, 900),
                "records:file": HandlerSpec(self._file_task, 1800, self._file_failed),
                "records:migrate": HandlerSpec(self._migrate_task, 3600),
                "maint:record_stats": HandlerSpec(self._stats_task, 1800)}

    async def _index_task(self, ctx: TaskContext) -> None:
        """Text roles (name + text) into kb_embeddings, source_type 5, chunk_id = record id."""
        kb_id, ids = ctx.payload["knowledge_base_id"], ctx.payload["record_ids"]
        async with async_session_scope(commit=False) as session:
            kb = await session.get(KnowledgeBase, kb_id)
            rows = (await session.execute(sa.select(Record).where(Record.id.in_(ids), Record.deleted_at.is_(None)))).scalars().all()
            for row in rows:
                session.expunge(row)
        if kb is None or kb.deleted_at is not None:
            return
        texts = [f"{r.name}\n{r.text}".strip() for r in rows]
        vectors, dim = None, 0
        if rows and kb.embedding_model_id and kb.indexing_strategy.get("vector_enabled", True):
            embedder = await self.models.embedder(kb.embedding_model_id)
            await anyio.to_thread.run_sync(ensure_vector_index, get_engine(), embedder.dimension)
            vectors, dim = await embedder.embed(texts), embedder.dimension
        async with async_session_scope() as session:
            await session.execute(sa.delete(Embedding).where(Embedding.source_type == SOURCE_RECORD,
                                                             Embedding.source_id.in_(ids)))
            for i, row in enumerate(rows):
                current = await session.get(Record, row.id)
                if current is None or current.deleted_at is not None:
                    continue
                session.add(Embedding(source_id=row.id, source_type=SOURCE_RECORD, chunk_id=row.id, knowledge_id="",
                                      knowledge_base_id=kb.id, owner_subject=kb.owner_subject, visibility="all",
                                      groups=[], content=texts[i], dimension=dim if vectors else 0,
                                      embedding=vectors[i] if vectors else None, is_enabled=True,
                                      valid_from=row.valid_from,
                                      valid_until=min((d for d in (row.valid_until, row.expires_at) if d), default=None),
                                      is_deprecated=row.deprecated_at is not None))

    async def upload_file(self, ctx: AuthContext, kb_ref: str, file_name: str, data: bytes, fields: dict) -> dict:
        """A records file: stored, then imported by a `records:file` task; the report is in the
        upload's metadata (`records_report`)."""
        from kb_runtime.services.knowledge import knowledge_dict
        from kb_runtime.storage import owner_scope, safe_name

        name = safe_name(fields.get("fileName") or file_name)
        if not name.lower().endswith((".json", ".jsonl", ".csv", ".xlsx")):
            raise Invalid("a records knowledge base takes .json, .jsonl, .csv or .xlsx files")
        column_map = fields.get("column_map") or {}
        if isinstance(column_map, str):
            try:
                column_map = json.loads(column_map) if column_map.strip() else {}
            except ValueError:
                raise Invalid("column_map must be JSON")
        async with async_session_scope() as session:
            kb = await self._records_kb(session, ctx, kb_ref, write=True, for_update=True)
            item = Knowledge(id=new_id(), knowledge_base_id=kb.id, type="file", title=name.rsplit(".", 1)[0],
                             file_name=name, file_type=name.rsplit(".", 1)[-1].lower(), file_size=len(data),
                             file_hash=hashlib.sha256(data).hexdigest(), owner_subject=ctx.subject,
                             owner_org=ctx.organization_id, visibility="all", groups=[], parse_status="pending",
                             meta={"record_type": fields.get("record_type") or "", "column_map": column_map})
            item.file_path = f"{owner_scope(ctx.subject)}/{kb.id}/{item.id}/original/{name}"
            await anyio.to_thread.run_sync(self.storage.save, item.file_path, data)
            session.add(item)
            await session.flush()
            await queue.enqueue(session, "records:file", {"knowledge_id": item.id, "subject": ctx.subject,
                                                          "auth": ctx.to_dict()},
                                owner_subject=ctx.subject, owner_org=ctx.organization_id, knowledge_base_id=kb.id,
                                max_retry=1)
            return knowledge_dict(item)

    async def _file_task(self, ctx: TaskContext) -> None:
        async with async_session_scope(commit=False) as session:
            item = await session.get(Knowledge, ctx.payload["knowledge_id"])
            if item is None or item.deleted_at is not None:
                return
            session.expunge(item)
        auth = AuthContext.from_dict(ctx.payload["auth"])
        await self._set_upload(item.id, "processing")
        data = await anyio.to_thread.run_sync(self.storage.read, item.file_path)
        async with async_session_scope(commit=False) as session:
            kb = await session.get(KnowledgeBase, item.knowledge_base_id)
            td = None
            name = (item.meta or {}).get("record_type") or (kb.records_config or {}).get("default_type")
            if name:
                td = await self.typedef(session, auth, name, [kb.id])
        records = parse_file(item.file_name, data, td, (item.meta or {}).get("column_map") or {})
        body = {"records": records}
        if td is not None:
            body["record_type"] = td.qualified
        report = await self.upsert(auth, item.knowledge_base_id, body, knowledge_id=item.id)
        await self._set_upload(item.id, "completed", report)

    async def _file_failed(self, ctx: TaskContext, error: str) -> None:
        await self._set_upload(ctx.payload["knowledge_id"], "failed", error=error)

    async def _set_upload(self, knowledge_id: str, status: str, report: dict | None = None, error: str = "") -> None:
        from kb_runtime.services.knowledge import event

        async with async_session_scope() as session:
            item = await session.get(Knowledge, knowledge_id)
            if item is None:
                return
            item.parse_status, item.error_message = status, error[:2000]
            if report is not None:
                item.meta = {**(item.meta or {}), "records_report": {k: v for k, v in report.items() if k != "outcomes"}
                             | {"invalid": [o for o in report["outcomes"] if o["outcome"] == "invalid"][:200]}}
            if status == "completed":
                item.processed_at = utcnow()
            await event(session, item, {"type": "knowledge_status", "parse_status": status})

    async def migrate(self, ctx: AuthContext, name: str, body: dict) -> dict:
        """Rewrite records of older versions to the latest one, as a task over the KBs the caller
        can write."""
        async with async_session_scope() as session:
            row = await self._owned(session, ctx, name)
            kb_ids = [kb.id for kb in (await session.execute(sa.select(KnowledgeBase).where(
                KnowledgeBase.deleted_at.is_(None), KnowledgeBase.type == "records"))).scalars()
                if access.can_contribute(ctx, kb)]
            task_id = await queue.enqueue(session, "records:migrate", {
                "record_type": f"{row.owner_subject}:{row.name}", "version": row.version,
                "mapping": body.get("mapping") or {}, "knowledge_base_ids": kb_ids, "auth": ctx.to_dict()},
                owner_subject=ctx.subject, owner_org=ctx.organization_id, max_retry=0)
            return {"task_id": task_id, "record_type": f"{row.owner_subject}:{row.name}", "to_version": row.version,
                    "knowledge_base_ids": kb_ids}

    async def _migrate_task(self, ctx: TaskContext) -> dict:
        p = ctx.payload
        auth = AuthContext.from_dict(p["auth"])
        mapping = p.get("mapping") or {}
        report = {"migrated": 0, "invalid": []}
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(Record).where(
                Record.record_type == p["record_type"], Record.type_version < p["version"],
                Record.knowledge_base_id.in_(p["knowledge_base_ids"] or ["-"]), Record.deleted_at.is_(None)))).scalars().all()
            for row in rows:
                session.expunge(row)
        by_kb: dict[str, list[dict]] = {}
        for row in rows:
            data = json.loads(json.dumps(row.data))
            for old, new in mapping.items():
                from kb_runtime.records.schema import pop_path

                value = pop_path(data, old)
                if value is not None and new:
                    set_path(data, new, value)
            for key, column in (("_valid_from", row.valid_from), ("_valid_until", row.valid_until),
                                ("_expires_at", row.expires_at)):
                if column:
                    data[key] = column.isoformat()
            if row.deprecated_at:
                data["_deprecated"] = True
            by_kb.setdefault(row.knowledge_base_id, []).append(data)
        for kb_id, records in by_kb.items():
            result = await self.upsert(auth, kb_id, {"record_type": p["record_type"], "records": records})
            report["migrated"] += result["created"] + result["updated"] + result["unchanged"]
            report["invalid"] += [o for o in result["outcomes"] if o["outcome"] == "invalid"][:100]
        LOG.info("records:migrate %s → v%s: %s", p["record_type"], p["version"], report)
        return report

    async def _stats_task(self, ctx: TaskContext | None = None) -> int:
        async with async_session_scope(commit=False) as session:
            pairs = (await session.execute(sa.select(Record.knowledge_base_id, Record.record_type).where(
                Record.deleted_at.is_(None)).distinct())).all()
        for kb_id, record_type in pairs:
            await self.recompute_stats(kb_id, record_type)
        return len(pairs)

    async def recompute_stats(self, kb_id: str, qualified: str) -> None:
        """Field statistics of one KB and type: count, min, max, distinct count, top 20 values."""
        owner, _, bare = qualified.rpartition(":")
        async with async_session_scope() as session:
            row = (await session.execute(sa.select(RecordType).where(
                RecordType.owner_subject == owner, RecordType.name == bare).order_by(
                RecordType.version.desc()).limit(1))).scalar_one_or_none()
            if row is None:
                return
            td = TypeDef.from_row(row)
            paths = [(p, "record") for p in td.roles["facets"] + list(td.roles["measures"]) + td.roles["features"]]
            if td.roles["time"]:
                paths.append((td.roles["time"], "record"))
            items = td.roles["items"]
            if items:
                paths += [(p, "items") for p in list(items["measures"]) + items["features"]]
            await session.execute(sa.delete(RecordFieldStat).where(RecordFieldStat.knowledge_base_id == kb_id,
                                                                   RecordFieldStat.record_type == qualified))
            for path, scope in dict.fromkeys(paths):
                info = td.fields.get((scope, path))
                if info is None:
                    continue
                source = ("FROM kb_records r WHERE r.knowledge_base_id = :kb AND r.record_type = :t AND r.deleted_at IS NULL"
                          if scope == "record" else
                          "FROM kb_record_items i JOIN kb_records r ON r.id = i.record_id WHERE r.knowledge_base_id = :kb "
                          "AND r.record_type = :t AND r.deleted_at IS NULL")
                table = "r" if scope == "record" else "i"
                value = f"{table}.data #> CAST(:p AS text[])"
                params = {"kb": kb_id, "t": qualified, "p": path.split(".")}
                if info.type in ("array", "object"):
                    continue
                numeric = (f"CASE WHEN jsonb_typeof({value}) = 'number' THEN ({value})::text::float8 END"
                           if info.numeric else "NULL::float8")
                if info.temporal or (scope == "record" and path == td.roles["time"]):
                    numeric = "extract(epoch FROM r.event_time)" if path == td.roles["time"] else \
                        f"extract(epoch FROM ({table}.data #>> CAST(:p AS text[]))::timestamptz)"
                agg = (await session.execute(sa.text(
                    f"SELECT count({value}), min({numeric}), max({numeric}), count(DISTINCT {value}) {source}"), params)).first()
                if not agg or not agg[0]:
                    continue
                top = [] if info.numeric else [
                    {"value": json.loads(v) if isinstance(v, str) else v, "count": n} for v, n in (await session.execute(
                        sa.text(f"SELECT ({value})::text AS v, count(*) AS n {source} AND {value} IS NOT NULL "
                                "GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20"), params)).all()]
                stmt = insert(RecordFieldStat).values(
                    knowledge_base_id=kb_id, record_type=qualified, field=path, scope=scope, count=agg[0],
                    min=agg[1], max=agg[2], distinct_count=agg[3], top_values=top, updated_at=utcnow())
                await session.execute(stmt.on_conflict_do_update(
                    index_elements=["knowledge_base_id", "record_type", "field", "scope"],
                    set_={"count": stmt.excluded.count, "min": stmt.excluded.min, "max": stmt.excluded.max,
                          "distinct_count": stmt.excluded.distinct_count, "top_values": stmt.excluded.top_values,
                          "updated_at": stmt.excluded.updated_at}))


class _RecordErrors(Invalid):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors), {"errors": errors})
        self.errors = errors


def _counts(outcomes: list[dict]) -> dict:
    counts = {"created": 0, "updated": 0, "unchanged": 0, "invalid": 0}
    for o in outcomes:
        counts[o["outcome"]] += 1
    return counts


def merge_stats(rows) -> list[dict]:
    """Rows of several KBs → one entry per (field, scope)."""
    merged: dict[tuple, dict] = {}
    for r in rows:
        key = (r.field, r.scope)
        m = merged.setdefault(key, {"field": r.field, "scope": r.scope, "count": 0, "min": None, "max": None,
                                    "distinct_count": 0, "top_values": {}})
        m["count"] += r.count
        if r.min is not None:
            m["min"] = r.min if m["min"] is None else min(m["min"], r.min)
        if r.max is not None:
            m["max"] = r.max if m["max"] is None else max(m["max"], r.max)
        m["distinct_count"] = max(m["distinct_count"], r.distinct_count)
        for tv in r.top_values or []:
            key2 = json.dumps(tv["value"], sort_keys=True, default=str)
            m["top_values"][key2] = m["top_values"].get(key2, 0) + tv["count"]
    out = []
    for m in merged.values():
        tops = sorted(m["top_values"].items(), key=lambda kv: -kv[1])[:20]
        m["top_values"] = [{"value": json.loads(k), "count": n} for k, n in tops]
        out.append(m)
    return sorted(out, key=lambda m: (m["scope"], m["field"]))


# --- files -------------------------------------------------------------------------------------------------
def parse_file(name: str, data: bytes, td: TypeDef | None, column_map: dict) -> list[dict]:
    lower = name.lower()
    if lower.endswith(".jsonl"):
        out = []
        for n, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    raise Invalid(f"line {n} is not JSON")
        return out
    if lower.endswith(".json"):
        try:
            value = json.loads(data.decode("utf-8-sig"))
        except ValueError:
            raise Invalid("the file is not JSON")
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
        return value if isinstance(value, list) else [value]
    if lower.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    elif lower.endswith(".xlsx"):
        try:
            import openpyxl
        except ImportError:
            raise Invalid("XLSX needs openpyxl (the parsers extra)") from None
        sheet = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True).worksheets[0]
        rows = [["" if v is None else (v if not isinstance(v, str) else v) for v in row]
                for row in sheet.iter_rows(values_only=True)]
    else:
        raise Invalid("records files are .json, .jsonl, .csv or .xlsx")
    rows = [r for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        return []
    header = [str(h).strip() for h in rows[0]]
    out = []
    for row in rows[1:]:
        record: dict = {}
        for column, value in zip(header, row):
            if not column:
                continue
            path = column_map.get(column, column)
            if not path:
                continue
            info = td.fields.get(("record", td.canonical(path))) if td else None
            converted = coerce(value if isinstance(value, str) else ("" if value is None else value), info) \
                if isinstance(value, str) or value is None else value
            if converted is not None:
                set_path(record, path, converted)
        out.append(record)
    return out
