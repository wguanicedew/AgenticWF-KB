"""
The `hpo` plugin (docs/hpo-warm-start.md): the system type `hpo_job` and the operations
`suggest` (read-only; the agentic-search tool `suggest_initial_points`) and `alias_suggest`.

Every number in a suggestion comes from the deterministic algorithm (algorithm.py); the one
LLM-assisted operation proposes parameter names and changes nothing.
"""

from __future__ import annotations

import json
import math

import sqlalchemy as sa

from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.ext import Record, RecordItem
from kb_core.orm.models.kb import SOURCE_RECORD, KnowledgeBase
from kb_runtime import access
from kb_runtime.errors import Forbidden, Invalid
from kb_runtime.records.plugins import Operation, RecordTypePlugin
from kb_runtime.records.plugins.hpo import algorithm as alg

TYPE_NAME = "hpo_job"
QUALIFIED = f"{access.SYSTEM_OWNER}:{TYPE_NAME}"
MAX_CANDIDATES = 500

PARAM_SPEC = {"type": "object", "required": ["type"], "properties": {
    "type": {"type": "string", "enum": ["float", "int", "categorical"]},
    "low": {"type": "number"}, "high": {"type": "number"}, "log": {"type": "boolean"}, "step": {"type": "number"},
    "choices": {"type": "array", "items": {"type": ["number", "string", "boolean"]}},
    "default": {}}}

HPO_JOB = {
    "name": TYPE_NAME,
    "title": "Hyperparameter optimisation job",
    "description": "One HPO study: its search space, objective and trials (docs/hpo-warm-start.md)",
    "schema": {
        "type": "object",
        "required": ["job_id", "objective", "search_space", "trials"],
        "properties": {
            "job_id": {"type": "string", "minLength": 1},
            "record_name": {"type": "string"},
            "description": {"type": "string"},
            "finished_at": {"type": "string", "format": "date-time"},
            "objective": {"type": "object", "required": ["name", "direction"], "properties": {
                "name": {"type": "string"}, "direction": {"type": "string", "enum": ["minimize", "maximize"]}}},
            "search_space": {"type": "object", "additionalProperties": PARAM_SPEC},
            "trials": {"type": "array", "items": {"type": "object", "properties": {
                "number": {"type": "integer"},
                "params": {"type": "object", "additionalProperties": {"type": ["number", "string", "boolean"]}},
                "value": {"type": ["number", "null"]},
                "status": {"type": "string", "enum": ["complete", "pruned", "fail", "running"]},
                "percentile": {"type": ["number", "null"]}}}},
            "metadata": {"type": "object", "properties": {
                "dataset": {"type": "string"}, "model": {"type": "string"}, "framework": {"type": "string"}},
                "additionalProperties": {"type": ["string", "number", "boolean"]}},
            "best_value": {"type": ["number", "null"]},
            "n_complete": {"type": "integer"},
            "_param_names": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    "roles": {
        "id": "job_id", "name": "record_name", "text": ["record_name", "description"], "time": "finished_at",
        "facets": ["objective.name", "objective.direction", "metadata.dataset", "metadata.model", "metadata.framework"],
        "measures": {"best_value": {}, "n_complete": {"direction": "maximize"}},
        "features": ["_param_names"],
        "items": {"path": "trials", "measures": {"value": {}, "percentile": {"direction": "maximize"}},
                  "features": ["params"]},
    },
    "aliases": {},
}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class HPOPlugin(RecordTypePlugin):
    name = "hpo"
    types = [HPO_JOB]

    def __init__(self):
        self.operations = {
            "suggest": Operation("suggest", "Warm-start points for a new HPO job from similar past jobs.", True,
                                 self.suggest, SUGGEST_PARAMETERS, tool_name="suggest_initial_points"),
            "alias_suggest": Operation("alias_suggest", "Likely aliases for unknown parameter names (LLM).", False,
                                       self.alias_suggest),
        }

    # -- hooks ------------------------------------------------------------------------------------------
    @staticmethod
    def aliases(cfg: dict) -> tuple[dict, dict]:
        hpo = (cfg or {}).get("hpo") or {}
        return alg.alias_map(hpo.get("param_aliases") or {}), alg.alias_map(hpo.get("objective_aliases") or {})

    def normalize(self, record: dict, cfg: dict) -> dict:
        params, objectives = self.aliases(cfg)
        record = dict(record)
        if isinstance(record.get("objective"), dict) and record["objective"].get("name"):
            record["objective"] = {**record["objective"],
                                   "name": alg.canonical_name(record["objective"]["name"], objectives)}
        if isinstance(record.get("search_space"), dict):
            record["search_space"] = {alg.canonical_name(k, params): v for k, v in record["search_space"].items()}
        trials = []
        for trial in record.get("trials") or []:
            if isinstance(trial, dict) and isinstance(trial.get("params"), dict):
                trial = {**trial, "params": {alg.canonical_name(k, params): v for k, v in trial["params"].items()}}
            trials.append(trial)
        if "trials" in record:
            record["trials"] = trials
        if not record.get("record_name") and record.get("job_id"):
            record["record_name"] = str(record["job_id"])
        for derived in ("best_value", "n_complete", "_param_names"):
            record.pop(derived, None)
        return record

    def validate(self, record: dict, cfg: dict) -> list[str]:
        errors = []
        space = record.get("search_space") or {}
        for name, spec in space.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("type") in ("float", "int"):
                if not (_number(spec.get("low")) and _number(spec.get("high"))):
                    errors.append(f"search_space.{name}: low and high are required")
                elif spec["low"] >= spec["high"]:
                    errors.append(f"search_space.{name}: low must be below high")
                elif spec.get("log") and spec["low"] <= 0:
                    errors.append(f"search_space.{name}: a log range needs low > 0")
            elif spec.get("type") == "categorical" and not spec.get("choices"):
                errors.append(f"search_space.{name}: choices are required")
        strict = (cfg or {}).get("strict", True)
        for i, trial in enumerate(record.get("trials") or []):
            if not isinstance(trial, dict):
                continue
            extra = [p for p in (trial.get("params") or {}) if p not in space]
            if extra and strict:
                errors.append(f"trials.{i}: parameters not in search_space: {', '.join(sorted(extra))}")
            if trial.get("status", "complete") == "complete" and not _number(trial.get("value")):
                errors.append(f"trials.{i}: a complete trial needs a numeric value")
        return errors

    def derive(self, record: dict) -> tuple[dict, list[dict]]:
        record = dict(record)
        direction = (record.get("objective") or {}).get("direction", "minimize")
        trials = [dict(t) for t in record.get("trials") or []]
        complete = [t.get("value") if t.get("status", "complete") == "complete" and _number(t.get("value")) else None
                    for t in trials]
        ranks = alg.percentiles(complete, direction)
        derived = []
        for trial, pct in zip(trials, ranks):
            trial["percentile"] = pct
            derived.append({"percentile": pct})
        record["trials"] = trials
        values = [v for v in complete if v is not None]
        record["best_value"] = (max(values) if direction == "maximize" else min(values)) if values else None
        record["n_complete"] = len(values)
        record["_param_names"] = sorted(record.get("search_space") or {})
        return record, derived

    # -- suggest -------------------------------------------------------------------------------------------
    async def suggest(self, env, body: dict) -> dict:
        space = body.get("search_space")
        objective = body.get("objective") or {}
        if not isinstance(space, dict) or not space:
            raise Invalid("search_space is required")
        if objective.get("direction") not in ("minimize", "maximize") or not objective.get("name"):
            raise Invalid("objective needs name and direction (minimize or maximize)")
        n = int(body.get("n") or 10)
        n_total = int(body.get("n_total") or n)
        if not 0 <= n <= 200 or not n <= n_total <= 1000:
            raise Invalid("n must be 0..200 and n ≤ n_total ≤ 1000")
        records = env.records
        async with async_session_scope(commit=False) as session:
            kb_ids = await records._readable_kb_ids(session, env.auth, body.get("knowledge_base_ids") or None)
            td = await records.typedef(session, env.auth, QUALIFIED, kb_ids)
            kbs = {kb.id: kb for kb in (await session.execute(sa.select(KnowledgeBase).where(
                KnowledgeBase.id.in_(kb_ids or ["-"])))).scalars()}
            # the request's space, canonicalised with the union of the KBs' aliases
            params_alias, objective_alias = {}, {}
            for kb in kbs.values():
                p, o = self.aliases(kb.records_config or {})
                params_alias.update(p)
                objective_alias.update(o)
            space = {alg.canonical_name(k, params_alias): v for k, v in space.items()}
            objective = {"name": alg.canonical_name(objective["name"], objective_alias),
                         "direction": objective["direction"]}
            self._check_space(space)
            from kb_runtime.records import query as rq

            c = rq.Compiler(td)
            where = c.where(body.get("filter"))
            params = {**c.params, "kbs": kb_ids, "types": [QUALIFIED], "with_deprecated": False,
                      "obj": objective["name"], "dir": objective["direction"], "names": sorted(space)}
            rows = (await records._run(session, (
                f"SELECT r.id, r.knowledge_base_id, r.external_id, r.name, r.data FROM kb_records r "
                f"WHERE {rq.BASE_WHERE} AND {where} AND r.facets->>'objective.name' = :obj "
                f"AND r.facets->>'objective.direction' = :dir "
                f"AND (r.measures->>'n_complete')::float8 >= 1 "
                f"AND r.features->'_param_names' ?| CAST(:names AS text[]) "
                f"ORDER BY r.event_time DESC NULLS LAST, r.id LIMIT {MAX_CANDIDATES}"), params)).mappings().all()
            ids = [r["id"] for r in rows]
            items: dict[str, list] = {}
            if ids:
                for item in (await session.execute(sa.select(RecordItem).where(RecordItem.record_id.in_(ids))
                                                   .order_by(RecordItem.record_id, RecordItem.item_index))).scalars():
                    items.setdefault(item.record_id, []).append(item)
            task = await self._task_similarity(env, session, kbs, ids, body.get("description") or "")
        jobs = []
        for row in rows:
            data = row["data"]
            trials = []
            for item in items.get(row["id"], []):
                t = item.data
                trials.append({"index": item.item_index, "params": t.get("params") or {}, "value": t.get("value"),
                               "status": t.get("status", "complete"),
                               "percentile": (item.derived or {}).get("percentile", t.get("percentile"))})
            jobs.append({"record_id": row["id"], "knowledge_base_id": row["knowledge_base_id"], "job_id": row["external_id"],
                         "record_name": row["name"], "search_space": data.get("search_space") or {},
                         "objective": data.get("objective"), "trials": trials,
                         "n_complete": data.get("n_complete") or 0,
                         "task": task.get(row["id"]) if body.get("description") else None})
        weights = {}
        for kb_id in {j["knowledge_base_id"] for j in jobs}:
            weights.update(((kbs[kb_id].records_config or {}).get("hpo") or {}).get("similarity_weights") or {})
        return alg.suggest(space, objective, jobs, n, n_total, body.get("options") or {}, weights)

    @staticmethod
    def _check_space(space: dict) -> None:
        for name, spec in space.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("float", "int", "categorical"):
                raise Invalid(f"search_space.{name}: type must be float, int or categorical")
            if spec["type"] == "categorical":
                if not spec.get("choices"):
                    raise Invalid(f"search_space.{name}: choices are required")
            elif not (_number(spec.get("low")) and _number(spec.get("high")) and spec["low"] < spec["high"]):
                raise Invalid(f"search_space.{name}: low < high are required")
            elif spec.get("log") and spec["low"] <= 0:
                raise Invalid(f"search_space.{name}: a log range needs low > 0")

    @staticmethod
    async def _task_similarity(env, session, kbs, ids, description) -> dict[str, float]:
        if not ids or not description:
            return {}
        out: dict[str, float] = {}
        models = env.services.models
        for model_id in {kb.embedding_model_id for kb in kbs.values() if kb.embedding_model_id}:
            embedder = await models.embedder(model_id)
            vector = (await embedder.embed([description]))[0]
            dim = int(embedder.dimension)
            q = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
            kb_ids = [kb.id for kb in kbs.values() if kb.embedding_model_id == model_id]
            rows = await session.execute(sa.text(
                f"SELECT source_id, 1 - (embedding::halfvec({dim}) <=> CAST(:q AS halfvec({dim}))) FROM kb_embeddings "
                f"WHERE source_type = {SOURCE_RECORD} AND dimension = {dim} AND source_id = ANY(:ids) "
                "AND knowledge_base_id = ANY(:kbs)"), {"q": q, "ids": ids, "kbs": kb_ids})
            out.update({sid: float(score) for sid, score in rows.all() if score is not None})
        return out

    # -- alias_suggest -------------------------------------------------------------------------------------
    async def alias_suggest(self, env, body: dict) -> dict:
        """Likely aliases for parameter names that match nothing in the KB; nothing changes until
        the user saves them into records_config.hpo.param_aliases."""
        from kb_runtime.prompts import render

        names = [str(n) for n in body.get("names") or [] if str(n).strip()][:50]
        if not names or not body.get("knowledge_base_id"):
            raise Invalid("give knowledge_base_id and names")
        async with async_session_scope(commit=False) as session:
            kb = await env.records._records_kb(session, env.auth, body["knowledge_base_id"])
            if not access.can_contribute(env.auth, kb):
                raise Forbidden("alias_suggest needs write access to the knowledge base")
            known = set()
            for (data,) in (await session.execute(sa.select(Record.data).where(
                    Record.knowledge_base_id == kb.id, Record.record_type == QUALIFIED,
                    Record.deleted_at.is_(None)).limit(2000))).all():
                known |= set((data or {}).get("search_space") or {})
        params_alias, _ = self.aliases(kb.records_config or {})
        unknown = [n for n in names if alg.canonical_name(n, params_alias) not in known]
        if not unknown or not known:
            return {"suggestions": [], "unknown": unknown, "known": sorted(known)}
        model_id = await env.services.models.default_id("KnowledgeQA")
        if not model_id:
            raise Invalid("alias_suggest needs a chat model")
        chat = await env.services.models.chat(model_id)
        reply = await chat.complete([
            {"role": "system", "content": render("records", "alias_system")},
            {"role": "user", "content": render("records", "alias_user", unknown=json.dumps(unknown),
                                               known=json.dumps(sorted(known)))}], max_tokens=600, temperature=0.0)
        from kb_runtime.agentic.engine import _json_object

        try:
            proposed = _json_object(reply["content"]).get("aliases") or []
        except ValueError:
            proposed = []
        suggestions = []
        for p in proposed if isinstance(proposed, list) else []:
            if isinstance(p, dict) and p.get("name") in unknown and p.get("alias_of") in known:
                confidence = p.get("confidence")
                suggestions.append({"name": p["name"], "alias_of": p["alias_of"],
                                    "confidence": float(confidence) if _number(confidence) else None})
        return {"suggestions": suggestions, "unknown": unknown, "known": sorted(known)}


SUGGEST_PARAMETERS = {"type": "object", "properties": {
    "knowledge_base_ids": {"type": "array", "items": {"type": "string"}},
    "objective": {"type": "object", "properties": {"name": {"type": "string"},
                                                    "direction": {"type": "string", "enum": ["minimize", "maximize"]}}},
    "description": {"type": "string"},
    "search_space": {"type": "object", "description": "name → {type: float|int|categorical, low, high, log, step, choices}"},
    "n": {"type": "integer", "minimum": 0, "maximum": 50}, "n_total": {"type": "integer"},
    "options": {"type": "object"}}, "required": ["objective", "search_space"]}
