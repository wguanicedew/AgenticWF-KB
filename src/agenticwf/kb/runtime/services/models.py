"""
ModelService: system models from kb.cfg, users' models, and resolving a model to a client
(docs/configuration.md, System models; docs/data-model.md, kb_models).

System models are `[models.<kind>.<name>]` sections, synced at start into rows owned by `system`
with visibility `all`, addressed as `system:<name>`. Users add their own models through the API;
their API keys are stored encrypted with the Fernet `secrets_key` and never returned.
"""

from __future__ import annotations

import uuid
from typing import Any

import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.config import config_get, config_has_option, get_config
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import MODEL_TYPES, KBModel, utcnow
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.errors import Forbidden, Invalid, NotFound
from agenticwf.kb.runtime.models.clients import ChatModel, Embedder, ModelError, ModelSpec, Reranker
from agenticwf.kb.runtime.security.secrets import SecretBox

LOG = get_logger(__name__)

KINDS = {"chat": "KnowledgeQA", "embedding": "Embedding", "rerank": "Rerank", "vlm": "VLLM"}
DEFAULT_OPTIONS = {"Embedding": "default_embedding", "KnowledgeQA": "default_chat", "Rerank": "default_rerank",
                   "VLLM": "default_vlm"}
PROVIDERS = ("openai_compat", "hash")
MASK = "******"
SYSTEM_NAMESPACE = uuid.UUID("6f1e0c39-6a0f-4c9a-9a51-6d1b7b0b1a01")


def system_model_id(name: str) -> str:
    return str(uuid.uuid5(SYSTEM_NAMESPACE, name))


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def system_sections() -> dict[str, tuple[str, dict[str, str]]]:
    """name -> (type, options) of every [models.<kind>.<name>] section."""
    found = {}
    config = get_config()
    for section in config.sections():
        parts = section.split(".", 2)
        if len(parts) == 3 and parts[0] == "models" and parts[1] in KINDS:
            found[parts[2]] = (KINDS[parts[1]], dict(config.items(section)))
    return found


def _parameters_from_section(options: dict[str, str]) -> dict[str, Any]:
    params: dict[str, Any] = {"provider": options.get("provider", "openai_compat"),
                              "base_url": options.get("base_url", "")}
    if options.get("api_key_env"):
        params["api_key_env"] = options["api_key_env"]
    for key in ("dimension", "max_concurrency", "context_window", "max_output_tokens", "truncate_prompt_tokens"):
        if options.get(key):
            params[key] = int(options[key])
    for key in ("supports_tools", "supports_vision"):
        if options.get(key):
            params[key] = _bool(options[key])
    if options.get("custom_headers"):
        params["custom_headers"] = dict(line.split(":", 1) for line in options["custom_headers"].splitlines()
                                        if ":" in line)
        params["custom_headers"] = {k.strip(): v.strip() for k, v in params["custom_headers"].items()}
    return params


def model_dict(row: KBModel) -> dict:
    params = dict(row.parameters or {})
    if params.get("api_key"):
        params["api_key"] = MASK
    return {"id": row.id, "name": row.name, "display_name": row.display_name, "description": row.description,
            "type": row.type, "source": row.source, "parameters": params, "is_default": row.is_default,
            "is_builtin": row.is_builtin, "status": row.status, "owner_subject": row.owner_subject,
            "visibility": row.visibility, "groups": list(row.groups or []),
            "ref": f"system:{row.display_name}" if row.is_builtin else row.id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}


class ModelService:
    def __init__(self, secrets: SecretBox | None = None):
        self._secrets = secrets

    @property
    def secrets(self) -> SecretBox | None:
        if self._secrets is None:
            self._secrets = SecretBox.from_environment()
        return self._secrets

    # -- system models ---------------------------------------------------------------------------
    async def sync_system_models(self) -> list[str]:
        sections = system_sections()
        defaults = {t: (config_get("models", opt).strip() if config_has_option("models", opt) else "")
                    for t, opt in DEFAULT_OPTIONS.items()}
        synced = []
        async with async_session_scope() as session:
            existing = {r.id: r for r in (await session.execute(
                sa.select(KBModel).where(KBModel.is_builtin.is_(True)))).scalars()}
            for name, (mtype, options) in sections.items():
                model_id = system_model_id(name)
                row = existing.pop(model_id, None)
                if row is None:
                    row = KBModel(id=model_id, owner_subject=access.SYSTEM_OWNER, visibility="all", is_builtin=True,
                                  type=mtype, name="", display_name=name)
                    session.add(row)
                row.type, row.name = mtype, options.get("model", name)
                row.display_name = name
                row.description = options.get("description", "")
                row.parameters = _parameters_from_section(options)
                row.status, row.deleted_at = "active", None
                row.is_default = defaults.get(mtype) == name
                synced.append(name)
            for row in existing.values():     # a section that disappeared: KBs may still use it
                row.status = "failed"
                row.is_default = False
        if synced:
            LOG.info("system models: %s", ", ".join(sorted(synced)))
        return synced

    # -- lookups -----------------------------------------------------------------------------------
    @staticmethod
    def _id_of(ref: str) -> str:
        return system_model_id(ref.split(":", 1)[1]) if ref.startswith("system:") else ref

    async def _row(self, session, ref: str) -> KBModel | None:
        row = await session.get(KBModel, self._id_of(ref))
        return row if row is not None and row.deleted_at is None else None

    async def default_id(self, mtype: str) -> str | None:
        option = DEFAULT_OPTIONS.get(mtype)
        if not option or not config_has_option("models", option):
            return None
        name = config_get("models", option).strip()
        return (self._id_of(name) if name.startswith("system:") else system_model_id(name)) if name else None

    async def spec(self, ref: str) -> ModelSpec:
        """The client spec of a model (no access check: callers pass ids from rows they may use)."""
        async with async_session_scope(commit=False) as session:
            row = await self._row(session, ref)
        if row is None or row.status != "active":
            raise ModelError(f"model {ref} is not available")
        params = row.parameters or {}
        api_key = ""
        if params.get("api_key"):
            if self.secrets is None:
                raise ModelError(f"model {row.display_name or row.name}: no secrets key to decrypt its API key")
            api_key = self.secrets.decrypt(params["api_key"])
        elif params.get("api_key_env"):
            import os
            api_key = os.environ.get(params["api_key_env"], "")
        return ModelSpec(id=row.id, name=row.name, type=row.type, provider=params.get("provider", "openai_compat"),
                         base_url=params.get("base_url", ""), api_key=api_key,
                         dimension=int(params.get("dimension") or (params.get("embedding_parameters") or {}).get(
                             "dimension") or 0),
                         max_concurrency=int(params.get("max_concurrency") or 0),
                         context_window=int(params.get("context_window") or 0),
                         max_output_tokens=int(params.get("max_output_tokens") or 0),
                         supports_tools=bool(params.get("supports_tools")),
                         supports_vision=bool(params.get("supports_vision")),
                         headers=dict(params.get("custom_headers") or {}),
                         truncate_prompt_tokens=int(params.get("truncate_prompt_tokens") or 0))

    async def embedder(self, ref: str) -> Embedder:
        return Embedder(await self.spec(ref))

    async def reranker(self, ref: str) -> Reranker:
        return Reranker(await self.spec(ref))

    async def chat(self, ref: str) -> ChatModel:
        return ChatModel(await self.spec(ref))

    # -- the API ---------------------------------------------------------------------------------
    async def list(self, ctx: AuthContext, mtype: str | None = None) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            query = sa.select(KBModel).where(KBModel.deleted_at.is_(None), access.visible_clause(KBModel, ctx))
            if mtype:
                query = query.where(KBModel.type == mtype)
            rows = (await session.execute(query.order_by(KBModel.is_builtin.desc(), KBModel.created_at))).scalars()
            return [model_dict(r) for r in rows]

    async def get(self, ctx: AuthContext, ref: str) -> dict:
        async with async_session_scope(commit=False) as session:
            row = await self._row(session, ref)
            if row is None or not access.can_see(ctx, row):
                raise NotFound("model not found")
            return model_dict(row)

    async def usable(self, ctx: AuthContext, ref: str, mtype: str) -> str:
        """The id of a model the principal may use for `mtype`, or Invalid."""
        async with async_session_scope(commit=False) as session:
            row = await self._row(session, ref)
        if row is None or not access.can_see(ctx, row):
            raise Invalid(f"model {ref} not found")
        if row.type != mtype:
            raise Invalid(f"model {ref} is a {row.type} model, not {mtype}")
        return row.id

    def _parameters(self, body: dict, old: dict | None = None) -> dict:
        params = dict(old or {})
        incoming = dict(body.get("parameters") or {})
        key = incoming.pop("api_key", None)
        params.update(incoming)
        if params.get("provider", "openai_compat") not in PROVIDERS:
            raise Invalid(f"provider must be one of {', '.join(PROVIDERS)}")
        if key and key != MASK:        # MASK echoed back = keep the stored key
            if self.secrets is None:
                raise Invalid("this KB has no secrets key: use api_key_env instead of storing a key")
            params["api_key"] = self.secrets.encrypt(key)
        elif key == "":
            params.pop("api_key", None)
        return params

    async def create(self, ctx: AuthContext, body: dict) -> dict:
        mtype = body.get("type")
        if mtype not in MODEL_TYPES:
            raise Invalid(f"type must be one of {', '.join(MODEL_TYPES)}")
        if not (body.get("name") or "").strip():
            raise Invalid("name is required")
        params = self._parameters(body)
        if mtype == "Embedding" and not int(params.get("dimension") or 0):
            raise Invalid("an embedding model needs parameters.dimension")
        visibility = body.get("visibility", "private")
        groups = access.check_share(ctx, visibility, body.get("groups") or [])
        if visibility == "all" and not access.is_admin(ctx):
            raise Forbidden("only an admin shares a model with everyone")
        async with async_session_scope() as session:
            row = KBModel(owner_subject=ctx.subject, owner_org=ctx.organization_id, visibility=visibility,
                          groups=groups, name=body["name"].strip(), display_name=body.get("display_name", ""),
                          description=body.get("description", ""), type=mtype,
                          source=body.get("source", "remote"), parameters=params)
            session.add(row)
            await session.flush()
            return model_dict(row)

    async def update(self, ctx: AuthContext, ref: str, body: dict) -> dict:
        async with async_session_scope() as session:
            row = await self._row(session, ref)
            if row is None or not access.can_see(ctx, row):
                raise NotFound("model not found")
            if row.is_builtin:
                raise Forbidden("system models are defined in kb.cfg")
            if not access.can_manage(ctx, row):
                raise Forbidden("only the owner changes a model")
            for field in ("name", "display_name", "description", "source"):
                if field in body and body[field] is not None:
                    setattr(row, field, body[field])
            if "parameters" in body:
                row.parameters = self._parameters(body, row.parameters)
            if "visibility" in body or "groups" in body:
                visibility = body.get("visibility", row.visibility)
                row.groups = access.check_share(ctx, visibility, body.get("groups", row.groups) or [])
                row.visibility = visibility
            row.updated_at = utcnow()
            return model_dict(row)

    async def delete(self, ctx: AuthContext, ref: str) -> None:
        async with async_session_scope() as session:
            row = await self._row(session, ref)
            if row is None or not access.can_see(ctx, row):
                raise NotFound("model not found")
            if row.is_builtin:
                raise Forbidden("system models are defined in kb.cfg")
            if not access.can_manage(ctx, row):
                raise Forbidden("only the owner deletes a model")
            row.deleted_at = utcnow()

    async def debug(self, ctx: AuthContext, ref: str) -> dict:
        """One small real call, to check the model answers."""
        model = await self.get(ctx, ref)
        try:
            if model["type"] == "Embedding":
                vectors = await (await self.embedder(model["id"])).embed(["hello world"])
                return {"ok": True, "dimension": len(vectors[0])}
            if model["type"] == "Rerank":
                scores = await (await self.reranker(model["id"])).rerank("hello", ["hello world", "goodbye"])
                return {"ok": True, "scores": scores}
            reply = await (await self.chat(model["id"])).complete(
                [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5)
            return {"ok": True, "reply": reply["content"][:100]}
        except ModelError as exc:
            return {"ok": False, "error": str(exc)}
