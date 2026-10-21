"""
WebSearchService (docs/web-search.md): providers, search and fetch, behind `[features] web`.

Providers come from kb.cfg (`[websearch.<name>]`, owned by `system`, usable by everyone) or from
users (`/web-search-providers`, visible to their owner only; API keys stored encrypted). A search
runs the chosen provider (or the caller's default, else the deployment's), de-duplicates the
results, and fetches the text of the top `fetch_top` pages that came without it, through the
SSRF-safe fetcher and the ingestion parsers.

    [websearch]
    default = ddg                  # the provider used when a request names none
    max_results = 5
    fetch_top = 3                  # pages whose text is fetched with the results
    fetch_chars = 4000             # characters of each fetched page returned with the results

    [websearch.ddg]
    kind = duckduckgo
    [websearch.brave]
    kind = brave
    api_key_env = BRAVE_API_KEY
"""

from __future__ import annotations

import asyncio
import uuid

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.common.config import config_get, config_has_option, get_config
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import async_session_scope
from kb_core.orm.models.ext import WebSearchProvider
from kb_core.orm.models.kb import utcnow
from kb_runtime import access, features
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound
from kb_runtime.security.secrets import SecretBox, SecretsUnavailable, hint
from kb_runtime.security.ssrf import FetchError, resolve_checked
from kb_runtime.websearch.fetch import FetchCache, PageError, window
from kb_runtime.websearch.providers import KINDS, Provider, WebResult, WebSearchError, dedupe

LOG = get_logger(__name__)
NAMESPACE = uuid.UUID("0d7b8e1a-5c2f-4f39-9f5e-7d5a1c3b2e10")
SYSTEM = access.SYSTEM_OWNER
SECRET_PARAMS = ("api_key",)
ALLOWED_PARAMS = ("base_url", "api_key", "cx", "region", "search_depth")
MASK = "******"


def setting(name: str, default):
    if not config_has_option("websearch", name):
        return default
    raw = config_get("websearch", name).strip()
    return type(default)(raw) if not isinstance(default, str) else raw


def system_sections() -> dict[str, dict[str, str]]:
    config = get_config()
    return {s.split(".", 1)[1]: dict(config.items(s)) for s in config.sections()
            if s.startswith("websearch.") and s.count(".") == 1}


def provider_dict(row: WebSearchProvider, secrets: SecretBox | None = None) -> dict:
    params = {k: v for k, v in (row.parameters or {}).items() if k != "api_key_env"}
    if params.get("api_key"):
        try:
            params["api_key"] = MASK + (hint(secrets.decrypt(params["api_key"])) if secrets else "")
        except SecretsUnavailable:
            params["api_key"] = MASK
    return {"id": row.id, "name": row.name, "kind": row.kind, "title": KINDS[row.kind].title if row.kind in KINDS else row.kind,
            "parameters": params, "is_default": row.is_default, "is_builtin": row.is_builtin, "status": row.status,
            "owner_subject": row.owner_subject, "ref": f"system:{row.name}" if row.is_builtin else row.id,
            "created_at": row.created_at.isoformat() if row.created_at else None}


class WebSearchService:
    CACHE_KEYS = 64
    CACHE_TTL = 900.0           # seconds a caller's page cache (cache_key) lives after its last use

    def __init__(self, secrets: SecretBox | None = None, transport=None):
        self._secrets = secrets
        self.transport = transport             # tests mock the providers' HTTP
        self._caches: dict[tuple[str, str], tuple[float, FetchCache]] = {}

    def cache_for(self, ctx: AuthContext, key: str | None) -> FetchCache:
        """A caller's page cache across requests (the agent passes its task id), so paged reads
        through /web-fetch see one version of a page; per principal, expiring after CACHE_TTL."""
        import time

        if not key:
            return FetchCache()
        now = time.monotonic()
        self._caches = {k: v for k, v in self._caches.items() if now - v[0] < self.CACHE_TTL}
        entry = self._caches.get((ctx.subject, str(key)[:128]))
        cache = entry[1] if entry else FetchCache()
        self._caches[(ctx.subject, str(key)[:128])] = (now, cache)
        while len(self._caches) > self.CACHE_KEYS:
            self._caches.pop(min(self._caches, key=lambda k: self._caches[k][0]))
        return cache

    @property
    def secrets(self) -> SecretBox | None:
        if self._secrets is None:
            self._secrets = SecretBox.from_environment()
        return self._secrets

    @staticmethod
    def _check():
        if not features.enabled("web"):
            raise NotFound("web search is switched off in this deployment ([features] web)")

    # -- providers ---------------------------------------------------------------------------------------
    async def sync_system_providers(self) -> list[str]:
        sections = system_sections()
        default = setting("default", "")
        names = []
        async with async_session_scope() as session:
            existing = {r.id: r for r in (await session.execute(sa.select(WebSearchProvider).where(
                WebSearchProvider.is_builtin.is_(True)))).scalars()}
            for name, options in sections.items():
                kind = options.get("kind", "")
                if kind not in KINDS:
                    LOG.warning("[websearch.%s]: unknown kind %r", name, kind)
                    continue
                provider_id = str(uuid.uuid5(NAMESPACE, name))
                row = existing.pop(provider_id, None)
                if row is None:
                    row = WebSearchProvider(id=provider_id, owner_subject=SYSTEM, is_builtin=True, name=name, kind=kind)
                    session.add(row)
                row.kind = kind
                row.parameters = {k: v for k, v in options.items() if k in ALLOWED_PARAMS + ("api_key_env",)
                                  and k != "api_key"}
                row.status, row.deleted_at, row.is_default = "active", None, name == default
                names.append(name)
            for row in existing.values():
                row.status, row.is_default = "failed", False
        return names

    def _readable(self, ctx: AuthContext):
        return sa.and_(WebSearchProvider.deleted_at.is_(None),
                       sa.or_(WebSearchProvider.owner_subject == SYSTEM, WebSearchProvider.owner_subject == ctx.subject))

    async def _row(self, session, ctx, ref: str, write: bool = False) -> WebSearchProvider:
        if ref.startswith("system:"):
            ref = str(uuid.uuid5(NAMESPACE, ref.split(":", 1)[1]))
        row = await session.get(WebSearchProvider, ref)
        if row is None or row.deleted_at is not None or row.owner_subject not in (SYSTEM, ctx.subject):
            raise NotFound("web search provider not found")
        if write and (row.is_builtin or row.owner_subject != ctx.subject) and not access.is_admin(ctx):
            raise Forbidden("operators' providers change in kb.cfg")
        return row

    def kinds(self) -> list[dict]:
        return [{"kind": k.name, "title": k.title, "needs_api_key": k.needs_key, "required": list(k.params),
                 "default_base_url": k.default_url, "returns_content": k.returns_content} for k in KINDS.values()]

    async def list(self, ctx: AuthContext) -> list[dict]:
        self._check()
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(WebSearchProvider).where(self._readable(ctx)).order_by(
                WebSearchProvider.is_builtin.desc(), WebSearchProvider.name))).scalars().all()
        return [provider_dict(r, self.secrets) for r in rows]

    async def get(self, ctx, ref) -> dict:
        self._check()
        async with async_session_scope(commit=False) as session:
            return provider_dict(await self._row(session, ctx, ref), self.secrets)

    def _parameters(self, kind: str, incoming: dict, old: dict | None = None) -> dict:
        if kind not in KINDS:
            raise Invalid(f"kind must be one of {', '.join(KINDS)}")
        unknown = set(incoming) - set(ALLOWED_PARAMS)
        if unknown:
            raise Invalid(f"unknown parameters: {', '.join(sorted(unknown))}")
        params = dict(old or {})
        for key, value in incoming.items():
            if key == "api_key":
                if value in (MASK, None) or str(value).startswith(MASK):
                    continue              # the masked value echoed back: keep the stored key
                if value == "":
                    params.pop("api_key", None)
                    continue
                if self.secrets is None:
                    raise Invalid("this KB has no secrets key (KB_SECRETS_KEY_FILE): API keys cannot be stored")
                params["api_key"] = self.secrets.encrypt(str(value))
            elif value in (None, ""):
                params.pop(key, None)
            else:
                params[key] = str(value).strip()
        spec = KINDS[kind]
        missing = [p for p in spec.params if not params.get(p)]
        if missing:
            raise Invalid(f"{kind} needs {', '.join(missing)}")
        if spec.needs_key and not params.get("api_key"):
            raise Invalid(f"{kind} needs an api_key")
        return params

    async def create(self, ctx: AuthContext, body: dict) -> dict:
        self._check()
        name = str(body.get("name") or "").strip()[:128]
        if not name:
            raise Invalid("name is required")
        params = self._parameters(body.get("kind", ""), dict(body.get("parameters") or {}))
        async with async_session_scope() as session:
            if (await session.execute(sa.select(WebSearchProvider.id).where(
                    WebSearchProvider.owner_subject == ctx.subject, WebSearchProvider.name == name,
                    WebSearchProvider.deleted_at.is_(None)))).first():
                raise Conflict(f"you already have a provider {name!r}")
            if body.get("is_default"):
                await self._clear_default(session, ctx)
            row = WebSearchProvider(owner_subject=ctx.subject, owner_org=ctx.organization_id, name=name,
                                    kind=body["kind"], parameters=params, is_default=bool(body.get("is_default")))
            session.add(row)
            await session.flush()
            return provider_dict(row, self.secrets)

    @staticmethod
    async def _clear_default(session, ctx):
        await session.execute(sa.update(WebSearchProvider).where(
            WebSearchProvider.owner_subject == ctx.subject).values(is_default=False))

    async def update(self, ctx, ref, body: dict) -> dict:
        self._check()
        async with async_session_scope() as session:
            row = await self._row(session, ctx, ref, write=True)
            if body.get("name"):
                row.name = str(body["name"]).strip()[:128]
            if "parameters" in body:
                row.parameters = self._parameters(row.kind, dict(body["parameters"] or {}), row.parameters)
            if "is_default" in body:
                if body["is_default"]:
                    await self._clear_default(session, ctx)
                row.is_default = bool(body["is_default"])
            row.updated_at = utcnow()
            return provider_dict(row, self.secrets)

    async def set_credentials(self, ctx, ref, body: dict) -> dict:
        return await self.update(ctx, ref, {"parameters": {"api_key": body.get("api_key", "")}})

    async def delete(self, ctx, ref) -> None:
        self._check()
        async with async_session_scope() as session:
            row = await self._row(session, ctx, ref, write=True)
            row.deleted_at, row.is_default = utcnow(), False

    def _provider(self, row: WebSearchProvider) -> Provider:
        params = dict(row.parameters or {})
        if params.get("api_key"):
            if self.secrets is None:
                raise Invalid("the stored API key cannot be read: this KB has no secrets key")
            params["api_key"] = self.secrets.decrypt(params["api_key"])
        elif params.get("api_key_env"):
            import os

            params["api_key"] = os.environ.get(params["api_key_env"], "")
        return Provider(row.kind, params, timeout=float(setting("timeout", 15.0)), transport=self.transport)

    async def _check_url(self, row_like_owner: str, params: dict) -> None:
        """A user's own base_url (SearXNG) must be a public address, like any URL users give."""
        if row_like_owner == SYSTEM or not params.get("base_url"):
            return
        from urllib.parse import urlsplit

        parts = urlsplit(params["base_url"])
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise Invalid("base_url must be an http(s) URL")
        try:
            await resolve_checked(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
        except FetchError as exc:
            raise Invalid(f"base_url: {exc}")

    async def test(self, ctx, ref: str | None = None, body: dict | None = None) -> dict:
        """Run one query through a saved provider, or an unsaved configuration."""
        self._check()
        if ref:
            async with async_session_scope(commit=False) as session:
                row = await self._row(session, ctx, ref)
                session.expunge(row)
            provider, owner = self._provider(row), row.owner_subject
        else:
            body = body or {}
            if body.get("kind") not in KINDS:
                raise Invalid(f"kind must be one of {', '.join(KINDS)}")
            provider, owner = Provider(body["kind"], dict(body.get("parameters") or {}), transport=self.transport), ctx.subject
        await self._check_url(owner, provider.params)
        try:
            results = await provider.search((body or {}).get("query") or "open source knowledge base", 3)
        except Exception as exc:          # the provider's problem, reported, not raised
            return {"ok": False, "error": str(exc)[:500]}
        return {"ok": True, "results": [r.to_dict() for r in results]}

    # -- search and fetch ----------------------------------------------------------------------------------
    async def _choose(self, ctx, ref: str | None) -> list[Provider]:
        async with async_session_scope(commit=False) as session:
            if ref:
                rows = [await self._row(session, ctx, ref)]
            else:
                rows = (await session.execute(sa.select(WebSearchProvider).where(
                    self._readable(ctx), WebSearchProvider.status == "active", WebSearchProvider.is_default.is_(True))
                    .order_by((WebSearchProvider.owner_subject == ctx.subject).desc()))).scalars().all()[:1]
            for row in rows:
                session.expunge(row)
        if not rows:
            raise Invalid("no web search provider is configured ([websearch] default, or a provider of yours)")
        providers = []
        for row in rows:
            provider = self._provider(row)
            await self._check_url(row.owner_subject, provider.params)
            providers.append(provider)
        return providers

    async def search(self, ctx: AuthContext, query: str, provider: str | None = None, max_results: int | None = None,
                     fetch_top: int | None = None, cache: FetchCache | None = None,
                     providers: list[str] | None = None) -> dict:
        """Search one provider (or several, merged), de-duplicate, and fetch the text of the top
        `fetch_top` pages that came without it."""
        self._check()
        query = (query or "").strip()
        if not query:
            raise Invalid("query is required")
        n = max(1, min(int(max_results or setting("max_results", 5)), 20))
        top = max(0, min(int(setting("fetch_top", 3) if fetch_top is None else fetch_top), 5))
        refs = providers or ([provider] if provider else [None])     # None: the default provider
        chosen = []
        for ref in refs:
            chosen += await self._choose(ctx, ref)
        outcomes = await asyncio.gather(*(p.search(query, n) for p in chosen), return_exceptions=True)
        results: list[WebResult] = []
        errors = []
        for p, outcome in zip(chosen, outcomes):
            if isinstance(outcome, BaseException):
                errors.append(f"{p.kind}: {outcome}")
            else:
                results += outcome
        if not results and errors:
            raise WebSearchError("; ".join(errors))
        results = dedupe(results)[:n]
        cache = cache or FetchCache()
        chars = int(setting("fetch_chars", 4000))

        async def grab(r: WebResult):
            try:
                page = await cache.get(r.url)
                r.content = page.markdown[:chars]
                r.title = r.title or page.title
            except PageError as exc:
                LOG.info("web fetch of %s failed: %s", r.url, exc)

        await asyncio.gather(*(grab(r) for r in [r for r in results if not r.content][:top]))
        return {"query": query, "providers": [p.kind for p in chosen], "results": [r.to_dict() for r in results],
                "errors": errors}

    async def fetch(self, ctx: AuthContext, url: str, offset: int = 0, limit: int = 8000,
                    cache: FetchCache | None = None) -> dict:
        self._check()
        try:
            page = await (cache or FetchCache()).get(str(url or "").strip())
        except PageError as exc:
            raise Invalid(str(exc))
        return {"url": page.url, "title": page.title, "content_type": page.content_type, **window(page.markdown, offset, limit)}
