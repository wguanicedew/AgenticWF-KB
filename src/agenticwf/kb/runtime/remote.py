"""
Remote knowledge bases (docs/retrieval.md, Remote knowledge bases): the knowledge bases of
another KB server (an AgenticWF deployment's system and group KBs, a colleague's standalone KB)
that a user also lists and searches here.

Users add their own remotes (`/remotes`, the admin page), each with the user's own credential:
a remote serves only its owner. Operators may configure remotes in kb.cfg; they act as the owner
of the configured credential, so they serve only the standalone user (`local:<KB_LOCAL_USER>`):

    [remote.central]
    url = https://host.example.org/app/kb      # the remote KB's base path (its API is <url>/api/v1)
    token_file = ~/.config/agenticwf/kb-central-token   # a bearer token, or:
    pat_file = ~/.config/agenticwf/token        # an AgenticWF personal access token, exchanged at Logto
    auth_url = https://auth.example.org
    client_id = <Logto native app id>
    resource = https://api.example.org
    knowledge_bases = system:eic-docs           # empty: every KB the remote user can read

A remote, and each of its KBs, can be switched off by its user: off, it is left out of listings
and searches. Remote objects keep the remote's ids behind a prefix, `remote:<name>:<id>`, for
KBs, items and chunks; a ref is `remote:<name>:<owner>:<slug>`, and `remote:<name>` is the
remote's whole enabled set. Remote KBs are read-only here. Results are untrusted data.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.config import get_config
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.ext import RemoteSource
from agenticwf.kb.core.orm.models.kb import utcnow
from agenticwf.kb.runtime import access, features
from agenticwf.kb.runtime.errors import Conflict, Forbidden, Invalid, NotFound, RuntimeError_
from agenticwf.kb.runtime.security.secrets import SecretBox, SecretsUnavailable, hint

LOG = get_logger(__name__)

PREFIX = "remote:"
TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
PAT_TOKEN_TYPE = "urn:logto:token-type:personal_access_token"
EXPIRY_MARGIN = 60
LIST_TTL = 60
NAMESPACE = uuid.UUID("5b0f3c62-8a1e-4d7b-9c55-2e8f6a4d1b93")
SYSTEM = access.SYSTEM_OWNER
MASK = "******"
AUTH_KEYS = ("auth_url", "client_id", "resource", "scopes")
SECRET_KEYS = ("token", "pat")
DEFAULT_SCOPES = "app:access knowledge:read"


class RemoteError(RuntimeError_):
    """The remote is unreachable or refused the call."""
    code = 1000


def is_remote(ref: str | None) -> bool:
    return bool(ref) and str(ref).startswith(PREFIX)


def split(ref: str) -> tuple[str, str]:
    """(remote name, the remote's own id or ref) of `remote:<name>[:<rest>]`."""
    name, _, rest = ref[len(PREFIX):].partition(":")
    return name, rest


def _read(path: str) -> str:
    return Path(path).expanduser().read_text().strip() if path else ""


@dataclass
class Remote:
    """The client of one remote KB server."""

    name: str
    url: str
    token: str = field(default="", repr=False)
    token_file: str = ""
    pat: str = field(default="", repr=False)
    pat_file: str = ""
    auth_url: str = ""
    client_id: str = ""
    resource: str = ""
    scopes: str = DEFAULT_SCOPES
    knowledge_bases: list[str] = field(default_factory=list)
    timeout: float = 10.0
    check_addresses: bool = False          # users' remotes outside standalone mode: public addresses only
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False)   # tests mock HTTP
    _issued: dict = field(default_factory=dict, repr=False)
    _listed: tuple[float, list[dict]] | None = field(default=None, repr=False)
    last_error: str = ""

    @classmethod
    def from_section(cls, name: str, values: dict[str, str]) -> "Remote":
        if not values.get("url"):
            raise Invalid(f"[remote.{name}] needs url")
        if not values.get("token_file") and not values.get("pat_file"):
            raise Invalid(f"[remote.{name}] needs token_file or pat_file")
        if values.get("pat_file") and not all(values.get(k) for k in ("auth_url", "client_id", "resource")):
            raise Invalid(f"[remote.{name}] pat_file needs auth_url, client_id and resource")
        return cls(name=name, url=values["url"].rstrip("/"), token_file=values.get("token_file", ""),
                   pat_file=values.get("pat_file", ""), auth_url=values.get("auth_url", "").rstrip("/"),
                   client_id=values.get("client_id", ""), resource=values.get("resource", ""),
                   scopes=values.get("scopes") or DEFAULT_SCOPES, knowledge_bases=values.get("knowledge_bases", "").split(),
                   timeout=float(values.get("timeout") or 10))

    def wrap(self, remote_id: str) -> str:
        return f"{PREFIX}{self.name}:{remote_id}" if remote_id else ""

    # -- credentials --------------------------------------------------------------------------------
    async def _token(self, client: httpx.AsyncClient) -> str:
        """The bearer token: a stored token or a file's content (read each time: rotation), or an
        access token exchanged for the personal access token, kept until it nearly expires."""
        if self.token or self.token_file:
            return self.token or _read(self.token_file)
        if self._issued and self._issued["expires_at"] - time.time() > EXPIRY_MARGIN:
            return self._issued["access_token"]
        await self._check(self.auth_url)
        discovery = await client.get(f"{self.auth_url}/oidc/.well-known/openid-configuration")
        if discovery.status_code != 200:
            raise RemoteError(f"remote {self.name}: OIDC discovery failed (HTTP {discovery.status_code})")
        endpoint = discovery.json()["token_endpoint"]
        await self._check(endpoint)
        answer = await client.post(endpoint, data={
            "grant_type": TOKEN_EXCHANGE_GRANT, "subject_token": self.pat or _read(self.pat_file),
            "subject_token_type": PAT_TOKEN_TYPE, "client_id": self.client_id, "resource": self.resource,
            "scope": self.scopes})
        body = answer.json() if answer.headers.get("content-type", "").startswith("application/json") else {}
        if answer.status_code != 200 or "access_token" not in body:
            raise RemoteError(f"remote {self.name}: personal access token rejected "
                              f"({body.get('error') or answer.status_code}; expired or revoked?)")
        self._issued = {"access_token": body["access_token"],
                        "expires_at": time.time() + int(body.get("expires_in", 3600))}
        return self._issued["access_token"]

    async def _check(self, url: str) -> None:
        if not self.check_addresses:
            return
        from agenticwf.kb.runtime.security.ssrf import FetchError, resolve_checked

        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise RemoteError(f"remote {self.name}: {url} is not an http(s) URL")
        try:
            await resolve_checked(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
        except FetchError as exc:
            raise RemoteError(f"remote {self.name}: {exc}") from exc

    # -- calls --------------------------------------------------------------------------------------
    async def call(self, method: str, path: str, **kwargs):
        """(`data`, the whole body) of the remote's `{success, data}` answer; RemoteError otherwise."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                await self._check(self.url)
                token = await self._token(client)
                response = await client.request(method, f"{self.url}/api/v1{path}",
                                                headers={"Authorization": f"Bearer {token}"}, **kwargs)
        except (httpx.HTTPError, OSError) as exc:
            raise RemoteError(f"remote {self.name} unreachable: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 404 or (isinstance(body, dict) and (body.get("error") or {}).get("code") == 1003):
            raise NotFound("not found")
        if response.status_code >= 400 or not isinstance(body, dict) or body.get("success") is False:
            message = (body or {}).get("error", {}).get("message") if isinstance(body, dict) else ""
            raise RemoteError(f"remote {self.name}: HTTP {response.status_code} {message or ''}".strip())
        return body.get("data"), body

    # -- what callers see ---------------------------------------------------------------------------
    def kb(self, data: dict) -> dict:
        return {**data, "id": self.wrap(data["id"]), "ref": self.wrap(data.get("ref") or data["id"]),
                "remote": self.name, "remote_id": data["id"], "remote_url": self.url, "read_only": True}

    def hit(self, data: dict) -> dict:
        out = {**data, "remote": self.name}
        for key in ("id", "knowledge_id", "knowledge_base_id", "parent_chunk_id", "pre_chunk_id", "next_chunk_id"):
            if out.get(key):
                out[key] = self.wrap(out[key])
        out["sub_chunk_id"] = [self.wrap(c) for c in out.get("sub_chunk_id") or []]
        return out

    async def list(self, refresh: bool = False) -> list[dict]:
        """Every remote KB this remote offers (`knowledge_bases`, else all), cached for a minute."""
        if not refresh and self._listed and time.monotonic() - self._listed[0] < LIST_TTL:
            return self._listed[1]
        if self.knowledge_bases:
            kbs = []
            for ref in self.knowledge_bases:
                try:
                    kbs.append((await self.call("GET", f"/knowledge-bases/{ref}"))[0])
                except NotFound:
                    LOG.warning("remote %s: knowledge base %s not found or not readable", self.name, ref)
        else:
            kbs, page = [], 1
            while True:
                data, body = await self.call("GET", "/knowledge-bases", params={"page": page, "page_size": 100})
                kbs.extend(data or [])
                if not data or len(kbs) >= int(body.get("total") or 0):
                    break
                page += 1
        out = [self.kb(kb) for kb in kbs]
        self._listed = (time.monotonic(), out)
        return out

    async def search(self, query: str, refs: list[str] | None, body: dict) -> list[dict]:
        """`refs` are the remote's own ids or refs; None: the configured set."""
        refs = refs if refs is not None else list(self.knowledge_bases)
        payload = {**body, "query": query, **({"knowledge_base_ids": refs} if refs else {})}
        data, _ = await self.call("POST", "/knowledge-search", json=payload)
        return [self.hit(item) for item in data or []]


@dataclass
class Source:
    """A remote as one request sees it: its row's switches and its client."""

    id: str
    name: str
    is_builtin: bool
    is_enabled: bool
    disabled: frozenset[str]
    client: Remote

    async def enabled_kbs(self) -> list[dict]:
        return [kb for kb in await self.client.list() if kb["remote_id"] not in self.disabled]


class RemoteService:
    """Users' remotes and the operator's (`[remote.*]`, for the standalone user only)."""

    def __init__(self, secrets: SecretBox | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self._secrets = secrets
        self.transport = transport             # tests mock the remotes' HTTP
        self.subject: str | None = None        # the standalone user, who may use operators' remotes
        self.mode = "standalone"
        self._clients: dict[str, tuple[str, Remote]] = {}

    @property
    def secrets(self) -> SecretBox | None:
        if self._secrets is None:
            self._secrets = SecretBox.from_environment()
        return self._secrets

    def bind(self, mode: str, local_user: str) -> None:
        """Called by the API with its mode: operators' remotes serve the standalone user only, and
        users' remotes must be public addresses unless the KB is standalone."""
        self.mode = mode
        self.subject = f"local:{local_user}" if mode == "standalone" else None

    # -- operators' remotes ---------------------------------------------------------------------------
    async def sync_config(self) -> list[str]:
        """`[remote.<name>]` sections as built-in rows; a user's switches on them are kept."""
        config, names = get_config(), []
        async with async_session_scope() as session:

            existing = {r.id: r for r in (await session.execute(sa.select(RemoteSource).where(
                RemoteSource.is_builtin.is_(True)))).scalars()}
            for section in config.sections():
                if not section.startswith("remote.") or section.count(".") != 1:
                    continue
                name, values = section.split(".", 1)[1], dict(config.items(section))
                try:
                    remote = Remote.from_section(name, values)
                except Invalid as exc:
                    LOG.warning("%s; remote ignored", exc)
                    continue
                row = existing.pop(str(uuid.uuid5(NAMESPACE, name)), None)
                if row is None:
                    row = RemoteSource(id=str(uuid.uuid5(NAMESPACE, name)), owner_subject=SYSTEM, is_builtin=True,
                                       name=name, url=remote.url)
                    session.add(row)
                row.url, row.timeout, row.deleted_at = remote.url, remote.timeout, None
                row.knowledge_bases = remote.knowledge_bases
                row.auth = {k: values[k] for k in ("token_file", "pat_file", *AUTH_KEYS) if values.get(k)}
                names.append(name)
            for row in existing.values():
                row.deleted_at = row.deleted_at or utcnow()
        return names

    # -- rows and clients ----------------------------------------------------------------------------
    def _owners(self, ctx: AuthContext) -> list[str]:
        owners = [ctx.subject] if features.enabled("remote") else []
        if self.subject is not None and ctx.subject == self.subject:
            owners.append(SYSTEM)
        return owners

    async def _rows(self, ctx: AuthContext) -> list:

        owners = self._owners(ctx)
        if not owners:
            return []
        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(RemoteSource).where(
                RemoteSource.owner_subject.in_(owners), RemoteSource.deleted_at.is_(None)).order_by(
                RemoteSource.is_builtin.desc(), RemoteSource.name))).scalars().all()
            for row in rows:
                session.expunge(row)
        # a user's remote wins over an operator's of the same name
        by_name = {}
        for row in rows:
            if row.name not in by_name or not row.is_builtin:
                by_name[row.name] = row
        return list(by_name.values())

    def _client(self, row) -> Remote:
        """The row's client, kept while its connection settings stay the same (token cache, listing)."""
        auth = dict(row.auth or {})
        key = json.dumps([row.name, row.url, auth, list(row.knowledge_bases or []), row.timeout], sort_keys=True)
        cached = self._clients.get(row.id)
        if cached and cached[0] == key:
            return cached[1]
        secrets = {}
        for name in SECRET_KEYS:
            if auth.get(name):
                if self.secrets is None:
                    raise RemoteError(f"remote {row.name}: the stored credential cannot be read (no secrets key)")
                try:
                    secrets[name] = self.secrets.decrypt(auth[name])
                except SecretsUnavailable as exc:
                    raise RemoteError(f"remote {row.name}: {exc}") from exc
        client = Remote(name=row.name, url=row.url, token=secrets.get("token", ""), pat=secrets.get("pat", ""),
                        token_file=auth.get("token_file", "") if row.is_builtin else "",
                        pat_file=auth.get("pat_file", "") if row.is_builtin else "",
                        auth_url=auth.get("auth_url", ""), client_id=auth.get("client_id", ""),
                        resource=auth.get("resource", ""), scopes=auth.get("scopes") or DEFAULT_SCOPES,
                        knowledge_bases=list(row.knowledge_bases or []), timeout=row.timeout,
                        check_addresses=not row.is_builtin and self.mode != "standalone", transport=self.transport)
        self._clients[row.id] = (key, client)
        return client

    async def sources(self, ctx: AuthContext, enabled_only: bool = True) -> list[Source]:
        out = []
        for row in await self._rows(ctx):
            if enabled_only and not row.is_enabled:
                continue
            try:
                client = self._client(row)
            except RemoteError as exc:
                LOG.warning("%s", exc)
                continue
            out.append(Source(row.id, row.name, row.is_builtin, row.is_enabled, frozenset(row.disabled_kbs or ()),
                              client))
        return out

    async def _source(self, ctx: AuthContext, name: str) -> Source:
        source = next((s for s in await self.sources(ctx) if s.name == name), None)
        if source is None:
            raise NotFound("not found")
        return source

    # -- knowledge bases ----------------------------------------------------------------------------
    async def list_kbs(self, ctx: AuthContext, kb_type: str | None = None) -> list[dict]:
        """The enabled KBs of the enabled remotes, in remote order; an unreachable remote gives none."""
        out = []
        for source in await self.sources(ctx):
            try:
                kbs = await source.enabled_kbs()
                source.client.last_error = ""
            except (RemoteError, NotFound) as exc:
                source.client.last_error = str(exc)
                LOG.warning("%s", exc)
                continue
            out.extend(kb for kb in kbs if not kb_type or kb.get("type") == kb_type)
        return out

    async def get_kb(self, ctx: AuthContext, ref: str) -> dict:
        name, rest = split(ref)
        source = await self._source(ctx, name)
        if not rest:
            raise NotFound("knowledge base not found")
        kb = source.client.kb((await source.client.call("GET", f"/knowledge-bases/{rest}"))[0])
        return {**kb, "enabled": kb["remote_id"] not in source.disabled}

    # -- search -------------------------------------------------------------------------------------
    async def plan(self, ctx: AuthContext, refs: list[str] | None) -> dict[str, tuple[Remote, list[str] | None]]:
        """Which remotes a search reaches, with their own refs. Without `refs`, every enabled
        remote with its enabled KBs; with them, the remotes named by `remote:<name>[:<ref>]`."""
        if refs is not None and not any(is_remote(r) for r in refs):
            return {}
        sources = {s.name: s for s in await self.sources(ctx)}
        plan: dict[str, tuple[Remote, list[str] | None]] = {}
        if refs is None:
            for name, source in sources.items():
                if not source.disabled:
                    plan[name] = (source.client, None)
                    continue
                try:
                    enabled = [kb["remote_id"] for kb in await source.enabled_kbs()]
                except (RemoteError, NotFound) as exc:
                    source.client.last_error = str(exc)
                    LOG.warning("remote search skipped: %s", exc)
                    continue
                if enabled:
                    plan[name] = (source.client, enabled)
            return plan
        for ref in refs:
            if not is_remote(ref):
                continue
            name, rest = split(ref)
            if name not in sources:
                raise NotFound("knowledge base not found")
            if not rest:
                plan[name] = (sources[name].client, None)
            elif name not in plan or plan[name][1] is not None:
                plan.setdefault(name, (sources[name].client, []))[1].append(rest)
        return plan

    @staticmethod
    async def search(query: str, plan: dict[str, tuple[Remote, list[str] | None]], body: dict) -> list[list[dict]]:
        """One result list per remote, each in the remote's order; a failing remote gives []."""
        async def one(remote: Remote, refs):
            try:
                hits = await remote.search(query, refs, body)
                remote.last_error = ""
                return hits
            except (RemoteError, NotFound) as exc:
                remote.last_error = str(exc)
                LOG.warning("remote search skipped: %s", exc)
                return []

        return list(await asyncio.gather(*(one(remote, refs) for remote, refs in plan.values())))

    # -- items and chunks ---------------------------------------------------------------------------
    async def read(self, ctx: AuthContext, kind: str, ref: str, **params) -> dict:
        """GET a remote item (`knowledge`, `content`) or chunk (`chunk`) by its prefixed id."""
        name, rest = split(ref)
        remote = (await self._source(ctx, name)).client
        path = {"knowledge": f"/knowledge/{rest}", "content": f"/knowledge/{rest}/content",
                "chunk": f"/chunks/by-id/{rest}"}[kind]
        data, _ = await remote.call("GET", path, params=params or None)
        return remote.hit(data) if isinstance(data, dict) else data

    # -- management (/remotes) ----------------------------------------------------------------------
    def _dict(self, row) -> dict:
        auth = {k: v for k, v in (row.auth or {}).items() if k not in SECRET_KEYS}
        for name in SECRET_KEYS:
            if (row.auth or {}).get(name):
                try:
                    auth[name] = MASK + (hint(self.secrets.decrypt(row.auth[name])) if self.secrets else "")
                except SecretsUnavailable:
                    auth[name] = MASK
        cached = self._clients.get(row.id)
        return {"id": row.id, "name": row.name, "url": row.url, "auth": auth,
                "knowledge_bases": list(row.knowledge_bases or []), "disabled_kbs": list(row.disabled_kbs or []),
                "is_enabled": row.is_enabled, "is_builtin": row.is_builtin, "timeout": row.timeout,
                "error": cached[1].last_error if cached else "",
                "created_at": row.created_at.isoformat() if row.created_at else None}

    async def _row(self, session, ctx: AuthContext, remote_id: str, for_update: bool = False):

        row = await session.get(RemoteSource, remote_id, with_for_update=for_update)
        if row is None or row.deleted_at is not None or row.owner_subject not in self._owners(ctx):
            raise NotFound("remote not found")
        return row

    def _check_feature(self, ctx: AuthContext) -> None:
        if not self._owners(ctx):
            raise NotFound("remote knowledge bases are switched off in this deployment ([features] remote)")

    async def list_remotes(self, ctx: AuthContext) -> list[dict]:
        self._check_feature(ctx)

        async with async_session_scope(commit=False) as session:
            rows = (await session.execute(sa.select(RemoteSource).where(
                RemoteSource.owner_subject.in_(self._owners(ctx)), RemoteSource.deleted_at.is_(None)).order_by(
                RemoteSource.is_builtin.desc(), RemoteSource.name))).scalars().all()
            return [self._dict(r) for r in rows]

    async def get_remote(self, ctx: AuthContext, remote_id: str) -> dict:
        self._check_feature(ctx)
        async with async_session_scope(commit=False) as session:
            return self._dict(await self._row(session, ctx, remote_id))

    def _connection(self, body: dict, old: dict | None = None) -> dict:
        """The stored `auth`: a bearer `token`, or a personal access token `pat` with the Logto
        `auth_url`, `client_id` and `resource`. Secrets are encrypted; a masked value keeps the old one."""
        auth = dict(old or {})
        for key in AUTH_KEYS:
            if key in body:
                value = str(body[key] or "").strip()
                if value:
                    auth[key] = value.rstrip("/") if key == "auth_url" else value
                else:
                    auth.pop(key, None)
        for key in SECRET_KEYS:
            value = body.get(key)
            if value is None or str(value).startswith(MASK):
                continue
            if value == "":
                auth.pop(key, None)
                continue
            if self.secrets is None:
                raise Invalid("this KB has no secrets key (KB_SECRETS_KEY_FILE): credentials cannot be stored")
            auth[key] = self.secrets.encrypt(str(value).strip())
            auth.pop("pat" if key == "token" else "token", None)       # one credential at a time
        if not auth.get("token") and not auth.get("pat"):
            raise Invalid("a token or a personal access token (pat) is required")
        if auth.get("pat") and not all(auth.get(k) for k in ("auth_url", "client_id", "resource")):
            raise Invalid("a personal access token needs auth_url, client_id and resource")
        return auth

    @staticmethod
    def _url(value) -> str:
        url = str(value or "").strip().rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise Invalid("url must be the remote KB's http(s) base, e.g. https://host.example.org/app/kb")
        return url

    async def _try(self, row) -> None:
        """Reach the remote once, so a wrong URL or credential is reported when it is saved."""
        self._clients.pop(row.id, None)
        try:
            await self._client(row).list(refresh=True)
        except NotFound:
            raise Invalid(f"no knowledge service at {row.url}/api/v1")
        except RemoteError as exc:
            raise Invalid(str(exc))

    async def create(self, ctx: AuthContext, body: dict) -> dict:
        self._check_feature(ctx)
        if not features.enabled("remote"):
            raise Forbidden("remote knowledge bases are switched off in this deployment ([features] remote)")

        name = str(body.get("name") or "").strip()
        if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name):
            raise Invalid("name is required: letters, digits, '-', '_' or '.' (at most 64)")
        row = RemoteSource(id=str(uuid.uuid4()), owner_subject=ctx.subject, owner_org=ctx.organization_id, name=name,
                           url=self._url(body.get("url")), auth=self._connection(body),
                           knowledge_bases=[str(k) for k in body.get("knowledge_bases") or []],
                           disabled_kbs=[], is_enabled=True, is_builtin=False,
                           timeout=float(body.get("timeout") or 10.0))
        if body.get("check", True):
            await self._try(row)
        async with async_session_scope() as session:
            if (await session.execute(sa.select(RemoteSource.id).where(
                    RemoteSource.owner_subject.in_(self._owners(ctx)), RemoteSource.name == name,
                    RemoteSource.deleted_at.is_(None)))).first():
                raise Conflict(f"you already have a remote {name!r}")
            session.add(row)
            await session.flush()
            return self._dict(row)

    async def update(self, ctx: AuthContext, remote_id: str, body: dict) -> dict:
        """Switches (`is_enabled`, `disabled_kbs`) on any remote of the user; the connection
        (`url`, credentials, `knowledge_bases`, `name`, `timeout`) on the user's own only."""
        self._check_feature(ctx)
        connection = {"name", "url", "token", "pat", "knowledge_bases", "timeout", *AUTH_KEYS} & set(body)
        async with async_session_scope() as session:
            row = await self._row(session, ctx, remote_id, for_update=True)
            if connection and row.is_builtin:
                raise Forbidden("operators' remotes change in kb.cfg; only their switches change here")
            if "name" in body:
                name = str(body["name"] or "").strip()
                if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name):
                    raise Invalid("name: letters, digits, '-', '_' or '.' (at most 64)")
                row.name = name
            if "url" in body:
                row.url = self._url(body["url"])
            if {"token", "pat", *AUTH_KEYS} & set(body):
                row.auth = self._connection(body, row.auth)
            if "knowledge_bases" in body:
                row.knowledge_bases = [str(k) for k in body["knowledge_bases"] or []]
            if "timeout" in body:
                row.timeout = float(body["timeout"] or 10.0)
            if "is_enabled" in body:
                row.is_enabled = bool(body["is_enabled"])
            if "disabled_kbs" in body:
                row.disabled_kbs = sorted({split(k)[1] if is_remote(k) else str(k) for k in body["disabled_kbs"] or []})
            row.updated_at = utcnow()
            if connection and body.get("check", True):
                await self._try(row)
            return self._dict(row)

    async def set_kb(self, ctx: AuthContext, remote_id: str, kb_id: str, enabled: bool) -> dict:
        """Switch one remote KB on or off (its remote id, or its `remote:<name>:<id>`)."""
        self._check_feature(ctx)
        kb_id = split(kb_id)[1] if is_remote(kb_id) else kb_id
        async with async_session_scope() as session:
            row = await self._row(session, ctx, remote_id, for_update=True)
            disabled = set(row.disabled_kbs or ())
            (disabled.discard if enabled else disabled.add)(kb_id)
            row.disabled_kbs, row.updated_at = sorted(disabled), utcnow()
            return self._dict(row)

    async def delete(self, ctx: AuthContext, remote_id: str) -> None:
        self._check_feature(ctx)
        async with async_session_scope() as session:
            row = await self._row(session, ctx, remote_id, for_update=True)
            if row.is_builtin:
                raise Forbidden("operators' remotes are removed from kb.cfg; switch it off instead")
            row.deleted_at = utcnow()
        self._clients.pop(remote_id, None)

    async def remote_kbs(self, ctx: AuthContext, remote_id: str, refresh: bool = False) -> dict:
        """Every KB the remote offers, each with `enabled`, for the admin page; a remote that
        cannot be reached answers with its error and no KBs."""
        self._check_feature(ctx)
        async with async_session_scope(commit=False) as session:
            row = await self._row(session, ctx, remote_id)
            session.expunge(row)
        disabled = set(row.disabled_kbs or ())
        try:
            client = self._client(row)
            kbs = await client.list(refresh=refresh)
            client.last_error = ""
        except (RemoteError, NotFound) as exc:
            return {"remote": self._dict(row), "knowledge_bases": [], "error": str(exc)}
        return {"remote": self._dict(row), "error": "",
                "knowledge_bases": [{**kb, "enabled": kb["remote_id"] not in disabled} for kb in kbs]}


