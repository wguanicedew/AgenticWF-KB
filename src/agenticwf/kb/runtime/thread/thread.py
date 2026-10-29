"""
KBThread: one principal's handle on the KB, and the only way a host reaches the services.

A host (an API route, a future agentic-search run) opens a thread for the request's AuthContext
through KBThreadManager and calls its methods; the thread passes the principal to the services,
which apply the data-level rules (docs/data-model.md, Who may do what): what is visible, who may
contribute, manage or share, someone else's object answering "not found". Route checks
(agenticwf.kb.api.deps.require) and authsvc's policy come before it; the thread does not rely on them.

SystemThread is the same façade for callers that act without a principal: health checks, the
CLI (an operator) and workers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from agenticwf.auth.context import AuthContext

from agenticwf.kb.runtime.access import group_names, principal_groups
from agenticwf.kb.runtime.features import enabled as feature_enabled  # noqa: F401  (hosts check [features] here)
from agenticwf.kb.runtime.errors import Forbidden
from agenticwf.kb.runtime.remote import is_remote
from agenticwf.kb.runtime.retrieval.search import SearchOptions
from agenticwf.kb.runtime.services import KBServices

SCOPE_ADMIN = "admin"


def owner_of(ctx: AuthContext) -> str:
    """What the KB keys ownership by: the user, not the token (docs/data-model.md)."""
    return ctx.subject


@dataclass
class KBThread:
    auth: AuthContext
    services: KBServices = field(repr=False)
    source: str = "api"                      # standalone | authsvc | service | cli | worker
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    closed: bool = False

    @property
    def owner(self) -> str:
        return owner_of(self.auth)

    @property
    def is_admin(self) -> bool:
        return SCOPE_ADMIN in self.auth.scopes

    def _require_admin(self) -> None:
        if not self.is_admin:
            raise Forbidden("missing scopes: admin")

    # -- identity and system -----------------------------------------------------------------
    async def me(self) -> dict:
        ctx = self.auth
        return {
            "subject": ctx.subject,
            "user_id": ctx.user_id,
            "client_id": ctx.client_id,
            "organization_id": ctx.organization_id,
            "roles": sorted(ctx.roles),
            "scopes": sorted(ctx.scopes),
            "groups": principal_groups(ctx),
            "group_names": group_names(ctx),
            "is_machine": ctx.is_machine,
            "auth_method": ctx.auth_method,
            "personal_knowledge_base_id": await self.services.knowledge_bases.ensure_personal(ctx),
        }

    async def system_info(self) -> dict:
        return await self.services.system.info()

    # -- models ------------------------------------------------------------------------------------
    async def list_models(self, model_type: str | None = None) -> list[dict]:
        return await self.services.models.list(self.auth, model_type)

    async def get_model(self, ref: str) -> dict:
        return await self.services.models.get(self.auth, ref)

    async def create_model(self, body: dict) -> dict:
        return await self.services.models.create(self.auth, body)

    async def update_model(self, ref: str, body: dict) -> dict:
        return await self.services.models.update(self.auth, ref, body)

    async def delete_model(self, ref: str) -> None:
        await self.services.models.delete(self.auth, ref)

    async def debug_model(self, ref: str) -> dict:
        return await self.services.models.debug(self.auth, ref)

    # -- knowledge bases ---------------------------------------------------------------------------
    async def list_kbs(self, page=1, page_size=20, shared_only=False, kb_type=None):
        await self.services.knowledge_bases.ensure_personal(self.auth)
        rows, total = await self.services.knowledge_bases.list(self.auth, page, page_size, shared_only, kb_type)
        remote = getattr(self.services, "remote", None)
        extra = await remote.list_kbs(self.auth, kb_type) if remote is not None and not shared_only else []
        if not extra:
            return rows, total
        # The enabled remote KBs follow the local ones, so pages stay stable while a remote is down.
        start = max(0, (page - 1) * page_size - total)
        return rows + extra[start:start + page_size - len(rows)], total + len(extra)

    async def get_kb(self, ref: str) -> dict:
        if is_remote(ref):
            return await self.services.remote.get_kb(self.auth, ref)
        return await self.services.knowledge_bases.get(self.auth, ref)

    # -- remote knowledge bases (docs/retrieval.md, Remote knowledge bases) -------------------------------
    async def remotes(self) -> list[dict]:
        return await self.services.remote.list_remotes(self.auth)

    async def get_remote(self, remote_id: str) -> dict:
        return await self.services.remote.get_remote(self.auth, remote_id)

    async def create_remote(self, body: dict) -> dict:
        return await self.services.remote.create(self.auth, body)

    async def update_remote(self, remote_id: str, body: dict) -> dict:
        return await self.services.remote.update(self.auth, remote_id, body)

    async def delete_remote(self, remote_id: str) -> None:
        await self.services.remote.delete(self.auth, remote_id)

    async def remote_kbs(self, remote_id: str, refresh: bool = False) -> dict:
        return await self.services.remote.remote_kbs(self.auth, remote_id, refresh)

    async def set_remote_kb(self, remote_id: str, kb_id: str, enabled: bool) -> dict:
        return await self.services.remote.set_kb(self.auth, remote_id, kb_id, enabled)

    async def create_kb(self, body: dict) -> dict:
        if body.get("records_config"):
            body = {**body, "records_config": await self.services.records.check_records_config(
                self.auth, body["records_config"])}
        return await self.services.knowledge_bases.create(self.auth, body)

    async def update_kb(self, ref: str, body: dict) -> dict:
        if body.get("records_config") is not None:
            body = {**body, "records_config": await self.services.records.check_records_config(
                self.auth, body["records_config"])}
        kb = await self.services.knowledge_bases.update(self.auth, ref, body)
        if body.get("faq_config") is not None:
            await self.services.faq.reindex(self.auth, kb["id"])
        return kb

    async def delete_kb(self, ref: str, purge: bool = False) -> None:
        await self.services.knowledge_bases.delete(self.auth, ref, purge)

    # -- knowledge -----------------------------------------------------------------------------------
    async def add_file(self, kb_ref: str, file_name: str, data: bytes, fields: dict) -> tuple[str, dict]:
        kb = await self.services.knowledge_bases.get(self.auth, kb_ref)
        if kb["type"] == "records":
            return "created", await self.services.records.upload_file(self.auth, kb_ref, file_name, data, fields)
        return await self.services.knowledge.create_file(self.auth, kb_ref, file_name, data, fields)

    async def add_url(self, kb_ref: str, body: dict) -> dict:
        return await self.services.knowledge.create_url(self.auth, kb_ref, body)

    async def add_site(self, kb_ref: str, body: dict) -> dict:
        return await self.services.sites.create(self.auth, kb_ref, body)

    async def list_sites(self, kb_ref: str) -> list[dict]:
        return await self.services.sites.list(self.auth, kb_ref)

    async def get_site(self, site_id: str) -> dict:
        return await self.services.sites.get(self.auth, site_id)

    async def update_site(self, site_id: str, body: dict) -> dict:
        return await self.services.sites.update(self.auth, site_id, body)

    async def refresh_site(self, site_id: str) -> dict:
        return await self.services.sites.refresh(self.auth, site_id)

    async def delete_site(self, site_id: str, delete_pages: bool = True, purge: bool = False) -> dict:
        return await self.services.sites.delete(self.auth, site_id, delete_pages, purge)

    async def add_manual(self, kb_ref: str, body: dict) -> dict:
        return await self.services.knowledge.create_manual(self.auth, kb_ref, body)

    async def update_manual(self, knowledge_id: str, body: dict) -> dict:
        return await self.services.knowledge.update_manual(self.auth, knowledge_id, body)

    async def list_knowledge(self, kb_ref: str, page=1, page_size=20, filters=None):
        return await self.services.knowledge.list(self.auth, kb_ref, page, page_size, filters)

    async def get_knowledge(self, knowledge_id: str) -> dict:
        if is_remote(knowledge_id):
            return await self.services.remote.read(self.auth, "knowledge", knowledge_id)
        return await self.services.knowledge.get(self.auth, knowledge_id)

    async def batch_knowledge(self, ids: list[str]) -> list[dict]:
        return await self.services.knowledge.batch(self.auth, ids)

    async def knowledge_content(self, knowledge_id: str, offset: int = 0, limit: int = 8000) -> dict:
        if is_remote(knowledge_id):
            return await self.services.remote.read(self.auth, "content", knowledge_id, offset=offset, limit=limit)
        return await self.services.knowledge.content(self.auth, knowledge_id, offset, limit)

    async def download(self, knowledge_id: str) -> tuple[str, bytes]:
        return await self.services.knowledge.download(self.auth, knowledge_id)

    async def update_knowledge(self, knowledge_id: str, body: dict) -> dict:
        return await self.services.knowledge.update(self.auth, knowledge_id, body)

    async def delete_knowledge(self, knowledge_id: str, purge: bool = False) -> None:
        await self.services.knowledge.delete(self.auth, knowledge_id, purge)

    async def redact(self, knowledge_id: str, body: dict) -> dict:
        return await self.services.knowledge.redact(self.auth, knowledge_id, body)

    async def redact_kb(self, kb_ref: str, body: dict) -> dict:
        return await self.services.knowledge.redact_kb(self.auth, kb_ref, body)

    async def reparse(self, knowledge_id: str) -> dict:
        return await self.services.knowledge.reparse(self.auth, knowledge_id)

    async def cancel(self, knowledge_id: str) -> dict:
        return await self.services.knowledge.cancel(self.auth, knowledge_id)

    # -- chunks --------------------------------------------------------------------------------------
    async def list_chunks(self, knowledge_id: str, page=1, page_size=20):
        return await self.services.knowledge.chunks(self.auth, knowledge_id, page, page_size)

    async def get_chunk(self, chunk_id: str) -> dict:
        if is_remote(chunk_id):
            return await self.services.remote.read(self.auth, "chunk", chunk_id)
        return await self.services.knowledge.chunk(self.auth, chunk_id)

    async def update_chunk(self, knowledge_id: str, chunk_id: str, body: dict) -> dict:
        return await self.services.knowledge.update_chunk(self.auth, knowledge_id, chunk_id, body)

    async def delete_chunks(self, knowledge_id: str, chunk_id: str | None = None) -> None:
        await self.services.knowledge.delete_chunk(self.auth, knowledge_id, chunk_id)

    # -- search --------------------------------------------------------------------------------------
    async def search(self, query: str, kb_refs: list[str] | None = None,
                     options: SearchOptions | None = None) -> list[dict]:
        return await self.services.search.search(self.auth, query, kb_refs, options)

    async def grep(self, pattern: str, kb_refs: list[str] | None = None, knowledge_ids: list[str] | None = None,
                   **options) -> dict:
        from agenticwf.kb.runtime.retrieval.grep import grep

        return await grep(self.auth, self.services.knowledge_bases, pattern, kb_refs=kb_refs,
                          knowledge_ids=knowledge_ids, **options)

    # -- tags ----------------------------------------------------------------------------------------
    async def list_tags(self, kb_ref: str, keyword=None, page=1, page_size=50):
        return await self.services.tags.list(self.auth, kb_ref, keyword, page, page_size)

    async def create_tag(self, kb_ref: str, body: dict) -> dict:
        return await self.services.tags.create(self.auth, kb_ref, body)

    async def update_tag(self, kb_ref: str, tag_id: str, body: dict) -> dict:
        return await self.services.tags.update(self.auth, kb_ref, tag_id, body)

    async def delete_tag(self, kb_ref: str, tag_id: str, force: bool = False) -> dict:
        return await self.services.tags.delete(self.auth, kb_ref, tag_id, force)

    async def set_knowledge_tags(self, updates: dict) -> dict:
        return await self.services.knowledge.set_tags(self.auth, updates)

    # -- versions and freshness -------------------------------------------------------------------
    async def versions(self, knowledge_id: str) -> dict:
        return await self.services.knowledge.versions(self.auth, knowledge_id)

    async def download_version(self, knowledge_id: str, version: int) -> tuple[str, bytes]:
        return await self.services.knowledge.download_version(self.auth, knowledge_id, version)

    async def staleness(self, kb_ref: str, **params) -> dict:
        return await self.services.freshness.staleness(self.auth, kb_ref, **params)

    async def conflicts(self, kb_ref: str, status: str = "open", knowledge_id: str | None = None) -> list[dict]:
        return await self.services.freshness.conflicts(self.auth, kb_ref, status, knowledge_id)

    async def review_conflict(self, conflict_id: str, body: dict) -> dict:
        return await self.services.freshness.review(self.auth, conflict_id, body)

    async def reindex(self, kb_ref: str, body: dict) -> dict:
        return await self.services.knowledge.reindex(self.auth, kb_ref, body)

    # -- structured records (phase 3) --------------------------------------------------------------
    async def record_types(self, all_versions: bool = False) -> list[dict]:
        return await self.services.records.list_types(self.auth, all_versions)

    async def record_type(self, name: str) -> dict:
        return await self.services.records.get_type(self.auth, name)

    async def register_record_type(self, body: dict) -> dict:
        return await self.services.records.register(self.auth, body)

    async def update_record_type(self, name: str, body: dict, new_version: bool = False) -> dict:
        return await self.services.records.update_type(self.auth, name, body, new_version)

    async def deprecate_record_type(self, name: str) -> dict:
        return await self.services.records.deprecate_type(self.auth, name)

    async def infer_record_type(self, body: dict, file_name: str | None = None, data: bytes | None = None) -> dict:
        return await self.services.records.infer_type(self.auth, body, file_name, data)

    async def record_type_stats(self, name: str, kb_ref: str | None = None) -> list[dict]:
        return await self.services.records.stats(self.auth, name, kb_ref)

    async def migrate_record_type(self, name: str, body: dict) -> dict:
        return await self.services.records.migrate(self.auth, name, body)

    async def record_type_operation(self, name: str, op: str, body: dict) -> dict:
        return await self.services.records.operation(self.auth, name, op, body, self.services)

    async def upsert_records(self, kb_ref: str, body: dict) -> dict:
        return await self.services.records.upsert(self.auth, kb_ref, body)

    async def list_records(self, kb_ref: str, record_type=None, facets=None, page=1, page_size=20):
        return await self.services.records.list_records(self.auth, kb_ref, record_type, facets, page, page_size)

    async def get_record(self, record_id: str, version: int | None = None) -> dict:
        return await self.services.records.get_record(self.auth, record_id, version)

    async def delete_record(self, record_id: str) -> None:
        await self.services.records.delete_record(self.auth, record_id)

    async def query_records(self, body: dict) -> dict:
        return await self.services.records.query(self.auth, body)

    async def aggregate_records(self, body: dict) -> dict:
        return await self.services.records.aggregate(self.auth, body)

    async def similar_records(self, body: dict) -> dict:
        return await self.services.records.similar(self.auth, body)

    # -- LLM wiki (phase 3) -------------------------------------------------------------------------
    async def wiki_pages(self, kb_ref, **filters):
        return await self.services.wiki.list_pages(self.auth, kb_ref, **filters)

    async def wiki_page(self, kb_ref, slug) -> dict:
        return await self.services.wiki.get_page(self.auth, kb_ref, slug)

    async def wiki_create_page(self, kb_ref, body) -> dict:
        return await self.services.wiki.create_page(self.auth, kb_ref, body)

    async def wiki_update_page(self, kb_ref, slug, body) -> dict:
        return await self.services.wiki.update_page(self.auth, kb_ref, slug, body)

    async def wiki_delete_page(self, kb_ref, slug) -> None:
        await self.services.wiki.delete_page(self.auth, kb_ref, slug)

    async def wiki_move_page(self, kb_ref, slug, folder_id) -> dict:
        return await self.services.wiki.move_page(self.auth, kb_ref, slug, folder_id)

    async def wiki_folders(self, kb_ref) -> list[dict]:
        return await self.services.wiki.folders(self.auth, kb_ref)

    async def wiki_create_folder(self, kb_ref, body) -> dict:
        return await self.services.wiki.create_folder(self.auth, kb_ref, body)

    async def wiki_update_folder(self, kb_ref, folder_id, body) -> dict:
        return await self.services.wiki.update_folder(self.auth, kb_ref, folder_id, body)

    async def wiki_delete_folder(self, kb_ref, folder_id) -> None:
        await self.services.wiki.delete_folder(self.auth, kb_ref, folder_id)

    async def wiki_index(self, kb_ref) -> dict:
        return await self.services.wiki.index(self.auth, kb_ref)

    async def wiki_graph(self, kb_ref, slug=None) -> dict:
        return await self.services.wiki.graph(self.auth, kb_ref, slug)

    async def wiki_stats(self, kb_ref) -> dict:
        return await self.services.wiki.stats(self.auth, kb_ref)

    async def wiki_search(self, kb_ref, q, regex=True, limit=10) -> list[dict]:
        return await self.services.wiki.search(self.auth, kb_ref, q, regex, limit)

    async def wiki_revisions(self, kb_ref, slug, version=None):
        return await self.services.wiki.revisions(self.auth, kb_ref, slug, version)

    async def wiki_revert(self, kb_ref, slug, version) -> dict:
        return await self.services.wiki.revert(self.auth, kb_ref, slug, version)

    async def wiki_issues(self, kb_ref, status="pending") -> list[dict]:
        return await self.services.wiki.issues(self.auth, kb_ref, status)

    async def wiki_flag_issue(self, kb_ref, body) -> dict:
        return await self.services.wiki.flag_issue(self.auth, kb_ref, body)

    async def wiki_issue_status(self, kb_ref, issue_id, status) -> dict:
        return await self.services.wiki.set_issue_status(self.auth, kb_ref, issue_id, status)

    async def wiki_quality(self, kb_ref) -> dict:
        return await self.services.wiki.quality(self.auth, kb_ref)

    async def wiki_lint(self, kb_ref) -> dict:
        return await self.services.wiki.lint(self.auth, kb_ref)

    async def wiki_rebuild_links(self, kb_ref) -> dict:
        return await self.services.wiki.rebuild_links(self.auth, kb_ref)

    async def wiki_rebuild(self, kb_ref) -> dict:
        return await self.services.wiki.rebuild(self.auth, kb_ref)

    async def wiki_global_search(self, kb_ref, query, level=0) -> dict:
        return await self.services.wiki_pipeline.global_search(self.auth, kb_ref, query, level)

    async def wiki_relations(self, kb_ref, slug=None) -> list[dict]:
        return await self.services.wiki.relations(self.auth, kb_ref, slug)

    async def wiki_communities(self, kb_ref) -> list[dict]:
        return await self.services.wiki.communities(self.auth, kb_ref)

    # -- web search ---------------------------------------------------------------------------------
    def web_kinds(self) -> list[dict]:
        return self.services.web.kinds()

    async def web_providers(self) -> list[dict]:
        return await self.services.web.list(self.auth)

    async def web_preferences(self) -> dict:
        return await self.services.web.preferences(self.auth)

    async def web_set_preferences(self, body: dict) -> dict:
        return await self.services.web.set_preferences(self.auth, body)

    async def web_provider(self, ref) -> dict:
        return await self.services.web.get(self.auth, ref)

    async def web_create_provider(self, body) -> dict:
        return await self.services.web.create(self.auth, body)

    async def web_update_provider(self, ref, body) -> dict:
        return await self.services.web.update(self.auth, ref, body)

    async def web_provider_credentials(self, ref, body) -> dict:
        return await self.services.web.set_credentials(self.auth, ref, body)

    async def web_delete_provider(self, ref) -> None:
        await self.services.web.delete(self.auth, ref)

    async def web_test_provider(self, ref=None, body=None) -> dict:
        return await self.services.web.test(self.auth, ref, body)

    async def web_search(self, body: dict) -> dict:
        return await self.services.web.search(self.auth, body.get("query") or "", body.get("provider_id"),
                                              body.get("max_results"), body.get("fetch_top"),
                                              cache=self.services.web.cache_for(self.auth, body.get("cache_key")),
                                              providers=body.get("provider_ids") or None)

    async def web_fetch(self, body: dict) -> dict:
        return await self.services.web.fetch(self.auth, body.get("url") or "", body.get("offset") or 0,
                                             body.get("limit") or 8000,
                                             cache=self.services.web.cache_for(self.auth, body.get("cache_key")))

    async def regenerate_summary(self, knowledge_id: str) -> dict:
        return await self.services.knowledge.regenerate_summary(self.auth, knowledge_id)

    # -- temporal memory (phase 3): the caller's own memory only -----------------------------------
    async def memory_add_episode(self, body: dict) -> dict:
        return await self.services.memory.add_episode(self.auth, body)

    async def memory_episodes(self, group_id=None, page=1, page_size=20):
        return await self.services.memory.list_episodes(self.auth, group_id, page, page_size)

    async def memory_episode(self, episode_id) -> dict:
        return await self.services.memory.get_episode(self.auth, episode_id)

    async def memory_delete_episode(self, episode_id) -> dict:
        return await self.services.memory.delete_episode(self.auth, episode_id)

    async def memory_delete_fact(self, fact_id) -> dict:
        return await self.services.memory.delete_fact(self.auth, fact_id)

    async def memory_delete_entity(self, entity_id) -> dict:
        return await self.services.memory.delete_entity(self.auth, entity_id)

    async def memory_redact(self, body: dict) -> dict:
        return await self.services.memory.redact(self.auth, body)

    async def memory_facts(self, **filters) -> list[dict]:
        return await self.services.memory.facts(self.auth, **filters)

    async def memory_search(self, body: dict) -> list[dict]:
        return await self.services.memory.search(self.auth, body)

    async def memory_entities(self, group_id=None, q=None, page=1, page_size=50):
        return await self.services.memory.entities(self.auth, group_id, q, page, page_size)

    async def memory_entity(self, entity_id) -> dict:
        return await self.services.memory.entity(self.auth, entity_id)

    # -- agentic search ----------------------------------------------------------------------------
    def agentic_presets(self) -> list[dict]:
        return self.services.agentic.presets()

    async def agentic_tools(self) -> dict:
        return await self.services.agentic.tools()

    async def agentic_search(self, body: dict):
        """A prepared run (validated, scoped, holding a per-owner slot); iterate `run.frames()`."""
        return await self.services.agentic.prepare(self.auth, body)

    # -- FAQ ---------------------------------------------------------------------------------------
    async def faq_entries(self, kb_ref: str, page=1, page_size=20, **filters):
        return await self.services.faq.list_entries(self.auth, kb_ref, page, page_size, **filters)

    async def faq_entry(self, kb_ref: str, entry_id) -> dict:
        return await self.services.faq.get_entry(self.auth, kb_ref, entry_id)

    async def faq_upsert(self, kb_ref: str, entries: list, mode: str = "append", dry_run: bool = False) -> dict:
        return await self.services.faq.upsert(self.auth, kb_ref, entries, mode=mode, dry_run=dry_run)

    async def faq_create(self, kb_ref: str, body: dict) -> dict:
        return await self.services.faq.create_entry(self.auth, kb_ref, body)

    async def faq_update(self, kb_ref: str, entry_id, body: dict) -> dict:
        return await self.services.faq.update_entry(self.auth, kb_ref, entry_id, body)

    async def faq_add_similar(self, kb_ref: str, entry_id, questions: list[str]) -> dict:
        return await self.services.faq.add_similar(self.auth, kb_ref, entry_id, questions)

    async def faq_update_fields(self, kb_ref: str, body: dict) -> dict:
        return await self.services.faq.update_fields(self.auth, kb_ref, body)

    async def faq_update_tags(self, kb_ref: str, updates: dict) -> dict:
        return await self.services.faq.update_tags(self.auth, kb_ref, updates)

    async def faq_delete(self, kb_ref: str, ids: list) -> dict:
        return await self.services.faq.delete_entries(self.auth, kb_ref, ids)

    async def faq_search(self, kb_ref: str, body: dict) -> list[dict]:
        return await self.services.faq.search(self.auth, kb_ref, body)

    async def faq_import(self, kb_ref: str, file_name: str, data: bytes, mode: str = "append",
                         dry_run: bool = False) -> dict:
        return await self.services.faq.import_file(self.auth, kb_ref, file_name, data, mode, dry_run)

    async def faq_export(self, kb_ref: str) -> tuple[str, bytes]:
        return await self.services.faq.export(self.auth, kb_ref)

    async def prompts(self) -> list[dict]:
        self._require_admin()
        return await self.services.prompts.list()

    async def set_prompt(self, file: str, key: str, text) -> dict:
        self._require_admin()
        return await self.services.prompts.set(self.auth, file, key, text)

    async def reset_prompt(self, file: str, key: str) -> dict:
        self._require_admin()
        return await self.services.prompts.reset(file, key)

    async def housekeeping(self, dry_run: bool = True, retention_days: int | None = None) -> dict:
        self._require_admin()
        return await self.services.freshness.housekeeping(dry_run, retention_days)


class SystemThread:
    """The runtime for callers without a principal: health, the CLI, workers."""

    def __init__(self, services: KBServices):
        self.services = services

    async def healthy(self) -> bool:
        return await self.services.system.healthy()

    async def info(self) -> dict:
        return await self.services.system.info()

    async def check_database(self) -> dict[str, str]:
        return await self.services.system.check_database()

    async def sync_system_models(self) -> list[str]:
        return await self.services.models.sync_system_models()
