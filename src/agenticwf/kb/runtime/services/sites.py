"""
SiteService: web sites imported from one URL (docs/ingestion.md, Sites).

A site is a root URL; every page under it becomes a `url` item with `site_id` set. Adding a site
and refreshing it enqueue `site:crawl`, which

    1. discovers the pages (agenticwf.kb.runtime.ingestion.crawl: sitemaps, then links);
    2. in one transaction: adds an item for each new page; enqueues a refresh
       (document:process with refresh=true) for each known page, so an unchanged page costs one
       fetch and no re-embedding; deletes the pages that are gone (`delete_removed`, on by
       default). Only a complete crawl deletes: when a page failed to fetch for now (timeout,
       429, 5xx) the pages under it may not have been found, and at `max_pages` the crawl did
       not see everything, so then nothing is deleted (`stats.removal_skipped`);
    3. records what it did in `stats`.

`maint:site_scan` re-crawls sites whose `refresh_interval` has elapsed. A URL that is already in
the KB as an item of its own is left alone (counted as `skipped`).
"""

from __future__ import annotations

import datetime

import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import Knowledge, KnowledgeSite, KnowledgeTagRelation, new_id, utcnow
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.errors import Conflict, Forbidden, Invalid, NotFound
from agenticwf.kb.runtime.ingestion import crawl
from agenticwf.kb.runtime.security.ssrf import FetchError
from agenticwf.kb.runtime.services.tags import resolve_tags
from agenticwf.kb.runtime.tasks import queue
from agenticwf.kb.runtime.tasks.runner import HandlerSpec, Permanent, ScheduleSpec, TaskContext

LOG = get_logger(__name__)

CRAWL_TIMEOUT = 3600
SCAN_SECONDS = 300
PAGE_SPACING = 1.0        # seconds between the page fetches a crawl queues, to stay under rate limits
CONFIG_FIELDS = ("max_pages", "max_depth", "follow_links", "use_sitemap", "include", "exclude", "delete_removed")


def schedules() -> tuple[ScheduleSpec, ...]:
    return (ScheduleSpec("site_scan", SCAN_SECONDS, "maint:site_scan"),)


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def site_dict(s: KnowledgeSite, pages: int = 0) -> dict:
    return {"id": s.id, "knowledge_base_id": s.knowledge_base_id, "url": s.url, "title": s.title,
            "config": s.config or {}, "status": s.status, "error_message": s.error_message, "stats": s.stats or {},
            "pages": pages, "refresh_interval": s.refresh_interval, "last_crawled_at": _iso(s.last_crawled_at),
            "owner_subject": s.owner_subject, "visibility": s.visibility, "groups": list(s.groups or []),
            "created_at": _iso(s.created_at), "updated_at": _iso(s.updated_at)}


async def _event(session, site: KnowledgeSite) -> None:
    await queue.write_event(session, site.knowledge_base_id, site.owner_subject,
                            {"type": "site_status", "site_id": site.id, "status": site.status},
                            site.visibility, list(site.groups or []))


def _config(body: dict, current: dict | None = None) -> dict:
    config = dict(current or {})
    for name in CONFIG_FIELDS:
        if name in body and body[name] is not None:
            config[name] = body[name]
    try:
        options = crawl.Options.from_config(config)
    except (TypeError, ValueError) as exc:
        raise Invalid(f"invalid crawl options: {exc}") from exc
    config.update(max_pages=options.max_pages, max_depth=options.max_depth, include=options.include,
                  exclude=options.exclude)
    config.setdefault("delete_removed", True)
    return config


class SiteService:
    def __init__(self, kbs, knowledge):
        self.kbs = kbs
        self.knowledge = knowledge          # KnowledgeService: sharing rules, enqueue, remove

    def handlers(self) -> dict[str, HandlerSpec]:
        return {"site:crawl": HandlerSpec(self._crawl_task, CRAWL_TIMEOUT, on_final_failure=self._crawl_failed),
                "maint:site_scan": HandlerSpec(self._scan, 120)}

    # -- loading with access checks ---------------------------------------------------------------
    async def _load(self, session, ctx: AuthContext, site_id: str, manage: bool = False):
        query = sa.select(KnowledgeSite).where(KnowledgeSite.id == site_id, KnowledgeSite.deleted_at.is_(None))
        site = (await session.execute(query.with_for_update() if manage else query)).scalar_one_or_none()
        if site is None:
            raise NotFound("site not found")
        kb = await self.kbs.load(session, ctx, site.knowledge_base_id)
        if not access.can_read_item(ctx, site):
            raise NotFound("site not found")
        if manage and not access.can_manage_item(ctx, site, kb):
            raise Forbidden("only the site's owner, the KB owner or an admin changes it")
        return site, kb

    async def _pages(self, session, site_ids: list[str]) -> dict[str, int]:
        if not site_ids:
            return {}
        rows = await session.execute(sa.select(Knowledge.site_id, sa.func.count()).where(
            Knowledge.site_id.in_(site_ids), Knowledge.deleted_at.is_(None)).group_by(Knowledge.site_id))
        return dict(rows.all())

    async def _enqueue(self, session, site: KnowledgeSite) -> None:
        site.status, site.error_message = "pending", ""
        await queue.enqueue(session, "site:crawl", {"site_id": site.id, "knowledge_base_id": site.knowledge_base_id},
                            owner_subject=site.owner_subject, owner_org=site.owner_org,
                            knowledge_base_id=site.knowledge_base_id, unique_key=f"site:{site.id}", max_retry=1)

    # -- API ---------------------------------------------------------------------------------------
    async def create(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        url = crawl.normalise(body.get("url") or "")
        if url is None:
            raise Invalid("url must be an http or https URL")
        config = _config(body)
        async with async_session_scope() as session:
            kb = await self.knowledge._contribute(session, ctx, kb_ref)
            existing = (await session.execute(sa.select(KnowledgeSite).where(
                KnowledgeSite.knowledge_base_id == kb.id, KnowledgeSite.url == url,
                KnowledgeSite.deleted_at.is_(None)))).scalars().all()
            readable = next((s for s in existing if access.can_read_item(ctx, s)), None)
            if readable is not None:
                raise Conflict("this site is already in the knowledge base", site_dict(readable))
            visibility, groups = self.knowledge._share(ctx, kb, body.get("visibility"), body.get("groups"))
            if body.get("tag_ids"):
                refs = body["tag_ids"] if isinstance(body["tag_ids"], list) else str(body["tag_ids"]).split(",")
                tag_ids, _ = await resolve_tags(session, [kb.id], [str(t).strip() for t in refs if str(t).strip()])
                if len(tag_ids) != len([t for t in refs if str(t).strip()]):
                    raise Invalid("unknown tag for this knowledge base")
                config["tag_ids"] = tag_ids
            site = KnowledgeSite(id=new_id(), knowledge_base_id=kb.id, owner_subject=ctx.subject,
                                 owner_org=ctx.organization_id, url=url, title=body.get("title") or url, config=config,
                                 visibility=visibility, groups=groups,
                                 refresh_interval=max(0, int(body.get("refresh_interval") or 0)))
            session.add(site)
            await session.flush()
            await self._enqueue(session, site)
            return site_dict(site)

    async def list(self, ctx: AuthContext, kb_ref: str) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self.kbs.load(session, ctx, kb_ref)
            rows = (await session.execute(sa.select(KnowledgeSite).where(
                KnowledgeSite.deleted_at.is_(None), access.items_clause(KnowledgeSite, ctx, [kb.id]))
                .order_by(KnowledgeSite.created_at.desc()))).scalars().all()
            pages = await self._pages(session, [s.id for s in rows])
            return [site_dict(s, pages.get(s.id, 0)) for s in rows]

    async def get(self, ctx: AuthContext, site_id: str) -> dict:
        async with async_session_scope(commit=False) as session:
            site, _ = await self._load(session, ctx, site_id)
            return site_dict(site, (await self._pages(session, [site.id])).get(site.id, 0))

    async def update(self, ctx: AuthContext, site_id: str, body: dict) -> dict:
        async with async_session_scope() as session:
            site, _ = await self._load(session, ctx, site_id, manage=True)
            site.config = {**_config(body, site.config), "tag_ids": (site.config or {}).get("tag_ids", [])}
            if body.get("title"):
                site.title = body["title"]
            if body.get("refresh_interval") is not None:
                site.refresh_interval = max(0, int(body["refresh_interval"]))
            site.updated_at = utcnow()
            return site_dict(site, (await self._pages(session, [site.id])).get(site.id, 0))

    async def refresh(self, ctx: AuthContext, site_id: str) -> dict:
        """Crawl again: add new pages, refresh the known ones, delete the ones that are gone."""
        async with async_session_scope() as session:
            site, _ = await self._load(session, ctx, site_id, manage=True)
            if site.status in ("pending", "crawling") and \
                    utcnow() - site.updated_at < datetime.timedelta(seconds=CRAWL_TIMEOUT):
                raise Conflict("this site is being crawled already", site_dict(site))
            await self._enqueue(session, site)
            site.updated_at = utcnow()
            return site_dict(site, (await self._pages(session, [site.id])).get(site.id, 0))

    async def delete(self, ctx: AuthContext, site_id: str, delete_pages: bool = True, purge: bool = False) -> dict:
        prefixes = []
        async with async_session_scope() as session:
            site, _ = await self._load(session, ctx, site_id, manage=True)
            items = (await session.execute(sa.select(Knowledge).where(
                Knowledge.site_id == site.id, Knowledge.deleted_at.is_(None)).with_for_update())).scalars().all()
            for item in items:
                if delete_pages:
                    prefixes.append(await self.knowledge.remove(session, item, purge))
                else:
                    item.site_id = None        # the pages stay, as items of their own
            site.deleted_at = utcnow()
            if purge:
                await session.flush()
                await session.execute(sa.delete(KnowledgeSite).where(KnowledgeSite.id == site.id))
        for prefix in prefixes:
            self.knowledge.storage.delete_prefix(prefix)
        return {"id": site_id, "deleted": True, "pages_deleted": len(prefixes) if delete_pages else 0}

    # -- tasks -------------------------------------------------------------------------------------
    async def _crawl_failed(self, ctx: TaskContext, error: str) -> None:
        async with async_session_scope() as session:
            site = await session.get(KnowledgeSite, ctx.payload["site_id"], with_for_update=True)
            if site is None:
                return
            site.status, site.error_message = "failed", error[:4000]
            site.last_crawled_at = site.updated_at = utcnow()
            await _event(session, site)

    async def _crawl_task(self, ctx: TaskContext) -> None:
        site_id = ctx.payload["site_id"]
        async with async_session_scope() as session:
            site = await session.get(KnowledgeSite, site_id, with_for_update=True)
            if site is None or site.deleted_at is not None:
                return
            site.status, site.updated_at = "crawling", utcnow()
            url, config = site.url, dict(site.config or {})
            await _event(session, site)
        try:
            found = await crawl.discover(url, crawl.Options.from_config(config))
        except (FetchError, ValueError) as exc:
            raise Permanent(str(exc)) from exc
        prefixes, stats = [], {"discovered": len(found.pages), "added": 0, "refreshed": 0, "removed": 0,
                               "skipped": 0, "unreachable": len(found.unreachable), "truncated": found.truncated,
                               "sitemap_pages": found.sitemap_pages, "errors": found.errors[:5]}
        async with async_session_scope() as session:
            site = await session.get(KnowledgeSite, site_id, with_for_update=True)
            if site is None or site.deleted_at is not None:
                return
            items = (await session.execute(sa.select(Knowledge).where(
                Knowledge.site_id == site.id, Knowledge.deleted_at.is_(None)).with_for_update())).scalars().all()
            known = {item.source: item for item in items}
            new = [u for u in found.pages if u not in known]
            taken = set((await session.execute(sa.select(Knowledge.source).where(
                Knowledge.knowledge_base_id == site.knowledge_base_id, Knowledge.type == "url",
                Knowledge.deleted_at.is_(None), Knowledge.source.in_(new)))).scalars()) if new else set()
            tag_ids, _ = await resolve_tags(session, [site.knowledge_base_id], config.get("tag_ids") or [])
            spaced = 0
            for page_url in new:
                if page_url in taken:
                    stats["skipped"] += 1
                    continue
                item = Knowledge(id=new_id(), knowledge_base_id=site.knowledge_base_id,
                                 owner_subject=site.owner_subject, owner_org=site.owner_org, type="url",
                                 source=page_url, title=page_url, visibility=site.visibility,
                                 groups=list(site.groups or []), channel="site", site_id=site.id)
                self.knowledge._apply_citation(item, {}, {})
                session.add(item)
                await session.flush()
                for tag_id in tag_ids:
                    session.add(KnowledgeTagRelation(knowledge_id=item.id, tag_id=tag_id))
                await self.knowledge._enqueue(session, item, delay=PAGE_SPACING * spaced)
                spaced += 1
                stats["added"] += 1
            pages, complete = set(found.pages), not found.unreachable and not found.truncated
            stats["removal_skipped"] = not complete
            for page_url, item in known.items():
                if page_url not in pages:
                    if complete and config.get("delete_removed", True):
                        prefixes.append(await self.knowledge.remove(session, item))
                        stats["removed"] += 1
                elif page_url in pages and item.parse_status in ("completed", "failed"):
                    if await queue.enqueue(session, "document:process", {
                            "knowledge_id": item.id, "knowledge_base_id": item.knowledge_base_id,
                            "refresh": item.parse_status == "completed"},
                            owner_subject=item.owner_subject, owner_org=item.owner_org,
                            knowledge_base_id=item.knowledge_base_id, unique_key=f"refresh:{item.id}", max_retry=1,
                            delay=PAGE_SPACING * spaced):
                        spaced += 1
                        stats["refreshed"] += 1
            site.status, site.error_message, site.stats = "completed", "", stats
            site.last_crawled_at = site.updated_at = utcnow()
            await _event(session, site)
        for prefix in prefixes:
            self.knowledge.storage.delete_prefix(prefix)
        LOG.info("crawled site %s (%s): %s", site_id, url, stats)

    async def _scan(self, ctx: TaskContext | None = None) -> int:
        now, queued = utcnow(), 0
        async with async_session_scope() as session:
            sites = (await session.execute(sa.select(KnowledgeSite).where(
                KnowledgeSite.deleted_at.is_(None), KnowledgeSite.refresh_interval > 0,
                KnowledgeSite.status.in_(("completed", "failed")), KnowledgeSite.last_crawled_at.is_not(None),
                KnowledgeSite.last_crawled_at + sa.func.make_interval(0, 0, 0, 0, 0, 0,
                                                                       KnowledgeSite.refresh_interval) <= now)
            )).scalars().all()
            for site in sites:
                await self._enqueue(session, site)
                queued += 1
        return queued
