"""
Freshness (docs/ingestion.md, Freshness: validity, refresh, expiry, housekeeping; Conflicts).

    maint:expire          hourly: items past `expires_at` are disabled (or deleted with
                          `delete_on_expiry`); search filters by time on every query anyway
    maint:refresh_scan    every 60 s: enqueue a refresh (document:process with refresh=true) for
                          URL items whose `refresh_interval` has elapsed; an unchanged page costs
                          one fetch and no re-embedding
    maint:housekeeping    daily, and POST /system/housekeeping (dry run by default): orphan index
                          rows and chunks, orphan storage directories (older than 1 h), purge of
                          soft-deleted items and knowledge bases older than
                          `[maintenance] purge_after_days`

plus the staleness report and the review of conflicts.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import anyio
import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import Chunk, Embedding, Knowledge, KnowledgeBase, KnowledgeConflict, utcnow
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.errors import Forbidden, Invalid, NotFound
from agenticwf.kb.runtime.services.knowledge_bases import KnowledgeBaseService, purge_kb_rows
from agenticwf.kb.runtime.storage import owner_scope
from agenticwf.kb.runtime.tasks import queue
from agenticwf.kb.runtime.tasks.runner import HandlerSpec, ScheduleSpec, TaskContext

LOG = get_logger(__name__)

ORPHAN_MIN_AGE = 3600


def _maintenance(option: str, default: int) -> int:
    return int(config_get("maintenance", option)) if config_has_option("maintenance", option) else default


def schedules() -> tuple[ScheduleSpec, ...]:
    return (ScheduleSpec("expire", _maintenance("expire_interval_seconds", 3600), "maint:expire"),
            ScheduleSpec("refresh_scan", _maintenance("refresh_scan_seconds", 60), "maint:refresh_scan"),
            ScheduleSpec("housekeeping", _maintenance("housekeeping_interval_hours", 24) * 3600, "maint:housekeeping"))


def conflict_dict(c: KnowledgeConflict, titles: dict[str, str]) -> dict:
    return {"id": c.id, "knowledge_base_id": c.knowledge_base_id, "knowledge_id": c.knowledge_id,
            "knowledge_title": titles.get(c.knowledge_id, ""), "other_knowledge_id": c.other_knowledge_id,
            "other_knowledge_title": titles.get(c.other_knowledge_id, ""), "similarity": round(c.similarity, 4),
            "matched_chunks": c.matched_chunks, "other_coverage": round(c.other_coverage, 4), "pairs": c.pairs,
            "verdict": c.verdict, "explanation": c.explanation, "status": c.status,
            "created_at": c.created_at.isoformat() if c.created_at else None}


class FreshnessService:
    def __init__(self, storage, kbs: KnowledgeBaseService, knowledge):
        self.storage = storage
        self.kbs = kbs
        self.knowledge = knowledge

    def handlers(self) -> dict[str, HandlerSpec]:
        return {"maint:expire": HandlerSpec(self._expire, 600),
                "maint:refresh_scan": HandlerSpec(self._refresh_scan, 120),
                "maint:housekeeping": HandlerSpec(self._housekeeping_task, 1800)}

    # -- expiry and refresh ------------------------------------------------------------------------
    async def _expire(self, ctx: TaskContext | None = None) -> dict:
        now = utcnow()
        disabled = deleted = 0
        async with async_session_scope() as session:
            rows = (await session.execute(sa.select(Knowledge, KnowledgeBase).join(
                KnowledgeBase, KnowledgeBase.id == Knowledge.knowledge_base_id).where(
                Knowledge.deleted_at.is_(None), Knowledge.expires_at.is_not(None), Knowledge.expires_at <= now,
                Knowledge.enable_status == "enabled"))).all()
            for item, kb in rows:
                if (kb.freshness_config or {}).get("delete_on_expiry"):
                    await session.execute(sa.delete(Embedding).where(Embedding.knowledge_id == item.id))
                    await session.execute(sa.delete(Chunk).where(Chunk.knowledge_id == item.id))
                    item.deleted_at, item.parse_status = now, "deleted"
                    kb.storage_used = max(0, kb.storage_used - item.file_size)
                    deleted += 1
                else:
                    item.enable_status = "disabled"
                    await session.execute(sa.update(Embedding).where(Embedding.knowledge_id == item.id)
                                          .values(is_enabled=False))
                    disabled += 1
        if disabled or deleted:
            LOG.info("expiry: %d disabled, %d deleted", disabled, deleted)
        return {"disabled": disabled, "deleted": deleted}

    async def _refresh_scan(self, ctx: TaskContext | None = None) -> int:
        now = utcnow()
        queued = 0
        async with async_session_scope() as session:
            items = (await session.execute(sa.select(Knowledge).where(
                Knowledge.deleted_at.is_(None), Knowledge.type == "url", Knowledge.refresh_interval > 0,
                Knowledge.parse_status.in_(("completed", "failed")),
                sa.or_(Knowledge.last_refreshed_at.is_(None),
                       Knowledge.last_refreshed_at + sa.func.make_interval(0, 0, 0, 0, 0, 0,
                                                                            Knowledge.refresh_interval) <= now))
            )).scalars().all()
            for item in items:
                if item.last_refreshed_at is None and item.processed_at is not None and \
                        (now - item.processed_at).total_seconds() < item.refresh_interval:
                    continue
                if await queue.enqueue(session, "document:process", {"knowledge_id": item.id,
                                                                     "knowledge_base_id": item.knowledge_base_id,
                                                                     "refresh": True},
                                       owner_subject=item.owner_subject, knowledge_base_id=item.knowledge_base_id,
                                       unique_key=f"refresh:{item.id}", max_retry=1):
                    queued += 1
        return queued

    # -- housekeeping --------------------------------------------------------------------------------
    async def _housekeeping_task(self, ctx: TaskContext | None = None) -> dict:
        return await self.housekeeping(dry_run=False)

    async def housekeeping(self, dry_run: bool = True, retention_days: int | None = None) -> dict:
        days = retention_days if retention_days is not None else _maintenance("purge_after_days", 30)
        cutoff = utcnow() - datetime.timedelta(days=days)
        report: dict = {"dry_run": dry_run, "retention_days": days}
        async with async_session_scope(commit=not dry_run) as session:
            orphan_rows = sa.select(Embedding.id).where(
                Embedding.source_type <= 3, ~sa.exists().where(Chunk.id == Embedding.chunk_id))
            report["orphan_index_rows"] = (await session.execute(
                sa.select(sa.func.count()).select_from(orphan_rows.subquery()))).scalar_one()
            dead_items = sa.select(Knowledge.id).where(Knowledge.deleted_at.is_not(None))
            report["orphan_chunks"] = (await session.execute(sa.select(sa.func.count()).select_from(Chunk).where(
                Chunk.knowledge_id.in_(dead_items)))).scalar_one()
            purge = sa.select(Knowledge.id).where(Knowledge.deleted_at.is_not(None), Knowledge.deleted_at < cutoff)
            report["purged_items"] = (await session.execute(
                sa.select(sa.func.count()).select_from(purge.subquery()))).scalar_one()
            dead_kbs = list((await session.execute(sa.select(KnowledgeBase.id).where(
                KnowledgeBase.deleted_at.is_not(None), KnowledgeBase.deleted_at < cutoff))).scalars())
            report["purged_knowledge_bases"] = len(dead_kbs)
            if not dry_run:
                await session.execute(sa.delete(Embedding).where(Embedding.id.in_(orphan_rows)))
                await session.execute(sa.delete(Chunk).where(Chunk.knowledge_id.in_(dead_items)))
                await session.execute(sa.delete(Knowledge).where(Knowledge.id.in_(purge)))
                for kb_id in dead_kbs:
                    await purge_kb_rows(session, kb_id)
            live = set((await session.execute(sa.select(Knowledge.id).where(Knowledge.deleted_at.is_(None)))).scalars())
            kbs = {kb.id: kb.owner_subject for kb in (await session.execute(sa.select(KnowledgeBase))).scalars()}
        report["orphan_directories"] = await anyio.to_thread.run_sync(self._orphan_dirs, live, kbs, dry_run)
        LOG.info("housekeeping: %s", report)
        return report

    def _orphan_dirs(self, live: set[str], kbs: dict[str, str], dry_run: bool) -> int:
        """Item directories (<owner-scope>/<kb>/<item>) without a live item, older than 1 h."""
        import shutil

        root: Path = self.storage.root
        if not root.is_dir():
            return 0
        found = 0
        now = time.time()
        for owner_dir in root.iterdir():
            if not owner_dir.is_dir():
                continue
            for kb_dir in owner_dir.iterdir():
                if not kb_dir.is_dir():
                    continue
                for item_dir in kb_dir.iterdir():
                    if item_dir.is_dir() and item_dir.name not in live and now - item_dir.stat().st_mtime > ORPHAN_MIN_AGE:
                        found += 1
                        if not dry_run:
                            shutil.rmtree(item_dir, ignore_errors=True)
        return found

    # -- staleness and conflicts -----------------------------------------------------------------------
    async def staleness(self, ctx: AuthContext, kb_ref: str, stale_after_days: int | None = None,
                        expiring_within_days: int = 30, never_retrieved_grace_days: int = 14) -> dict:
        now = utcnow()
        async with async_session_scope(commit=False) as session:
            kb = await self.kbs.load(session, ctx, kb_ref)
            stale_after = int(stale_after_days or (kb.freshness_config or {}).get("stale_after_days") or 180)
            items = (await session.execute(sa.select(Knowledge).where(
                Knowledge.deleted_at.is_(None), access.items_clause(Knowledge, ctx, [kb.id])))).scalars().all()
            superseded = set((await session.execute(sa.select(KnowledgeConflict.other_knowledge_id).where(
                KnowledgeConflict.knowledge_base_id == kb.id, KnowledgeConflict.status == "open"))).scalars())
        out, summary = [], {}
        for item in items:
            reasons = []
            updated = item.processed_at or item.updated_at
            if updated and (now - updated).days >= stale_after:
                reasons.append("not_updated")
            if item.retrieval_count == 0 and item.processed_at and \
                    (now - item.processed_at).days >= never_retrieved_grace_days:
                reasons.append("never_retrieved")
            if item.expires_at and now < item.expires_at <= now + datetime.timedelta(days=expiring_within_days):
                reasons.append("expiring_soon")
            if (item.expires_at and item.expires_at <= now) or (item.valid_until and item.valid_until <= now):
                reasons.append("expired")
            if item.deprecated_at:
                reasons.append("deprecated")
            if item.id in superseded:
                reasons.append("possibly_superseded")
            for reason in reasons:
                summary[reason] = summary.get(reason, 0) + 1
            if reasons:
                out.append({"knowledge_id": item.id, "title": item.title, "reasons": reasons,
                            "retrieval_count": item.retrieval_count,
                            "last_retrieved_at": item.last_retrieved_at.isoformat() if item.last_retrieved_at else None,
                            "updated_at": updated.isoformat() if updated else None})
        return {"documents": len(items), "flagged": len(out), "summary": summary, "stale_after_days": stale_after,
                "items": out}

    async def _readable_titles(self, session, ctx, conflicts) -> dict[str, str]:
        ids = {c.knowledge_id for c in conflicts} | {c.other_knowledge_id for c in conflicts}
        if not ids:
            return {}
        kb_ids = list({c.knowledge_base_id for c in conflicts})
        rows = (await session.execute(sa.select(Knowledge.id, Knowledge.title).where(
            Knowledge.id.in_(ids), Knowledge.deleted_at.is_(None), access.items_clause(Knowledge, ctx, kb_ids)))).all()
        return dict(rows)

    async def conflicts(self, ctx: AuthContext, kb_ref: str, status: str = "open",
                        knowledge_id: str | None = None) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            kb = await self.kbs.load(session, ctx, kb_ref)
            query = sa.select(KnowledgeConflict).where(KnowledgeConflict.knowledge_base_id == kb.id)
            if status and status != "all":
                query = query.where(KnowledgeConflict.status == status)
            if knowledge_id:
                query = query.where(sa.or_(KnowledgeConflict.knowledge_id == knowledge_id,
                                           KnowledgeConflict.other_knowledge_id == knowledge_id))
            rows = (await session.execute(query.order_by(KnowledgeConflict.created_at.desc()))).scalars().all()
            titles = await self._readable_titles(session, ctx, rows)
        # a conflict is shown only to someone who can read both items
        return [conflict_dict(c, titles) for c in rows if c.knowledge_id in titles and c.other_knowledge_id in titles]

    async def review(self, ctx: AuthContext, conflict_id: str, body: dict) -> dict:
        status = body.get("status") or ("resolved" if body.get("deprecate_older") else None)
        if status not in ("open", "dismissed", "resolved"):
            raise Invalid("status must be open, dismissed or resolved")
        async with async_session_scope() as session:
            conflict = await session.get(KnowledgeConflict, conflict_id, with_for_update=True)
            if conflict is None:
                raise NotFound("conflict not found")
            kb = await self.kbs.load(session, ctx, conflict.knowledge_base_id)
            titles = await self._readable_titles(session, ctx, [conflict])
            if conflict.knowledge_id not in titles or conflict.other_knowledge_id not in titles:
                raise NotFound("conflict not found")
            older = await session.get(Knowledge, conflict.other_knowledge_id)
            if not access.can_manage_item(ctx, older, kb):
                raise Forbidden("only the older item's owner, the KB owner or an admin reviews this conflict")
            conflict.status, conflict.updated_at = status, utcnow()
            if body.get("deprecate_older"):
                conflict.status = "resolved"
                older.deprecated_at = utcnow()
                await session.execute(sa.update(Embedding).where(Embedding.knowledge_id == older.id)
                                      .values(is_deprecated=True))
            return conflict_dict(conflict, titles)
