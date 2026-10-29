"""
EventRelay: status changes from kb_events to SSE subscribers (docs/architecture.md, Streaming).

Workers (in this process or in kb-worker processes) write rows to kb_events in the transaction
of the status change. The main `kb` process relays them: it reads new rows every 0.5 s, deletes
what it relayed, and hands each event to the subscribers who may read the item (the KB is
visible to them and the item's own visibility allows it). The service listener never relays,
so two processes never race on the delete.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import sqlalchemy as sa

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import Event, KnowledgeBase
from agenticwf.kb.runtime import access

LOG = get_logger(__name__)

POLL_SECONDS = 0.5
BATCH = 500
QUEUE_SIZE = 100
KB_CACHE_SECONDS = 30


@dataclass(eq=False)
class Subscription:
    ctx: AuthContext
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(QUEUE_SIZE))
    kb_ids: set[str] = field(default_factory=set)
    kbs_at: float = 0.0


class EventRelay:
    def __init__(self):
        self._subscribers: set[Subscription] = set()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="kb-event-relay")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def subscribe(self, ctx: AuthContext) -> Subscription:
        sub = Subscription(ctx)
        self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subscribers.discard(sub)

    async def _refresh(self, sub: Subscription) -> None:
        if time.monotonic() - sub.kbs_at < KB_CACHE_SECONDS:
            return
        async with async_session_scope(commit=False) as session:
            sub.kb_ids = set((await session.execute(sa.select(KnowledgeBase.id).where(
                KnowledgeBase.deleted_at.is_(None), access.visible_clause(KnowledgeBase, sub.ctx)))).scalars())
        sub.kbs_at = time.monotonic()

    @staticmethod
    def _readable(sub: Subscription, row: Event) -> bool:
        if row.knowledge_base_id not in sub.kb_ids:
            return False
        payload = row.payload or {}
        visibility = payload.get("_visibility", "all")
        if row.owner_subject == sub.ctx.subject or visibility == "all":
            return True
        return visibility == "groups" and bool(set(payload.get("_groups") or ()) & set(access.principal_groups(sub.ctx)))

    async def relay_once(self) -> int:
        async with async_session_scope() as session:
            rows = list((await session.execute(sa.select(Event).order_by(Event.id).limit(BATCH)
                                               .with_for_update(skip_locked=True))).scalars())
            if rows:
                await session.execute(sa.delete(Event).where(Event.id.in_([r.id for r in rows])))
        if not rows:
            return 0
        for sub in list(self._subscribers):
            try:
                await self._refresh(sub)
            except Exception:
                LOG.debug("cannot refresh a subscriber's KBs", exc_info=True)
                continue
            for row in rows:
                if self._readable(sub, row):
                    payload = {k: v for k, v in (row.payload or {}).items() if not k.startswith("_")}
                    payload.update(knowledge_base_id=row.knowledge_base_id, at=row.at.isoformat())
                    try:
                        sub.queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass       # a slow subscriber loses events, never the relay
        return len(rows)

    async def _run(self) -> None:
        while True:
            try:
                await self.relay_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("event relay failed")
            await asyncio.sleep(POLL_SECONDS)
