"""
The durable task queue on kb_tasks (docs/ingestion.md, Tasks and workers).

    enqueue(session, ...)     inside the caller's transaction; `unique_key` dedups queued tasks
    claim(queue, worker)      UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING
    succeed / fail            retries with backoff min(300, 2^(n−1)·2) s; the last failure is final
    recover(worker, …)        requeue `running` tasks whose process is gone or that overran
    write_event(...)          a status change for SSE subscribers (kb_events)

Claims prefer owners with fewer running tasks, so one user's bulk upload does not hold up
everyone else's single document. Processes that did not enqueue a task find it by polling
(`[workers] poll_seconds`); the enqueuing process wakes its own loops at commit.

A task's payload carries the span context of whatever queued it (`_trace`), so the span of its run
links back to that request (agenticwf.kb.core.common.telemetry).
"""

from __future__ import annotations

import asyncio
import datetime
import os
import socket
import threading
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from agenticwf.kb.core.common import telemetry
from agenticwf.kb.core.orm.models.kb import Event, Task, utcnow

QUEUES = ("default", "postprocess", "wiki")

# --- in-process wake-up -------------------------------------------------------------------------
_WAKERS: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
_WAKERS_LOCK = threading.Lock()


def register_waker(loop: asyncio.AbstractEventLoop, wake: asyncio.Event) -> None:
    with _WAKERS_LOCK:
        _WAKERS.append((loop, wake))


def unregister_waker(wake: asyncio.Event) -> None:
    with _WAKERS_LOCK:
        _WAKERS[:] = [(l, w) for l, w in _WAKERS if w is not wake]


def wake_all() -> None:
    with _WAKERS_LOCK:
        wakers = list(_WAKERS)
    for loop, wake in wakers:
        if not loop.is_closed():
            loop.call_soon_threadsafe(wake.set)


def _wake_after_commit(session) -> None:
    sync = getattr(session, "sync_session", session)
    if not sync.info.get("kb_wake"):
        sync.info["kb_wake"] = True

        def after_commit(s):
            s.info.pop("kb_wake", None)
            wake_all()

        event.listen(sync, "after_commit", after_commit, once=True)


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


# --- enqueue ------------------------------------------------------------------------------------
async def enqueue(session, type: str, payload: dict[str, Any], owner_subject: str, owner_org: str | None = None,
                  knowledge_base_id: str | None = None, queue: str = "default", priority: int = 0,
                  unique_key: str | None = None, delay: float = 0, max_retry: int = 3) -> str | None:
    """Add a task in the caller's transaction. Returns its id, or None when a queued task with the
    same `unique_key` exists already."""
    if queue not in QUEUES:
        raise ValueError(f"unknown queue {queue!r}")
    trace = telemetry.traceparent()
    if trace:
        payload = {**payload, "_trace": trace}
    values = dict(id=None, owner_subject=owner_subject, owner_org=owner_org, knowledge_base_id=knowledge_base_id,
                  type=type, queue=queue, priority=priority, payload=payload, status="queued", attempt=0,
                  max_retry=max_retry, run_after=utcnow() + datetime.timedelta(seconds=delay),
                  unique_key=unique_key, error="", created_at=utcnow(), updated_at=utcnow())
    from agenticwf.kb.core.orm.models.kb import new_id
    values["id"] = new_id()
    stmt = insert(Task).values(**values).on_conflict_do_nothing(
        index_elements=["unique_key"], index_where=sa.text("status = 'queued' AND unique_key IS NOT NULL")
    ).returning(Task.id)
    task_id = (await session.execute(stmt)).scalar_one_or_none()
    if task_id:
        _wake_after_commit(session)
    return task_id


# --- claim and finish ----------------------------------------------------------------------------
_CLAIM = sa.text("""
UPDATE kb_tasks SET status = 'running', locked_at = now(), locked_by = :worker, attempt = attempt + 1,
       updated_at = now()
WHERE id = (
    SELECT t.id FROM kb_tasks t
    WHERE t.status = 'queued' AND t.queue = :queue AND t.run_after <= now()
    ORDER BY (SELECT count(*) FROM kb_tasks r WHERE r.status = 'running' AND r.owner_subject = t.owner_subject),
             t.priority DESC, t.run_after
    FOR UPDATE SKIP LOCKED
    LIMIT 1)
RETURNING id, type, queue, payload, owner_subject, owner_org, knowledge_base_id, attempt, max_retry
""")


async def claim(session, queue: str, worker: str) -> dict | None:
    row = (await session.execute(_CLAIM, {"queue": queue, "worker": worker})).mappings().first()
    return dict(row) if row else None


def backoff(attempt: int) -> float:
    return float(min(300, (2 ** max(0, attempt - 1)) * 2))


async def succeed(session, task_id: str) -> None:
    await session.execute(sa.update(Task).where(Task.id == task_id).values(
        status="succeeded", locked_at=None, error="", updated_at=utcnow()))


async def fail(session, task: dict, error: str, permanent: bool = False) -> bool:
    """Record a failure; returns True when it was the last attempt (the task is `failed`)."""
    final = permanent or task["attempt"] > task["max_retry"]
    values: dict[str, Any] = dict(locked_at=None, locked_by=None, error=error[:4000], updated_at=utcnow())
    if final:
        values["status"] = "failed"
    else:
        values.update(status="queued", run_after=utcnow() + datetime.timedelta(seconds=backoff(task["attempt"])))
    try:
        async with session.begin_nested():
            await session.execute(sa.update(Task).where(Task.id == task["id"]).values(**values))
    except IntegrityError:
        # a newer task with the same unique_key is queued already: this one is superseded
        await session.execute(sa.update(Task).where(Task.id == task["id"]).values(
            status="cancelled", locked_at=None, error="superseded", updated_at=utcnow()))
        return False
    return final


async def requeue(session, task_id: str) -> None:
    """Put a task interrupted by shutdown back, without counting the attempt."""
    try:
        async with session.begin_nested():
            await session.execute(sa.update(Task).where(Task.id == task_id, Task.status == "running").values(
                status="queued", attempt=Task.attempt - 1, locked_at=None, locked_by=None, updated_at=utcnow()))
    except IntegrityError:
        await session.execute(sa.update(Task).where(Task.id == task_id).values(status="cancelled",
                                                                                 error="superseded"))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def recover(session, max_runtime: float) -> int:
    """Requeue `running` tasks of dead processes on this host, and tasks running longer than
    `max_runtime` seconds anywhere."""
    host = socket.gethostname()
    rows = (await session.execute(sa.select(Task.id, Task.locked_by, Task.locked_at).where(
        Task.status == "running"))).all()
    stale = []
    cutoff = utcnow() - datetime.timedelta(seconds=max_runtime)
    for task_id, locked_by, locked_at in rows:
        owner_host, _, pid = (locked_by or "").rpartition(":")
        dead_here = owner_host == host and pid.isdigit() and not _pid_alive(int(pid))
        if dead_here or (locked_at is not None and locked_at < cutoff):
            stale.append(task_id)
    for task_id in stale:
        await requeue(session, task_id)
    return len(stale)


# --- events -------------------------------------------------------------------------------------
async def write_event(session, knowledge_base_id: str | None, owner_subject: str, payload: dict,
                      visibility: str = "all", groups: list[str] | None = None) -> None:
    """A status change for SSE subscribers. The item's visibility travels with it, so the relay
    shows the event only to those who may read the item."""
    session.add(Event(knowledge_base_id=knowledge_base_id, owner_subject=owner_subject,
                      payload={**payload, "_visibility": visibility, "_groups": list(groups or [])}))
