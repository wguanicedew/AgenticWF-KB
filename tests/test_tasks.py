"""The task queue: enqueue, unique keys, claim, retries with backoff, permanent failures,
recovery, owner fairness (docs/ingestion.md, Tasks and workers)."""

import asyncio
import datetime

import sqlalchemy as sa

from agenticwf.kb.core.orm.base.session import async_session_scope, dispose_async_engine
from agenticwf.kb.core.orm.models.kb import Task, utcnow
from agenticwf.kb.runtime.tasks import queue as q
from agenticwf.kb.runtime.tasks.runner import HandlerSpec, Permanent, RunnerSettings, TaskRunner


def run(coro):
    async def wrapped():
        try:
            return await coro
        finally:
            await dispose_async_engine()
    return asyncio.run(wrapped())


async def _enqueue(type="t:ok", owner="alice", unique_key=None, priority=0):
    async with async_session_scope() as session:
        return await q.enqueue(session, type, {"n": 1}, owner_subject=owner, unique_key=unique_key, priority=priority)


async def _status(task_id):
    async with async_session_scope(commit=False) as session:
        return (await session.get(Task, task_id))


def test_unique_key_dedups_queued_tasks(database):
    async def scenario():
        first = await _enqueue(unique_key="process:1")
        second = await _enqueue(unique_key="process:1")
        return first, second
    first, second = run(scenario())
    assert first and second is None


def test_success_retry_and_permanent_failure(database):
    calls = {"flaky": 0}

    async def ok(ctx):
        pass

    async def flaky(ctx):
        calls["flaky"] += 1
        raise RuntimeError("temporary")

    async def broken(ctx):
        raise Permanent("cannot parse")

    failed = []

    async def on_final(ctx, error):
        failed.append((ctx.task["type"], error))

    handlers = {"t:ok": HandlerSpec(ok, 5), "t:flaky": HandlerSpec(flaky, 5, on_final),
                "t:broken": HandlerSpec(broken, 5, on_final)}

    async def scenario():
        ids = [await _enqueue("t:ok"), await _enqueue("t:flaky"), await _enqueue("t:broken")]
        runner = TaskRunner(handlers, services=None, settings=RunnerSettings(queues=("default",)))
        await runner.drain()
        rows = [await _status(i) for i in ids]
        return rows

    ok_row, flaky_row, broken_row = run(scenario())
    assert ok_row.status == "succeeded"
    assert flaky_row.status == "queued" and flaky_row.attempt == 1 and flaky_row.run_after > utcnow()
    assert "temporary" in flaky_row.error
    assert broken_row.status == "failed" and broken_row.error == "cannot parse"
    assert failed == [("t:broken", "cannot parse")]


def test_last_retry_is_final(database):
    async def flaky(ctx):
        raise RuntimeError("still down")

    seen = []

    async def on_final(ctx, error):
        seen.append(error)

    async def scenario():
        async with async_session_scope() as session:
            task_id = await q.enqueue(session, "t:flaky", {}, owner_subject="alice", max_retry=1)
        runner = TaskRunner({"t:flaky": HandlerSpec(flaky, 5, on_final)}, None, RunnerSettings(queues=("default",)))
        for _ in range(2):
            async with async_session_scope() as session:   # make the retry due now
                await session.execute(sa.update(Task).where(Task.id == task_id).values(run_after=utcnow()))
            await runner.run_once()
        return await _status(task_id)

    row = run(scenario())
    assert row.status == "failed" and row.attempt == 2 and seen == ["RuntimeError: still down"]


def test_backoff_grows_and_caps():
    assert [q.backoff(n) for n in (1, 2, 3, 4)] == [2, 4, 8, 16] and q.backoff(20) == 300


def test_recovery_of_tasks_left_running(database):
    async def scenario():
        task_id = await _enqueue()
        async with async_session_scope() as session:
            await session.execute(sa.update(Task).where(Task.id == task_id).values(
                status="running", locked_by=f"{__import__('socket').gethostname()}:999999",
                locked_at=utcnow() - datetime.timedelta(seconds=10), attempt=1))
        async with async_session_scope() as session:
            recovered = await q.recover(session, max_runtime=3600)
        return recovered, await _status(task_id)

    recovered, row = run(scenario())
    assert recovered == 1 and row.status == "queued" and row.attempt == 0


def test_owner_fairness(database):
    """With alice already running a task, bob's queued task is claimed before alice's next one."""
    async def scenario():
        await _enqueue(owner="alice")
        await _enqueue(owner="alice")
        async with async_session_scope() as session:
            first = await q.claim(session, "default", "w:1")
        await _enqueue(owner="bob")
        async with async_session_scope() as session:
            second = await q.claim(session, "default", "w:1")
        return first, second

    first, second = run(scenario())
    assert first["owner_subject"] == "alice" and second["owner_subject"] == "bob"
