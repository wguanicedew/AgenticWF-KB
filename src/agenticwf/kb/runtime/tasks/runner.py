"""
TaskRunner: the worker loops (docs/ingestion.md, Tasks and workers).

Runs in the API process (`KB_RUN_WORKERS=true`, the default with KB_WORKER_PROCESSES=0) or in
`agenticwf-kb-worker` processes. One loop per queue; each runs up to `[workers] concurrency`
tasks at once. A handler is `async def handler(ctx: TaskContext)`; it raises `Permanent` for a
failure that a retry cannot fix (a parse error), anything else is retried.

Each run is a span (`task <type>`) linked to the request that queued the task (for a periodic job,
the scheduler's `schedule <name>` span), with the run's model calls inside it
(agenticwf.kb.core.common.telemetry).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agenticwf.kb.core.common import telemetry
from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger, log_context
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.runtime.tasks import queue as q

LOG = get_logger(__name__)


class Permanent(Exception):
    """A task failure that retrying cannot fix."""


@dataclass
class TaskContext:
    task: dict
    services: Any                        # KBServices

    @property
    def payload(self) -> dict:
        return self.task["payload"]


Handler = Callable[[TaskContext], Awaitable[None]]


@dataclass(frozen=True)
class ScheduleSpec:
    """A periodic job: every `interval` seconds one `task_type` task is enqueued, by whichever
    process claims the slot in kb_schedules first (docs/ingestion.md, Freshness)."""

    name: str
    interval: int
    task_type: str
    queue: str = "postprocess"
FinalFailure = Callable[[TaskContext, str], Awaitable[None]]


@dataclass
class HandlerSpec:
    run: Handler
    timeout: float
    on_final_failure: FinalFailure | None = None


@dataclass
class RunnerSettings:
    queues: tuple[str, ...] = q.QUEUES
    concurrency: int = 3
    poll_seconds: float = 1.0
    scheduler_tick: float = 15.0

    @classmethod
    def from_core_config(cls) -> "RunnerSettings":
        def opt(name, default):
            return config_get("workers", name) if config_has_option("workers", name) else default
        queues = tuple(opt("queues", " ".join(q.QUEUES)).replace(",", " ").split())
        return cls(queues=queues, concurrency=int(opt("concurrency", 3)), poll_seconds=float(opt("poll_seconds", 1)),
                   scheduler_tick=float(opt("scheduler_tick", 15)))


@dataclass
class TaskRunner:
    handlers: dict[str, HandlerSpec]
    services: Any
    settings: RunnerSettings = field(default_factory=RunnerSettings)
    worker: str = field(default_factory=q.worker_id)
    schedules: tuple[ScheduleSpec, ...] = ()

    def __post_init__(self):
        self._loops: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}
        self._wake = asyncio.Event()
        self._stopping = False

    @property
    def max_runtime(self) -> float:
        return max((h.timeout for h in self.handlers.values()), default=1800) + 300

    async def start(self) -> None:
        async with async_session_scope() as session:
            recovered = await q.recover(session, self.max_runtime)
        if recovered:
            LOG.warning("requeued %d task(s) left running by a stopped process", recovered)
        q.register_waker(asyncio.get_running_loop(), self._wake)
        self._loops = [asyncio.create_task(self._loop(name), name=f"kb-queue-{name}") for name in self.settings.queues]
        if self.schedules:
            self._loops.append(asyncio.create_task(self._scheduler(), name="kb-scheduler"))
        LOG.info("task runner %s: queues %s, concurrency %d", self.worker, ", ".join(self.settings.queues),
                 self.settings.concurrency)

    async def stop(self) -> None:
        self._stopping = True
        q.unregister_waker(self._wake)
        self._wake.set()
        for loop in self._loops:
            loop.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)
        running = list(self._running.items())
        for _, task in running:
            task.cancel()
        await asyncio.gather(*(t for _, t in running), return_exceptions=True)
        if running:
            async with async_session_scope() as session:
                for task_id, _ in running:
                    await q.requeue(session, task_id)
        self._loops, self._running = [], {}

    async def run_once(self, queue: str = "default") -> bool:
        """Claim and run one task now (tests, CLI). Returns False when the queue had nothing due."""
        async with async_session_scope() as session:
            task = await q.claim(session, queue, self.worker)
        if task is None:
            return False
        await self._execute(task)
        return True

    async def drain(self, queues: tuple[str, ...] | None = None, limit: int = 1000) -> int:
        """Run due tasks until every queue is empty (tests)."""
        done = 0
        for _ in range(limit):
            ran = False
            for name in queues or self.settings.queues:
                if await self.run_once(name):
                    ran, done = True, done + 1
            if not ran:
                break
        return done

    async def tick(self) -> list[str]:
        """Enqueue every due periodic job (claimed atomically, so once across processes)."""
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from agenticwf.kb.core.orm.models.kb import Schedule

        due = []
        async with async_session_scope() as session:
            for spec in self.schedules:
                await session.execute(insert(Schedule).values(name=spec.name, interval_seconds=spec.interval)
                                      .on_conflict_do_nothing(index_elements=["name"]))
                claimed = (await session.execute(sa.text(
                    "UPDATE kb_schedules SET next_run_at = now() + make_interval(secs => :i), interval_seconds = :i "
                    "WHERE name = :n AND next_run_at <= now() RETURNING name"),
                    {"n": spec.name, "i": spec.interval})).scalar_one_or_none()
                if claimed:   # the span the task links to, as a request's would be
                    with telemetry.span(f"schedule {spec.name}", agenticwf__kb__schedule__name=spec.name,
                                        agenticwf__kb__schedule__interval=spec.interval,
                                        agenticwf__kb__task__type=spec.task_type):
                        await q.enqueue(session, spec.task_type, {}, owner_subject="system", queue=spec.queue,
                                        unique_key=f"schedule:{spec.name}", max_retry=0)
                    due.append(spec.name)
        return due

    async def _scheduler(self) -> None:
        while not self._stopping:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("scheduler tick failed")
            await asyncio.sleep(self.settings.scheduler_tick)

    async def _loop(self, queue: str) -> None:
        slots = asyncio.Semaphore(self.settings.concurrency)
        while not self._stopping:
            await slots.acquire()
            try:
                async with async_session_scope() as session:
                    task = await q.claim(session, queue, self.worker)
            except asyncio.CancelledError:
                slots.release()
                raise
            except Exception:
                LOG.exception("claiming from queue %s failed", queue)
                task = None
            if task is None:
                slots.release()
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.settings.poll_seconds)
                except asyncio.TimeoutError:
                    pass
                continue
            job = asyncio.create_task(self._execute(task), name=f"kb-task-{task['type']}")
            self._running[task["id"]] = job
            job.add_done_callback(lambda _, tid=task["id"]: (self._running.pop(tid, None), slots.release()))

    async def _execute(self, task: dict) -> None:
        with telemetry.span(f"task {task['type']}", links=[(task.get("payload") or {}).get("_trace")],
                            agenticwf__kb__task__id=task["id"], agenticwf__kb__task__type=task["type"],
                            agenticwf__kb__task__queue=task.get("queue"), agenticwf__kb__task__attempt=task.get("attempt"),
                            agenticwf__kb__knowledge_base__id=task.get("knowledge_base_id")) as current, \
                telemetry.measure_task(task["type"], task.get("queue") or "default") as measured:
            await self._run_task(task, current, measured)

    async def _run_task(self, task: dict, current, measured: dict) -> None:
        spec = self.handlers.get(task["type"])
        ctx = TaskContext(task=task, services=self.services)
        with log_context(task=task["type"], kb=task.get("knowledge_base_id") or "", sub=task["owner_subject"]):
            if spec is None:
                async with async_session_scope() as session:
                    await q.fail(session, task, f"no handler for {task['type']}", permanent=True)
                LOG.error("no handler for task type %s", task["type"])
                telemetry.fail(current, "no handler")
                measured["outcome"] = "failed"
                return
            try:
                await asyncio.wait_for(spec.run(ctx), timeout=spec.timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                permanent = isinstance(exc, Permanent)
                message = str(exc) if isinstance(exc, Permanent) else f"{type(exc).__name__}: {exc}"
                if isinstance(exc, asyncio.TimeoutError):
                    message = f"timed out after {spec.timeout:.0f}s"
                async with async_session_scope() as session:
                    final = await q.fail(session, task, message, permanent=permanent)
                LOG.warning("task %s %s failed (attempt %d): %s", task["type"], task["id"], task["attempt"], message)
                telemetry.fail(current, "timeout" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__)
                measured["outcome"] = "failed" if final else "retry"
                if final and spec.on_final_failure is not None:
                    try:
                        await spec.on_final_failure(ctx, message)
                    except Exception:
                        LOG.exception("final-failure hook of %s failed", task["type"])
                return
            async with async_session_scope() as session:
                await q.succeed(session, task["id"])
