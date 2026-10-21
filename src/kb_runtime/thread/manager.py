"""
KBThreadManager: the process-wide registry of open threads, and the host's only door into the
runtime.

Built once per process by kb_runtime.bootstrap.build_thread_manager around the process's
KBServices. A host opens a thread for a principal, works through it, and closes it:

    async with manager.thread(ctx, source="standalone") as thread:
        return thread.me()

Threads belong to one owner (the user's subject): every lookup takes the caller's AuthContext
and returns None for threads owned by someone else. `max_threads_per_owner` bounds how much one
user runs at once (agentic-search runs keep their thread open for the whole run; phase 2).
`manager.system` is the SystemThread for callers without a principal.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from authsvc.context import AuthContext

from kb_core.common.logging import get_logger
from kb_runtime.errors import Busy
from kb_runtime.events import EventRelay
from kb_runtime.services import KBServices
from kb_runtime.tasks.runner import RunnerSettings, TaskRunner
from kb_runtime.thread.thread import KBThread, SystemThread, owner_of

LOG = get_logger(__name__)


@dataclass
class ThreadShutdownReport:
    closed: list[str] = field(default_factory=list)


class KBThreadManager:
    def __init__(self, services: KBServices, max_threads_per_owner: int | None = None, run_workers: bool = True,
                 runner_settings: RunnerSettings | None = None):
        self._services = services
        self._threads: dict[str, KBThread] = {}
        self._lock = asyncio.Lock()
        self.max_threads_per_owner = max_threads_per_owner
        self.run_workers = run_workers
        self.runner_settings = runner_settings or RunnerSettings()
        self.runner: TaskRunner | None = None
        self.system = SystemThread(services)
        self.started = False
        self.relay_events = True         # the main `kb` process; false for the service listener
        self.events = EventRelay()

    @property
    def services(self) -> KBServices:
        """For the runtime's own components (bootstrap, workers); hosts use threads."""
        return self._services

    # -- lifecycle -----------------------------------------------------------------------------------
    def make_runner(self) -> TaskRunner:
        """A task runner over this process's services (the API's in-process workers, a worker
        process, or a test that drains the queue)."""
        return TaskRunner(self._services.task_handlers(), self._services, self.runner_settings,
                          schedules=self._services.schedules())

    async def start(self, sync_models: bool = True) -> None:
        if sync_models:
            await self._services.models.sync_system_models()
            if self._services.records is not None:
                await self._services.records.sync_plugin_types()
            if self._services.web is not None:
                await self._services.web.sync_system_providers()
        if self.run_workers and self.runner is None:
            self.runner = self.make_runner()
            await self.runner.start()
        if self.relay_events:
            await self.events.start()
        self.started = True

    async def shutdown(self) -> ThreadShutdownReport:
        await self.events.stop()
        if self.runner is not None:
            await self.runner.stop()
            self.runner = None
        async with self._lock:
            threads = list(self._threads.values())
            self._threads.clear()
        for thread in threads:
            thread.closed = True
        self.started = False
        return ThreadShutdownReport(closed=[t.thread_id for t in threads])

    # -- threads -----------------------------------------------------------------------------------
    async def open_thread(self, auth: AuthContext, source: str = "api") -> KBThread:
        owner = owner_of(auth)
        async with self._lock:
            if self.max_threads_per_owner is not None:
                live = sum(1 for t in self._threads.values() if t.owner == owner)
                if live >= self.max_threads_per_owner:
                    raise Busy(f"at most {self.max_threads_per_owner} requests at once per user")
            thread = KBThread(auth=auth, services=self._services, source=source)
            self._threads[thread.thread_id] = thread
        return thread

    async def close_thread(self, thread: KBThread) -> None:
        thread.closed = True
        async with self._lock:
            self._threads.pop(thread.thread_id, None)

    @asynccontextmanager
    async def thread(self, auth: AuthContext, source: str = "api") -> AsyncIterator[KBThread]:
        thread = await self.open_thread(auth, source)
        try:
            yield thread
        finally:
            await self.close_thread(thread)

    async def get_thread(self, thread_id: str, auth: AuthContext) -> KBThread | None:
        async with self._lock:
            thread = self._threads.get(thread_id)
        if thread is None or thread.owner != owner_of(auth):
            return None
        return thread

    async def list_threads(self, auth: AuthContext) -> list[KBThread]:
        owner = owner_of(auth)
        async with self._lock:
            return [t for t in self._threads.values() if t.owner == owner]

    def live_threads(self) -> int:
        return len(self._threads)
