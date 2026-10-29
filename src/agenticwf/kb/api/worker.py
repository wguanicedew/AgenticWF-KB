"""agenticwf-kb-worker: a worker process beside the API (docs/ingestion.md, Tasks and workers).

Runs the task loops (`[workers] queues`, `concurrency`) on this process's own KBThreadManager
until SIGTERM, then puts the tasks it was running back in the queue. The API process then runs
no loops (`KB_RUN_WORKERS=false`, set by the entrypoint when KB_WORKER_PROCESSES > 0).
"""

from __future__ import annotations

import asyncio
import signal

from agenticwf.kb.core.common.logging import get_logger

LOG = get_logger(__name__)


async def run() -> None:
    from agenticwf.kb.core.orm.base.session import dispose_async_engine
    from agenticwf.kb.core.common import telemetry
    from agenticwf.kb.runtime.bootstrap import RuntimeConfig, build_thread_manager

    telemetry.setup("agenticwf-kb-worker")
    config = RuntimeConfig.from_core_config()
    config.run_workers = True
    manager = build_thread_manager(config)
    await manager.system.check_database()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await manager.start(sync_models=False)      # the API process syncs system models
    LOG.info("KB worker ready")
    await stop.wait()
    await manager.shutdown()
    await dispose_async_engine()
    LOG.info("KB worker stopped")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
