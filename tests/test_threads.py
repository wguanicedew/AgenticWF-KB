"""KBThreadManager / KBThread: the only way from a host into the runtime."""

import ast
import asyncio
from pathlib import Path

import pytest

from conftest import user

SRC = Path(__file__).resolve().parents[1] / "src"
# What hosts (kb_api) may import from the runtime: the bootstrap, the threads, the errors.
ALLOWED_RUNTIME_IMPORTS = {"kb_runtime.bootstrap", "kb_runtime.thread.manager", "kb_runtime.thread.thread",
                           "kb_runtime.errors"}


def runtime_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kb_runtime"):
            names = {f"{node.module}.{a.name}" for a in node.names} if node.module == "kb_runtime" else {node.module}
            found |= names
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith("kb_runtime")}
    return found


def test_hosts_reach_the_runtime_only_through_threads():
    offenders = {}
    for path in (SRC / "kb_api").rglob("*.py"):
        bad = runtime_imports(path) - ALLOWED_RUNTIME_IMPORTS
        if bad:
            offenders[str(path.relative_to(SRC))] = sorted(bad)
    assert not offenders, f"kb_api must use KBThread / SystemThread, not runtime services: {offenders}"


def manager(**kwargs):
    from kb_runtime.bootstrap import RuntimeConfig, build_thread_manager

    return build_thread_manager(RuntimeConfig(**kwargs))


def test_threads_are_owner_scoped_and_closed():
    async def scenario():
        m = manager()
        alice, bob = user(subject="alice"), user(subject="bob")
        async with m.thread(alice) as thread:
            assert thread.owner == "alice" and m.live_threads() == 1
            assert await m.get_thread(thread.thread_id, alice) is thread
            assert await m.get_thread(thread.thread_id, bob) is None      # not revealed to others
            assert [t.thread_id for t in await m.list_threads(alice)] == [thread.thread_id]
            assert await m.list_threads(bob) == []
        assert thread.closed and m.live_threads() == 0

    asyncio.run(scenario())


def test_owner_is_the_user_not_the_token():
    async def scenario():
        m = manager()
        personal, in_org = user(subject="alice"), user(subject="alice", org="eic")
        async with m.thread(personal) as thread:
            assert await m.get_thread(thread.thread_id, in_org) is thread

    asyncio.run(scenario())


def test_per_owner_limit():
    from kb_runtime.errors import Busy

    async def scenario():
        m = manager(max_threads_per_owner=1)
        async with m.thread(user(subject="alice")):
            with pytest.raises(Busy):
                await m.open_thread(user(subject="alice"))
            async with m.thread(user(subject="bob")):      # other users are not affected
                pass
        async with m.thread(user(subject="alice")):         # free again once closed
            pass

    asyncio.run(scenario())


def test_shutdown_closes_open_threads():
    async def scenario():
        m = manager()
        await m.start()
        thread = await m.open_thread(user())
        report = await m.shutdown()
        assert report.closed == [thread.thread_id] and thread.closed and not m.started

    asyncio.run(scenario())


def test_busy_is_429_in_the_envelope(database):
    from fastapi.testclient import TestClient

    from kb_api.app import create_app
    from kb_api.settings import Settings
    from kb_runtime.bootstrap import RuntimeConfig, build_thread_manager
    from conftest import SECRET, signed

    m = build_thread_manager(RuntimeConfig(max_threads_per_owner=0))
    with TestClient(create_app(Settings(mode="authsvc", context_secret=SECRET), manager=m),
                    base_url="http://kb.test") as client:
        response = client.get("/kb/api/v1/me", headers=signed(user(), "GET", "/kb/api/v1/me"))
    assert response.status_code == 429 and response.json()["error"]["code"] == 1000
