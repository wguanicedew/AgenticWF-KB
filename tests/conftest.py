import os
import tempfile
import time
from pathlib import Path

# kb_core.common.config is read at import time (kb_core.orm, logging), so write the test
# configuration before importing anything from src/. The tests need PostgreSQL with the KB's
# extensions (the CI service is the ParadeDB image):
#   KB_TEST_DB_URL=postgresql+psycopg://postgres:test@127.0.0.1:5432/postgres pytest -q
# A local server without pg_search can narrow the extension check (never in CI):
#   KB_TEST_REQUIRED_EXTENSIONS="vector pg_trgm"
_TMP = Path(tempfile.mkdtemp(prefix="kb-tests-"))
_DB_URL = os.environ.get("KB_TEST_DB_URL", "")
_EXTENSIONS = os.environ.get("KB_TEST_REQUIRED_EXTENSIONS", "vector pg_trgm pg_search")
(_TMP / "kb.cfg").write_text(
    "[database]\ndefault = %s\nrequired_extensions = %s\n\n[logging]\nlevel = WARNING\n\n"
    "[features]\nfaq = true\nrecords = true\nwiki = true\nmemory = true\nweb = true\n\n[storage]\nroot = %s\n\n[security]\nssrf_whitelist = 127.0.0.0/8\n\n"
    "[models]\ndefault_embedding = test-hash\nmax_retries = 0\n\n"
    "[models.embedding.test-hash]\nprovider = hash\nmodel = hash\ndimension = 64\n"
    % (_DB_URL.replace("%", "%%") or "postgresql+psycopg://unset/unset", _EXTENSIONS, _TMP / "files"))
os.environ["AGENTICWF_KB_CONFIG"] = str(_TMP / "kb.cfg")
os.environ["KB_DATA_ROOT"] = str(_TMP / "data")
os.environ["KB_TOKEN_FILE"] = str(_TMP / "data" / "secrets" / "token")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from authsvc.context import AuthContext, encode_context  # noqa: E402

SECRET = b"s" * 48


def user(subject="user_alice", scopes=("app:access", "knowledge:read"), org=None, roles=(), groups=None, ttl=3600,
         client_id=None):
    return AuthContext(subject=subject, user_id=None if client_id == subject else subject, client_id=client_id,
                       organization_id=org, roles=frozenset(roles), scopes=frozenset(scopes),
                       external_identity={"provider": "indigo-iam", "groups": list(groups)} if groups else None,
                       expires_at=int(time.time()) + ttl)


def signed(ctx, method, path, secret=SECRET, ttl=60):
    """The headers Caddy copies from authsvc's /verify response."""
    payload, sig = encode_context(ctx, secret, ttl, method, path)
    return {"X-Auth-Context": payload, "X-Auth-Context-Sig": sig}


@pytest.fixture(scope="session")
def database():
    if not _DB_URL:
        pytest.skip("KB_TEST_DB_URL is not set (PostgreSQL with vector, pg_trgm, pg_search)")
    from kb_api.db import init_db

    init_db()


TABLES = ("kb_embeddings", "kb_chunks", "kb_knowledge_tag_relations", "kb_knowledge_tags", "kb_knowledge_versions",
          "kb_knowledge_conflicts", "kb_knowledges", "kb_knowledge_bases", "kb_models", "kb_tasks", "kb_schedules",
          "kb_events", "kb_record_types", "kb_records", "kb_record_items", "kb_record_versions",
          "kb_record_field_stats", "kb_wiki_pages", "kb_wiki_page_revisions", "kb_wiki_folders", "kb_wiki_page_issues",
          "kb_wiki_pending_ops", "kb_wiki_relations", "kb_wiki_communities", "kb_memory_episodes",
          "kb_memory_entities", "kb_memory_facts", "kb_web_search_providers")


@pytest.fixture(autouse=True)
def clean_tables(request):
    if "database" not in request.fixturenames:
        yield
        return
    request.getfixturevalue("database")
    import sqlalchemy as sa
    from kb_core.orm.base.session import get_engine

    with get_engine().begin() as conn:
        conn.execute(sa.text("TRUNCATE " + ", ".join(TABLES) + " CASCADE"))
    yield


def make_client(mode, **settings_kwargs):
    from kb_api.app import create_app
    from kb_api.settings import Settings
    from kb_runtime.bootstrap import RuntimeConfig, build_thread_manager

    settings = Settings(mode=mode, context_secret=SECRET, **settings_kwargs)
    manager = build_thread_manager(RuntimeConfig(run_workers=False))   # tests drain the queue themselves
    return TestClient(create_app(settings, manager=manager), base_url="http://kb.test")


def drain(client) -> int:
    """Run every due task to completion (the in-process workers are off in tests)."""
    import asyncio

    from kb_core.orm.base.session import dispose_async_engine

    async def run():
        try:
            return await client.app.state.manager.make_runner().drain()
        finally:
            await dispose_async_engine()

    return asyncio.run(run())


class Caller:
    """A TestClient that signs every request for `ctx` (authsvc mode), like Caddy + authsvc."""

    def __init__(self, client, ctx):
        self.client, self.ctx = client, ctx

    def request(self, method, path, **kwargs):
        headers = {**signed(self.ctx, method, path.split("?", 1)[0]), **kwargs.pop("headers", {})}
        return self.client.request(method, path, headers=headers, **kwargs)

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def put(self, path, **kw):
        return self.request("PUT", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)


@pytest.fixture
def authsvc_client(database):
    with make_client("authsvc") as client:
        yield client


@pytest.fixture
def standalone_client(database):
    with make_client("standalone") as client:
        yield client


@pytest.fixture
def service_client(database):
    from kb_api.settings import DEFAULT_SERVICE_SCOPES_CAP

    with make_client("service", required_scopes=frozenset(), service_scopes_cap=DEFAULT_SERVICE_SCOPES_CAP) as client:
        yield client


def sql_rows(statement, **params):
    """Rows of one SQL statement, outside any request (assertions on stored state)."""
    import sqlalchemy as sa

    from kb_core.orm.base.session import get_engine

    with get_engine().begin() as conn:
        return conn.execute(sa.text(statement), params).all()
