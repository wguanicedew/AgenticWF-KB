"""
The KB API application (docs/architecture.md, docs/api.md).

    uvicorn --factory agenticwf.kb.api.app:create_app      the mode comes from KB_AUTH_MODE

Every route lives under KB_BASE_PATH (/kb). Whatever the run mode, the request reaches the
routes through `AuthContextMiddleware`, which verifies a signed AuthContext and puts it in a
ContextVar; only the signer in front differs (docs/run-modes.md):

    standalone   StandaloneAuth   (the local token)        signs with the internal secret
    authsvc      Caddy + authsvc  (Logto token / session)  signs with the shared secret
    service      ServicePrincipal (X-KB-Principal)          signs with the internal secret

Behind the routes, the KBThreadManager built here (agenticwf.kb.runtime.bootstrap) is the only way into the
runtime: routes open a KBThread for the principal; health checks use the manager's SystemThread.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from agenticwf.auth.client import AuthContextMiddleware

from agenticwf.kb.api import errors
from agenticwf.kb.api.routes import (agentic, faq, knowledge, mcp, memory, models, prompts, quality, records, remotes,
                                     system, web_search, wiki)
from agenticwf.kb.api.service import ServicePrincipal
from agenticwf.kb.api.settings import Settings
from agenticwf.kb.api.standalone import StandaloneAuth
from agenticwf.kb.core.common import telemetry
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import dispose_async_engine
from agenticwf.kb.runtime.bootstrap import build_thread_manager
from agenticwf.kb.runtime.thread.manager import KBThreadManager

LOG = get_logger(__name__)


def create_app(settings: Settings | None = None, manager: KBThreadManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # Before any httpx client exists; off unless OTEL_EXPORTER_OTLP_ENDPOINT (docs/deployment.md).
    telemetry.setup("agenticwf-kb-service" if settings.mode == "service" else "agenticwf-kb")
    manager = manager or build_thread_manager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOG.info("KB API starting (%s mode, %s)", settings.mode, settings.base_path)
        await manager.start()
        yield
        await manager.shutdown()
        await dispose_async_engine()

    app = FastAPI(title="AgenticWF knowledge service", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager
    manager.relay_events = settings.mode != "service"     # only the main `kb` process relays events
    remote = getattr(manager.services, "remote", None)
    if remote is not None:                                 # [remote.*] serves the standalone user only
        remote.bind(settings.mode, settings.local_user)
    errors.install(app)

    routes = APIRouter(prefix=settings.base_path)
    routes.include_router(system.router)
    routes.include_router(models.router)
    routes.include_router(quality.router)      # first: PUT /knowledge/tags before PUT /knowledge/{id}
    routes.include_router(knowledge.router)
    routes.include_router(faq.router)
    routes.include_router(agentic.router)
    routes.include_router(records.router)
    routes.include_router(wiki.router)
    routes.include_router(memory.router)
    routes.include_router(web_search.router)
    routes.include_router(remotes.router)
    routes.include_router(prompts.router)
    routes.include_router(mcp.router)            # /kb/mcp (Streamable HTTP)
    app.include_router(routes)

    # Starlette runs the last added middleware first: the mode's front, then the verification.
    app.add_middleware(AuthContextMiddleware, secret=settings.context_secret,
                       bind_request=settings.bind_request if settings.mode == "authsvc" else True)
    if settings.mode == "standalone":
        app.add_middleware(StandaloneAuth, settings=settings)
    elif settings.mode == "service":
        app.add_middleware(ServicePrincipal, settings=settings)
    telemetry.instrument_app(app)   # outermost: a span per request, joining the caller's trace
    return app
