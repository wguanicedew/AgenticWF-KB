"""
The KB API application (docs/architecture.md, docs/api.md).

    uvicorn --factory kb_api.app:create_app      the mode comes from KB_AUTH_MODE

Every route lives under KB_BASE_PATH (/kb). Whatever the run mode, the request reaches the
routes through `AuthContextMiddleware`, which verifies a signed AuthContext and puts it in a
ContextVar; only the signer in front differs (docs/run-modes.md):

    standalone   StandaloneAuth   (the local token)        signs with the internal secret
    authsvc      Caddy + authsvc  (Logto token / session)  signs with the shared secret
    service      ServicePrincipal (X-KB-Principal)          signs with the internal secret

Behind the routes, the KBThreadManager built here (kb_runtime.bootstrap) is the only way into the
runtime: routes open a KBThread for the principal; health checks use the manager's SystemThread.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from authsvc.client import AuthContextMiddleware

from kb_api import errors
from kb_api.routes import agentic, faq, knowledge, mcp, memory, models, quality, records, system, web_search, wiki
from kb_api.service import ServicePrincipal
from kb_api.settings import Settings
from kb_api.standalone import StandaloneAuth
from kb_core.common.logging import get_logger
from kb_core.orm.base.session import dispose_async_engine
from kb_runtime.bootstrap import build_thread_manager
from kb_runtime.thread.manager import KBThreadManager

LOG = get_logger(__name__)


def create_app(settings: Settings | None = None, manager: KBThreadManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
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
    routes.include_router(mcp.router)            # /kb/mcp (Streamable HTTP)
    app.include_router(routes)

    # Starlette runs the last added middleware first: the mode's front, then the verification.
    app.add_middleware(AuthContextMiddleware, secret=settings.context_secret,
                       bind_request=settings.bind_request if settings.mode == "authsvc" else True)
    if settings.mode == "standalone":
        app.add_middleware(StandaloneAuth, settings=settings)
    elif settings.mode == "service":
        app.add_middleware(ServicePrincipal, settings=settings)
    return app
