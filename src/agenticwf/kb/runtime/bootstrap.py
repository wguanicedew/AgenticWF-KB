"""
The one module that imports across the runtime packages to assemble a working KBThreadManager.

Call `build_thread_manager()` once per process (the API's create_app, the CLI, a worker). Hosts
then reach the KB only through threads (docs/architecture.md, Threads):

    build_thread_manager()          once per process: KBServices (system, models, storage, knowledge
      │                             bases, knowledge, search, tags, enrichment, freshness, faq) and
      │                             the manager around them
      ├─ manager.start()            sync system models; start the task runner (in-process workers)
      ├─ manager.thread(auth)       per request: a KBThread for that principal
      │    └─ thread.create_kb(), thread.add_file(), thread.search(), …
      └─ manager.system             callers without a principal: health, CLI, workers
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.runtime.agentic.engine import AgenticService
from agenticwf.kb.runtime.memory.service import MemoryService
from agenticwf.kb.runtime.websearch.service import WebSearchService
from agenticwf.kb.runtime.records.service import RecordService
from agenticwf.kb.runtime.wiki.pipeline import WikiPipeline
from agenticwf.kb.runtime.wiki.service import WikiService
from agenticwf.kb.runtime.faq.service import FAQService
from agenticwf.kb.runtime.remote import RemoteService
from agenticwf.kb.runtime.retrieval.search import SearchService
from agenticwf.kb.runtime.services import KBServices
from agenticwf.kb.runtime.services.enrichment import EnrichmentService
from agenticwf.kb.runtime.services.freshness import FreshnessService
from agenticwf.kb.runtime.services.knowledge import KnowledgeService
from agenticwf.kb.runtime.services.tags import TagService
from agenticwf.kb.runtime.services.knowledge_bases import KnowledgeBaseService
from agenticwf.kb.runtime.services.prompts import PromptService
from agenticwf.kb.runtime.services.sites import SiteService
from agenticwf.kb.runtime.services.models import ModelService
from agenticwf.kb.runtime.services.system import SystemService
from agenticwf.kb.runtime.storage import LocalStorage
from agenticwf.kb.runtime.tasks.runner import RunnerSettings
from agenticwf.kb.runtime.thread.manager import KBThreadManager

LOG = get_logger(__name__)


@dataclass
class RuntimeConfig:
    """Runtime settings, normally read from kb.cfg and the container's environment."""

    max_threads_per_owner: int | None = None
    run_workers: bool = True          # task loops in this process (KB_RUN_WORKERS; false beside kb-worker)
    runner: RunnerSettings | None = None

    @classmethod
    def from_core_config(cls) -> "RuntimeConfig":
        """
        Sections:
            [runtime]  max_threads_per_owner (requests one user runs at once; empty: no limit)
            [workers]  queues, concurrency, poll_seconds
        Environment: KB_RUN_WORKERS (default true)
        """
        limit = None
        if config_has_option("runtime", "max_threads_per_owner"):
            value = config_get("runtime", "max_threads_per_owner").strip()
            limit = int(value) if value else None
        run = os.environ.get("KB_RUN_WORKERS", "true").strip().lower() in ("1", "true", "yes", "on")
        return cls(max_threads_per_owner=limit, run_workers=run, runner=RunnerSettings.from_core_config())


@dataclass
class EngineOptions:
    """Collaborators a host or a test passes in instead of the defaults."""

    system: SystemService | None = None
    models: ModelService | None = None
    storage: LocalStorage | None = None


def build_services(options: EngineOptions | None = None) -> KBServices:
    options = options or EngineOptions()
    models = options.models or ModelService()
    storage = options.storage or LocalStorage()
    kbs = KnowledgeBaseService(models, storage)
    knowledge = KnowledgeService(models, storage, kbs)
    tags = TagService(kbs)
    enrichment = EnrichmentService(models, storage)
    knowledge.tags, knowledge.postprocess = tags, enrichment
    remote = RemoteService(models.secrets if hasattr(models, "_secrets") else None)
    search = SearchService(models, kbs, remote)
    services = KBServices(system=options.system or SystemService(), models=models, storage=storage,
                      knowledge_bases=kbs, knowledge=knowledge, search=search, tags=tags,
                      enrichment=enrichment, freshness=FreshnessService(storage, kbs, knowledge),
                      faq=FAQService(models, kbs, search))
    services.agentic = AgenticService(services)
    services.records = RecordService(models, kbs, storage)
    services.wiki = WikiService(models, kbs)
    services.wiki_pipeline = WikiPipeline(services.wiki, models, kbs)
    services.wiki.enrichment = enrichment
    enrichment.wiki = knowledge.wiki = services.wiki
    services.memory = enrichment.memory = knowledge.memory = kbs.memory = MemoryService(models)
    services.remote = remote
    services.sites = SiteService(kbs, knowledge)
    services.prompts = PromptService()
    services.web = WebSearchService(models.secrets if hasattr(models, "_secrets") else None)
    return services


def build_thread_manager(config: RuntimeConfig | None = None, options: EngineOptions | None = None) -> KBThreadManager:
    config = config or RuntimeConfig.from_core_config()
    return KBThreadManager(build_services(options), max_threads_per_owner=config.max_threads_per_owner,
                           run_workers=config.run_workers, runner_settings=config.runner)
