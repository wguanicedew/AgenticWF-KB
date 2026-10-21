"""
KBServices: the process-wide services behind every KBThread.

kb_runtime.bootstrap builds one KBServices per process; KBThread and SystemThread are the only
callers. Hosts (kb_api routes, the CLI, workers) never use a service directly: they go through a
thread from KBThreadManager (docs/architecture.md, Threads).

    system            database health, extensions, deployment info
    models            system models from kb.cfg, users' models, model clients
    storage           originals, parsed markdown, images (local files)
    knowledge_bases   KBs: create, list, visibility, delete
    knowledge         items: add (file / url / manual), process, read, change; chunks
    search            hybrid search (pgvector + BM25), fusion, rerank
    tags              per-KB tags and their relations
    enrichment        subtasks after ingestion: summary, questions, VLM, conflicts
    freshness         expiry, URL refresh, housekeeping, staleness, conflict review
    faq               entries of FAQ knowledge bases: upsert, search, import, export
    agentic           agentic search: presets, the tool loop (kb_runtime.agentic)
    records           record types and records of records KBs (phase 3, kb_runtime.records)
    wiki              the LLM wiki: pages, revisions, folders, finalize (phase 3, kb_runtime.wiki)
    wiki_pipeline     wiki ingest map-reduce, communities, global search
    memory            temporal memory: episodes, entities, bi-temporal facts (phase 3)
    web               web search and fetch: providers, results, pages as markdown

Later phases add records, wiki, memory (phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kb_runtime.faq.service import FAQService
from kb_runtime.retrieval.search import SearchService
from kb_runtime.services.enrichment import EnrichmentService
from kb_runtime.services.freshness import FreshnessService
from kb_runtime.services.tags import TagService
from kb_runtime.services.knowledge import KnowledgeService
from kb_runtime.services.knowledge_bases import KnowledgeBaseService
from kb_runtime.services.models import ModelService
from kb_runtime.services.system import SystemService
from kb_runtime.storage import LocalStorage


@dataclass
class KBServices:
    system: SystemService
    models: ModelService
    storage: LocalStorage
    knowledge_bases: KnowledgeBaseService
    knowledge: KnowledgeService
    search: SearchService
    tags: TagService
    enrichment: EnrichmentService
    freshness: FreshnessService
    faq: FAQService
    agentic: Any = field(default=None, repr=False)        # AgenticService, set by build_services
    records: Any = field(default=None, repr=False)        # RecordService (phase 3)
    wiki: Any = field(default=None, repr=False)           # WikiService (phase 3)
    wiki_pipeline: Any = field(default=None, repr=False)  # WikiPipeline (phase 3)
    memory: Any = field(default=None, repr=False)         # MemoryService (phase 3)
    web: Any = field(default=None, repr=False)            # WebSearchService

    def task_handlers(self) -> dict:
        handlers = {**self.knowledge.handlers(), **self.enrichment.handlers(), **self.freshness.handlers()}
        for extra in (self.records, self.wiki_pipeline, self.memory):
            if extra is not None:
                handlers.update(extra.handlers())
        return handlers

    def schedules(self) -> tuple:
        from kb_runtime.records.service import schedules as record_schedules
        from kb_runtime.services.freshness import schedules

        return schedules() + (record_schedules() if self.records is not None else ())
