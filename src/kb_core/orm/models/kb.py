"""
The KB's tables (docs/data-model.md). All names carry the `kb_` prefix; the schema is
PostgreSQL-only (pgvector, pg_search, pg_trgm; JSONB and arrays).

Phase 1: knowledge bases, knowledge, chunks, the retrieval index, models, and the task queue
(tasks, schedules, events). Indexes that depend on configuration or data are created elsewhere:
the BM25 index by revision 0002 (and `python -m kb_api.db reindex-bm25`), one HNSW index per
embedding dimension at run time (kb_runtime.retrieval.index.ensure_vector_index).
"""

import datetime
import uuid

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer,
                        SmallInteger, String, Text, text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from kb_core.orm.base.models import ModelBase
from kb_core.orm.base.session import BASE

GIB = 1024 ** 3


def utcnow() -> datetime.datetime:
    """Now, timezone-aware (the KB's DateTime columns are timestamptz)."""
    return datetime.datetime.now(datetime.timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def _ts(**kw):
    return Column(DateTime(timezone=True), **kw)


class _Timestamps:
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class _Owned(_Timestamps):
    owner_subject = Column(String(255), nullable=False)
    owner_org = Column(String(255), nullable=True)


class _Visible:
    visibility = Column(String(16), nullable=False, default="private")
    groups = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")


def default_chunking_config() -> dict:
    return {"chunk_size": 512, "chunk_overlap": 50, "separators": ["\n\n", "\n", "。"], "strategy": "",
            "token_limit": 0, "enable_parent_child": False, "parent_chunk_size": 4096, "child_chunk_size": 384}


def default_indexing_strategy() -> dict:
    return {"vector_enabled": True, "keyword_enabled": True, "wiki_enabled": False, "graph_enabled": False}


# --- knowledge bases ----------------------------------------------------------------------------
class KnowledgeBase(BASE, ModelBase, _Owned, _Visible):
    __tablename__ = "kb_knowledge_bases"

    id = Column(String(36), primary_key=True, default=new_id)
    default_item_visibility = Column(String(16), nullable=False, default="all")
    contribute = Column(String(16), nullable=False, default="owner")
    slug = Column(String(128), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, default="")
    type = Column(String(32), nullable=False, default="document")
    chunking_config = Column(JSONB, nullable=False, default=default_chunking_config)
    embedding_model_id = Column(String(36), nullable=True)
    summary_model_id = Column(String(36), nullable=True)
    rerank_model_id = Column(String(36), nullable=True)
    indexing_strategy = Column(JSONB, nullable=False, default=default_indexing_strategy)
    retrieval_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    question_generation_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    vlm_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    image_processing_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    freshness_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    wiki_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    records_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    memory_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    is_pinned = Column(Boolean, nullable=False, default=False)
    is_temporary = Column(Boolean, nullable=False, default=False)
    storage_used = Column(BigInteger, nullable=False, default=0)
    storage_quota = Column(BigInteger, nullable=False, default=10 * GIB)
    deleted_at = _ts(nullable=True)

    __table_args__ = (
        Index("ix_kb_knowledge_bases_owner", "owner_subject"),
        Index("ix_kb_knowledge_bases_groups", "groups", postgresql_using="gin"),
        Index("uq_kb_knowledge_bases_owner_slug", "owner_subject", "slug", unique=True,
              postgresql_where=text("deleted_at IS NULL")),
    )


# --- knowledge (items) --------------------------------------------------------------------------
class Knowledge(BASE, ModelBase, _Owned, _Visible):
    __tablename__ = "kb_knowledges"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False)
    type = Column(String(32), nullable=False)              # file | url | manual | faq
    title = Column(String(1024), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    source = Column(String(2048), nullable=False, default="")
    channel = Column(String(32), nullable=False, default="api")
    parse_status = Column(String(32), nullable=False, default="pending")
    summary_status = Column(String(32), nullable=False, default="none")
    enable_status = Column(String(32), nullable=False, default="enabled")
    pending_subtasks_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=False, default="")
    file_name = Column(String(1024), nullable=False, default="")
    file_type = Column(String(32), nullable=False, default="")
    file_size = Column(BigInteger, nullable=False, default=0)
    storage_size = Column(BigInteger, nullable=False, default=0)
    file_hash = Column(String(64), nullable=False, default="")
    file_path = Column(String(2048), nullable=False, default="")
    parsed_path = Column(String(2048), nullable=False, default="")
    folder_path = Column(String(1024), nullable=False, default="")
    embedding_model_id = Column(String(36), nullable=True)
    meta = Column("metadata", JSONB, nullable=False, default=dict, server_default="{}")
    profile = Column(JSONB, nullable=False, default=dict, server_default="{}")
    citation = Column(JSONB, nullable=False, default=dict, server_default="{}")   # the source citation (revision 0005)
    processed_at = _ts(nullable=True)
    external_id = Column(String(512), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    content_hash = Column(String(64), nullable=False, default="")
    valid_from = _ts(nullable=True)
    valid_until = _ts(nullable=True)
    expires_at = _ts(nullable=True)
    deprecated_at = _ts(nullable=True)
    refresh_interval = Column(Integer, nullable=False, default=0)
    last_refreshed_at = _ts(nullable=True)
    retrieval_count = Column(BigInteger, nullable=False, default=0)
    last_retrieved_at = _ts(nullable=True)
    deleted_at = _ts(nullable=True)

    __table_args__ = (
        Index("ix_kb_knowledges_kb_hash", "knowledge_base_id", "file_hash"),
        Index("ix_kb_knowledges_kb_created", "knowledge_base_id", "created_at"),
        Index("ix_kb_knowledges_kb_owner", "knowledge_base_id", "owner_subject"),
        Index("ix_kb_knowledges_groups", "groups", postgresql_using="gin"),
        Index("ix_kb_knowledges_status", "parse_status"),
        Index("uq_kb_knowledges_external_id", "knowledge_base_id", "external_id", unique=True,
              postgresql_where=text("external_id IS NOT NULL AND deleted_at IS NULL")),
    )


class KnowledgeVersion(BASE, ModelBase):
    """A replaced version of an item (docs/ingestion.md, Versions). `stored_path` is set when the
    KB keeps previous files; kept files are never indexed."""

    __tablename__ = "kb_knowledge_versions"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_id = Column(String(36), ForeignKey("kb_knowledges.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False)
    file_name = Column(String(1024), nullable=False, default="")
    file_hash = Column(String(64), nullable=False, default="")
    file_size = Column(BigInteger, nullable=False, default=0)
    content_hash = Column(String(64), nullable=False, default="")
    stored_path = Column(String(2048), nullable=False, default="")
    replaced_by = Column(String(255), nullable=False, default="")
    replaced_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (Index("ix_kb_knowledge_versions_item", "knowledge_id", "version"),)


class KnowledgeConflict(BASE, ModelBase, _Timestamps):
    """A newer item repeating or contradicting an older one (docs/ingestion.md, Conflicts)."""

    __tablename__ = "kb_knowledge_conflicts"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), nullable=False)
    knowledge_id = Column(String(36), ForeignKey("kb_knowledges.id", ondelete="CASCADE"), nullable=False)
    other_knowledge_id = Column(String(36), ForeignKey("kb_knowledges.id", ondelete="CASCADE"), nullable=False)
    similarity = Column(Float, nullable=False, default=0.0)
    matched_chunks = Column(Integer, nullable=False, default=0)
    other_coverage = Column(Float, nullable=False, default=0.0)
    pairs = Column(JSONB, nullable=False, default=list, server_default="[]")
    verdict = Column(String(32), nullable=False, default="")
    explanation = Column(Text, nullable=False, default="")
    status = Column(String(16), nullable=False, default="open")

    __table_args__ = (
        Index("uq_kb_knowledge_conflicts_pair", "knowledge_id", "other_knowledge_id", unique=True),
        Index("ix_kb_knowledge_conflicts_kb", "knowledge_base_id", "status"),
    )


class KnowledgeTag(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_knowledge_tags"

    id = Column(String(36), primary_key=True, default=new_id)
    seq_id = Column(BigInteger, Identity(), unique=True, nullable=False)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    color = Column(String(32), nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    __table_args__ = (Index("uq_kb_knowledge_tags_name", "knowledge_base_id", "name", unique=True),)


class KnowledgeTagRelation(BASE, ModelBase):
    __tablename__ = "kb_knowledge_tag_relations"

    knowledge_id = Column(String(36), ForeignKey("kb_knowledges.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(String(36), ForeignKey("kb_knowledge_tags.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (Index("ix_kb_knowledge_tag_relations_tag", "tag_id"),)


# --- chunks ---------------------------------------------------------------------------------------
class Chunk(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_chunks"

    id = Column(String(36), primary_key=True, default=new_id)
    seq_id = Column(BigInteger, Identity(), unique=True, nullable=False)
    knowledge_base_id = Column(String(36), nullable=False)
    knowledge_id = Column(String(36), ForeignKey("kb_knowledges.id", ondelete="CASCADE"), nullable=False)
    tag_id = Column(String(36), nullable=False, default="")
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, default="")
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(32), nullable=False, default="text")
    start_at = Column(Integer, nullable=False, default=0)
    end_at = Column(Integer, nullable=False, default=0)
    pre_chunk_id = Column(String(36), nullable=False, default="")
    next_chunk_id = Column(String(36), nullable=False, default="")
    parent_chunk_id = Column(String(36), nullable=False, default="")
    context_header = Column(Text, nullable=False, default="")
    is_enabled = Column(Boolean, nullable=False, default=True)
    flags = Column(Integer, nullable=False, default=0)
    status = Column(SmallInteger, nullable=False, default=0)
    content_revision = Column(Integer, nullable=False, default=1)
    meta = Column("metadata", JSONB, nullable=False, default=dict, server_default="{}")
    image_info = Column(JSONB, nullable=False, default=list, server_default="[]")

    __table_args__ = (
        Index("ix_kb_chunks_knowledge_index", "knowledge_id", "chunk_index"),
        Index("ix_kb_chunks_kb", "knowledge_base_id"),
        Index("ix_kb_chunks_content_trgm", "content", postgresql_using="gin",
              postgresql_ops={"content": "gin_trgm_ops"}),
    )


# --- the retrieval index ----------------------------------------------------------------------------
SOURCE_CHUNK, SOURCE_QUESTION, SOURCE_SUMMARY, SOURCE_IMAGE, SOURCE_WIKI_PAGE, SOURCE_RECORD = 0, 1, 2, 3, 4, 5


class Embedding(BASE, ModelBase):
    """One row per indexed text. Visibility and freshness are copied from the item so the
    filters run inside the vector and BM25 queries."""

    __tablename__ = "kb_embeddings"

    id = Column(BigInteger, Identity(), primary_key=True)
    source_id = Column(String(128), nullable=False)
    source_type = Column(SmallInteger, nullable=False, default=SOURCE_CHUNK)
    chunk_id = Column(String(64), nullable=False, default="")
    knowledge_id = Column(String(64), nullable=False, default="")
    knowledge_base_id = Column(String(64), nullable=False, default="")
    owner_subject = Column(String(255), nullable=False, default="")
    visibility = Column(String(16), nullable=False, default="all")
    groups = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    tag_id = Column(String(64), nullable=False, default="")
    content = Column(Text, nullable=False)
    dimension = Column(Integer, nullable=False, default=0)
    embedding = Column(HALFVEC(), nullable=True)
    is_enabled = Column(Boolean, nullable=False, default=True)
    valid_from = _ts(nullable=True)
    valid_until = _ts(nullable=True)
    is_deprecated = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        Index("uq_kb_embeddings_source", "source_id", "source_type", unique=True),
        Index("ix_kb_embeddings_kb_enabled", "knowledge_base_id", "is_enabled"),
        Index("ix_kb_embeddings_knowledge", "knowledge_id"),
        Index("ix_kb_embeddings_groups", "groups", postgresql_using="gin"),
    )


# --- models -------------------------------------------------------------------------------------------
MODEL_TYPES = ("KnowledgeQA", "Embedding", "Rerank", "VLLM")


class KBModel(BASE, ModelBase, _Owned, _Visible):
    __tablename__ = "kb_models"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False)               # the provider's model name
    display_name = Column(String(255), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    type = Column(String(32), nullable=False)
    source = Column(String(64), nullable=False, default="remote")
    parameters = Column(JSONB, nullable=False, default=dict, server_default="{}")
    is_default = Column(Boolean, nullable=False, default=False)
    is_builtin = Column(Boolean, nullable=False, default=False)
    status = Column(String(32), nullable=False, default="active")
    deleted_at = _ts(nullable=True)

    __table_args__ = (Index("ix_kb_models_owner", "owner_subject"),)


# --- tasks, schedules, events -------------------------------------------------------------------
class Task(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_tasks"

    id = Column(String(36), primary_key=True, default=new_id)
    owner_subject = Column(String(255), nullable=False, default="")
    owner_org = Column(String(255), nullable=True)
    knowledge_base_id = Column(String(36), nullable=True)
    type = Column(String(64), nullable=False)
    queue = Column(String(32), nullable=False, default="default")
    priority = Column(Integer, nullable=False, default=0)
    payload = Column(JSONB, nullable=False, default=dict, server_default="{}")
    status = Column(String(16), nullable=False, default="queued")
    attempt = Column(Integer, nullable=False, default=0)
    max_retry = Column(Integer, nullable=False, default=3)
    run_after = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    locked_at = _ts(nullable=True)
    locked_by = Column(String(128), nullable=True)
    unique_key = Column(String(255), nullable=True)
    error = Column(Text, nullable=False, default="")

    __table_args__ = (
        Index("ix_kb_tasks_claim", "status", "queue", "priority", "run_after"),
        Index("ix_kb_tasks_owner", "status", "owner_subject"),
        Index("uq_kb_tasks_unique_key", "unique_key", unique=True,
              postgresql_where=text("status = 'queued' AND unique_key IS NOT NULL")),
    )


class Schedule(BASE, ModelBase):
    __tablename__ = "kb_schedules"

    name = Column(String(64), primary_key=True)
    interval_seconds = Column(Integer, nullable=False)
    next_run_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class Event(BASE, ModelBase):
    __tablename__ = "kb_events"

    id = Column(BigInteger, Identity(), primary_key=True)
    knowledge_base_id = Column(String(36), nullable=True)
    owner_subject = Column(String(255), nullable=False, default="")
    payload = Column(JSONB, nullable=False, default=dict, server_default="{}")
    at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


# Phase 3 tables (records, wiki, memory) live in their own module; importing it here registers
# them on BASE.metadata wherever the KB's tables are.
from kb_core.orm.models import ext  # noqa: E402,F401
