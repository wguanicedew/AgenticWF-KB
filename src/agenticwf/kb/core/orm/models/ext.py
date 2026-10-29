"""
Phase 3 tables (docs/data-model.md): structured records, the LLM wiki and temporal memory.

Each feature is behind a `[features]` switch; the tables exist regardless, so switching a
feature on needs no migration. Revision 0004 adds what `create_all` cannot (the BM25 index on
memory facts).
"""

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, String,
                        Text, func)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from agenticwf.kb.core.orm.base.models import ModelBase
from agenticwf.kb.core.orm.base.session import BASE
from agenticwf.kb.core.orm.models.kb import _Owned, _Timestamps, _Visible, _ts, new_id, utcnow


# --- structured records ---------------------------------------------------------------------------
class RecordType(BASE, ModelBase, _Owned, _Visible):
    __tablename__ = "kb_record_types"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(64), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    title = Column(String(255), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    plugin = Column(String(64), nullable=True)
    schema = Column(JSONB, nullable=False, default=dict, server_default="{}")
    roles = Column(JSONB, nullable=False, default=dict, server_default="{}")
    similarity = Column(JSONB, nullable=False, default=dict, server_default="{}")
    aliases = Column(JSONB, nullable=False, default=dict, server_default="{}")
    status = Column(String(16), nullable=False, default="active")

    __table_args__ = (
        Index("uq_kb_record_types_name", "owner_subject", sql_text("coalesce(owner_org, '')"), "name", "version",
              unique=True),
    )


class Record(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_records"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False)
    knowledge_id = Column(String(36), nullable=True)          # the upload it came from, if any
    record_type = Column(String(64 + 256), nullable=False)    # qualified: <owner>:<name>
    type_version = Column(Integer, nullable=False, default=1)
    external_id = Column(String(512), nullable=False)
    name = Column(String(512), nullable=False, default="")
    text = Column(Text, nullable=False, default="")
    data = Column(JSONB, nullable=False, default=dict, server_default="{}")
    event_time = _ts(nullable=True)
    facets = Column(JSONB, nullable=False, default=dict, server_default="{}")
    measures = Column(JSONB, nullable=False, default=dict, server_default="{}")
    features = Column(JSONB, nullable=False, default=dict, server_default="{}")
    content_hash = Column(String(64), nullable=False, default="")
    version = Column(Integer, nullable=False, default=1)
    created_by = Column(String(255), nullable=False, default="")
    valid_from = _ts(nullable=True)
    valid_until = _ts(nullable=True)
    expires_at = _ts(nullable=True)
    deprecated_at = _ts(nullable=True)
    deleted_at = _ts(nullable=True)

    __table_args__ = (
        Index("ix_kb_records_kb_type_time", "knowledge_base_id", "record_type", "event_time"),
        Index("ix_kb_records_data", "data", postgresql_using="gin", postgresql_ops={"data": "jsonb_path_ops"}),
        Index("ix_kb_records_facets", "facets", postgresql_using="gin"),
        Index("ix_kb_records_text_trgm", "text", postgresql_using="gin", postgresql_ops={"text": "gin_trgm_ops"}),
        Index("uq_kb_records_external_id", "knowledge_base_id", "record_type", "external_id", unique=True,
              postgresql_where=sql_text("deleted_at IS NULL")),
    )


class RecordItem(BASE, ModelBase):
    __tablename__ = "kb_record_items"

    record_id = Column(String(36), ForeignKey("kb_records.id", ondelete="CASCADE"), primary_key=True)
    item_index = Column(Integer, primary_key=True)
    data = Column(JSONB, nullable=False, default=dict, server_default="{}")
    measures = Column(JSONB, nullable=False, default=dict, server_default="{}")
    features = Column(JSONB, nullable=False, default=dict, server_default="{}")
    derived = Column(JSONB, nullable=False, default=dict, server_default="{}")

    __table_args__ = (
        Index("ix_kb_record_items_data", "data", postgresql_using="gin", postgresql_ops={"data": "jsonb_path_ops"}),
    )


class RecordVersion(BASE, ModelBase):
    __tablename__ = "kb_record_versions"

    id = Column(String(36), primary_key=True, default=new_id)
    record_id = Column(String(36), ForeignKey("kb_records.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    type_version = Column(Integer, nullable=False, default=1)
    data = Column(JSONB, nullable=False, default=dict)
    replaced_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (Index("uq_kb_record_versions", "record_id", "version", unique=True),)


class RecordFieldStat(BASE, ModelBase):
    __tablename__ = "kb_record_field_stats"

    knowledge_base_id = Column(String(36), primary_key=True)
    record_type = Column(String(320), primary_key=True)
    field = Column(String(255), primary_key=True)
    scope = Column(String(8), primary_key=True)                  # record | items
    count = Column(BigInteger, nullable=False, default=0)
    min = Column(Float, nullable=True)
    max = Column(Float, nullable=True)
    distinct_count = Column(BigInteger, nullable=False, default=0)
    top_values = Column(JSONB, nullable=False, default=list, server_default="[]")
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


# --- LLM wiki -------------------------------------------------------------------------------------
class WikiPage(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_wiki_pages"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False)
    slug = Column(String(512), nullable=False)
    title = Column(String(512), nullable=False, default="")
    summary = Column(Text, nullable=False, default="")
    content = Column(Text, nullable=False, default="")
    page_type = Column(String(32), nullable=False, default="entity")
    status = Column(String(16), nullable=False, default="published")
    aliases = Column(JSONB, nullable=False, default=list, server_default="[]")
    source_refs = Column(JSONB, nullable=False, default=list, server_default="[]")
    source_ids = Column(JSONB, nullable=False, default=list, server_default="[]")
    chunk_refs = Column(JSONB, nullable=False, default=list, server_default="[]")
    in_links = Column(JSONB, nullable=False, default=list, server_default="[]")
    out_links = Column(JSONB, nullable=False, default=list, server_default="[]")
    folder_id = Column(String(36), nullable=False, default="")
    category_path = Column(String(1024), nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)
    version = Column(Integer, nullable=False, default=1)
    last_edit_source = Column(String(16), nullable=False, default="pipeline")
    last_editor_id = Column(String(255), nullable=False, default="")
    deleted_at = _ts(nullable=True)

    __table_args__ = (
        Index("uq_kb_wiki_pages_slug", "knowledge_base_id", "slug", unique=True,
              postgresql_where=sql_text("deleted_at IS NULL")),
        Index("ix_kb_wiki_pages_source_ids", "source_ids", postgresql_using="gin"),
        Index("ix_kb_wiki_pages_content_trgm", "content", postgresql_using="gin",
              postgresql_ops={"content": "gin_trgm_ops"}),
    )


class WikiPageRevision(BASE, ModelBase):
    __tablename__ = "kb_wiki_page_revisions"

    id = Column(String(36), primary_key=True, default=new_id)
    page_id = Column(String(36), ForeignKey("kb_wiki_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    title = Column(String(512), nullable=False, default="")
    summary = Column(Text, nullable=False, default="")
    content = Column(Text, nullable=False, default="")
    page_type = Column(String(32), nullable=False, default="entity")
    status = Column(String(16), nullable=False, default="published")
    aliases = Column(JSONB, nullable=False, default=list)
    edit_source = Column(String(16), nullable=False, default="pipeline")
    editor_id = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (Index("uq_kb_wiki_page_revisions", "page_id", "version", unique=True),)


class WikiFolder(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_wiki_folders"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False, index=True)
    parent_id = Column(String(36), nullable=False, default="")
    name = Column(String(255), nullable=False)
    path = Column(String(1024), nullable=False)
    sort_order = Column(Integer, nullable=False, default=0)
    generated = Column(Boolean, nullable=False, default=True)

    __table_args__ = (Index("uq_kb_wiki_folders_path", "knowledge_base_id", "path", unique=True),)


class WikiPageIssue(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_wiki_page_issues"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False, index=True)
    page_slug = Column(String(512), nullable=False)
    issue_type = Column(String(32), nullable=False, default="other")
    description = Column(Text, nullable=False, default="")
    status = Column(String(16), nullable=False, default="pending")
    reporter = Column(String(255), nullable=False, default="pipeline")


class WikiPendingOp(BASE, ModelBase):
    __tablename__ = "kb_wiki_pending_ops"

    id = Column(BigInteger, Identity(), primary_key=True)
    knowledge_base_id = Column(String(36), nullable=False)
    knowledge_id = Column(String(36), nullable=False)
    op = Column(String(16), nullable=False)                      # ingest | retract
    payload = Column(JSONB, nullable=False, default=dict, server_default="{}")
    fail_count = Column(Integer, nullable=False, default=0)
    claimed_at = _ts(nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (Index("ix_kb_wiki_pending_ops_claim", "knowledge_base_id", "claimed_at", "fail_count"),)


class WikiRelation(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_wiki_relations"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False)
    source_slug = Column(String(512), nullable=False)
    relation = Column(String(128), nullable=False)
    target_slug = Column(String(512), nullable=False)
    description = Column(Text, nullable=False, default="")
    weight = Column(Float, nullable=False, default=1.0)
    chunk_refs = Column(JSONB, nullable=False, default=list, server_default="[]")
    source_ids = Column(JSONB, nullable=False, default=list, server_default="[]")

    __table_args__ = (
        Index("uq_kb_wiki_relations", "knowledge_base_id", "source_slug", "relation", "target_slug", unique=True),
        Index("ix_kb_wiki_relations_source_ids", "source_ids", postgresql_using="gin"),
    )


class WikiCommunity(BASE, ModelBase, _Timestamps):
    __tablename__ = "kb_wiki_communities"

    id = Column(String(36), primary_key=True, default=new_id)
    knowledge_base_id = Column(String(36), ForeignKey("kb_knowledge_bases.id"), nullable=False, index=True)
    level = Column(Integer, nullable=False, default=0)
    parent_id = Column(String(36), nullable=False, default="")
    member_slugs = Column(JSONB, nullable=False, default=list, server_default="[]")
    fingerprint = Column(String(64), nullable=False, default="")
    page_slug = Column(String(512), nullable=False, default="")


# --- temporal memory --------------------------------------------------------------------------------
class _MemoryOwned:
    owner_subject = Column(String(255), nullable=False)
    owner_org = Column(String(255), nullable=True)
    group_id = Column(String(128), nullable=False, default="default")


class MemoryEpisode(BASE, ModelBase, _MemoryOwned):
    __tablename__ = "kb_memory_episodes"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False, default="")
    source = Column(String(32), nullable=False, default="manual")      # conversation | document | manual
    source_ref = Column(String(255), nullable=False, default="")
    content = Column(Text, nullable=False)
    reference_time = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    status = Column(String(16), nullable=False, default="pending")
    error = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    processed_at = _ts(nullable=True)

    __table_args__ = (Index("ix_kb_memory_episodes_owner", "owner_subject", "group_id", "reference_time"),)


class MemoryEntity(BASE, ModelBase, _MemoryOwned):
    __tablename__ = "kb_memory_entities"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False)
    entity_type = Column(String(64), nullable=False, default="thing")
    summary = Column(Text, nullable=False, default="")
    aliases = Column(JSONB, nullable=False, default=list, server_default="[]")
    dimension = Column(Integer, nullable=False, default=0)
    embedding = Column(HALFVEC(), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    deleted_at = _ts(nullable=True)

    __table_args__ = (Index("ix_kb_memory_entities_owner", "owner_subject", "group_id"),)


class MemoryFact(BASE, ModelBase, _MemoryOwned):
    __tablename__ = "kb_memory_facts"

    id = Column(String(36), primary_key=True, default=new_id)
    seq = Column(BigInteger, Identity(), unique=True, nullable=False)     # BM25 key field
    source_entity_id = Column(String(36), nullable=False, index=True)
    target_entity_id = Column(String(36), nullable=False, index=True)
    relation = Column(String(128), nullable=False)
    fact = Column(Text, nullable=False)
    valid_at = _ts(nullable=True)
    invalid_at = _ts(nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now())
    expired_at = _ts(nullable=True)
    invalidated_by = Column(String(36), nullable=False, default="")
    episode_ids = Column(JSONB, nullable=False, default=list, server_default="[]")
    dimension = Column(Integer, nullable=False, default=0)
    embedding = Column(HALFVEC(), nullable=True)

    __table_args__ = (Index("ix_kb_memory_facts_owner", "owner_subject", "group_id"),)


MEMORY_BM25_INDEX = "kb_memory_facts_bm25"


# --- web search providers ------------------------------------------------------------------------------
class WebSearchProvider(BASE, ModelBase, _Timestamps):
    """A configured web search provider (docs/web-search.md): operators' rows come from kb.cfg
    `[websearch.<name>]` (owner `system`, usable by everyone); users add their own (owner only).
    `parameters.api_key` is stored encrypted (SecretBox)."""

    __tablename__ = "kb_web_search_providers"

    id = Column(String(36), primary_key=True, default=new_id)
    owner_subject = Column(String(255), nullable=False)
    owner_org = Column(String(255), nullable=True)
    name = Column(String(128), nullable=False)
    kind = Column(String(32), nullable=False)
    parameters = Column(JSONB, nullable=False, default=dict, server_default="{}")
    is_default = Column(Boolean, nullable=False, default=False)
    is_builtin = Column(Boolean, nullable=False, default=False)
    status = Column(String(16), nullable=False, default="active")
    deleted_at = _ts(nullable=True)

    __table_args__ = (Index("uq_kb_web_search_providers_name", "owner_subject", "name", unique=True,
                            postgresql_where=sql_text("deleted_at IS NULL")),)


class UserSetting(BASE, ModelBase):
    """A user's own settings (revision 0009), one row per user. `settings.web` holds their web
    search choices: `default` (a provider ref) and `disabled` (refs) (docs/web-search.md)."""

    __tablename__ = "kb_user_settings"

    owner_subject = Column(String(255), primary_key=True)
    settings = Column(JSONB, nullable=False, default=dict, server_default="{}")
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class PromptOverride(BASE, ModelBase):
    """A prompt template edited in the web page by an admin (revision 0009); it wins over the
    `[prompts] dir` file and the packaged template (docs/configuration.md, Prompt templates)."""

    __tablename__ = "kb_prompt_overrides"

    file = Column(String(64), primary_key=True)
    key = Column(String(128), primary_key=True)
    text = Column(Text, nullable=False)
    updated_by = Column(String(255), nullable=False, default="")
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


# --- remote knowledge bases ----------------------------------------------------------------------------
class RemoteSource(BASE, ModelBase, _Timestamps):
    """A remote KB server whose knowledge bases its owner also lists and searches
    (docs/retrieval.md, Remote knowledge bases). Users add their own (owner only); operators'
    rows come from kb.cfg `[remote.<name>]` (owner `system`, `is_builtin`, for the standalone user).
    `auth.token` and `auth.pat` are stored encrypted (SecretBox). `disabled_kbs` holds the remote's
    own KB ids that the owner switched off."""

    __tablename__ = "kb_remotes"

    id = Column(String(36), primary_key=True, default=new_id)
    owner_subject = Column(String(255), nullable=False)
    owner_org = Column(String(255), nullable=True)
    name = Column(String(64), nullable=False)
    url = Column(String(2048), nullable=False)
    auth = Column(JSONB, nullable=False, default=dict, server_default="{}")
    knowledge_bases = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    disabled_kbs = Column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    is_enabled = Column(Boolean, nullable=False, default=True)
    is_builtin = Column(Boolean, nullable=False, default=False)
    timeout = Column(Float, nullable=False, default=10.0)
    deleted_at = _ts(nullable=True)

    __table_args__ = (Index("uq_kb_remotes_name", "owner_subject", "name", unique=True,
                            postgresql_where=sql_text("deleted_at IS NULL")),)
