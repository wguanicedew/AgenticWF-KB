"""Extensions (phase 3): structured records, LLM wiki, temporal memory.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-16

The kb_record_*, kb_wiki_* and kb_memory_* tables and their ordinary indexes come from
`create_all`, which `build_database()` runs before Alembic. This revision adds the BM25 index on
memory facts (ParadeDB pg_search), which `create_all` cannot make.
"""
from alembic import op

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from agenticwf.kb.core.orm.models.indexes import memory_bm25_index_sql

    op.execute(memory_bm25_index_sql())


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS kb_memory_facts_bm25")
