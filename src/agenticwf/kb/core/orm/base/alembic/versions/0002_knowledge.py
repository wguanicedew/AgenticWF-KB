"""Knowledge: the BM25 index on kb_embeddings.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15

The phase-1 tables (kb_knowledge_bases, kb_knowledges, kb_chunks, kb_embeddings, kb_models,
kb_tasks, kb_schedules, kb_events) and their ordinary indexes, including the trigram index on
chunk content, come from `create_all`, which `build_database()` runs before Alembic. This revision
adds what `create_all` cannot: the ParadeDB BM25 index with the deployment's tokenizer. HNSW
indexes are created per dimension at run time.
"""
from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from agenticwf.kb.core.orm.models.indexes import ensure_bm25_index

    ensure_bm25_index(op.get_bind())


def downgrade() -> None:
    from agenticwf.kb.core.orm.models.indexes import BM25_INDEX

    op.execute(f"DROP INDEX IF EXISTS {BM25_INDEX}")
