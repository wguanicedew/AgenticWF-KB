"""Quality (phase 2): tags, tag relations, knowledge versions and conflicts.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15

The tables kb_knowledge_tags, kb_knowledge_tag_relations, kb_knowledge_versions and
kb_knowledge_conflicts and their indexes come from `create_all`, which `build_database()` runs
before Alembic; this revision records the schema step. FAQ entries need no table (they are
kb_chunks rows with chunk_type = faq).
"""

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
