"""Source citations: kb_knowledges.citation.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-17

Every knowledge item carries its source citation (title, authors, published, publisher, version,
url, doi, accessed, text), so search results and answers can name their sources
(docs/ingestion.md, Source citations). `create_all` makes the column on new databases; this
revision adds it to existing ones.
"""
from alembic import op

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kb_knowledges ADD COLUMN IF NOT EXISTS citation jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE kb_knowledges DROP COLUMN IF EXISTS citation")
