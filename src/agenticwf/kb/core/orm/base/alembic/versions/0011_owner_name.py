"""Owner names: kb_knowledge_bases.owner_name.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-03

A knowledge base's ref is `<owner username>:<slug>` (the owner's Logto username, or their email
when they have none); `owner_subject` stays the key that ownership and access use. `create_all`
makes the column on new databases; this revision adds it and its index to existing ones. Existing
knowledge bases get their owner name the next time the owner signs in (ensure_personal).
"""
from alembic import op

revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kb_knowledge_bases ADD COLUMN IF NOT EXISTS owner_name varchar(320)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_kb_knowledge_bases_owner_name ON kb_knowledge_bases (owner_name) "
               "WHERE owner_name IS NOT NULL")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_knowledge_bases_owner_name")
    op.execute("ALTER TABLE kb_knowledge_bases DROP COLUMN IF EXISTS owner_name")
