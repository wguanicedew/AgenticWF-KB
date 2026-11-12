"""Knowledge base status: kb_knowledge_bases.status.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-10

`active`, `removing` / `purging` (deleted; a task is removing its rows and files, and a periodic
job queues that task again while it is not done; purging: the KB row goes too) or `removed` (only the empty KB row is left until
housekeeping purges it). KBs deleted before this revision count as removed.
"""
from alembic import op

revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kb_knowledge_bases ADD COLUMN IF NOT EXISTS status varchar(16) NOT NULL DEFAULT 'active'")
    op.execute("UPDATE kb_knowledge_bases SET status = 'removed' WHERE deleted_at IS NOT NULL AND status = 'active'")
    op.execute("CREATE INDEX IF NOT EXISTS ix_kb_knowledge_bases_removing ON kb_knowledge_bases (status) "
               "WHERE status IN ('removing', 'purging')")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_knowledge_bases_removing")
    op.execute("ALTER TABLE kb_knowledge_bases DROP COLUMN IF EXISTS status")
