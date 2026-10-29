"""Sites: kb_sites and kb_knowledges.site_id.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-26

A site is imported from one URL: its pages are URL items that point back to it
(docs/ingestion.md, Sites). `create_all` makes the table and the column on new databases; this
revision adds the column and its index to existing ones.
"""
from alembic import op

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kb_knowledges ADD COLUMN IF NOT EXISTS site_id varchar(36)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_kb_knowledges_site ON kb_knowledges (site_id) WHERE site_id IS NOT NULL")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_knowledges_site")
    op.execute("ALTER TABLE kb_knowledges DROP COLUMN IF EXISTS site_id")
