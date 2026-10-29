"""Remote knowledge bases: kb_remotes.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-26

The table comes from `create_all`, which `build_database()` runs before Alembic; this revision
records the schema step (docs/retrieval.md, Remote knowledge bases).
"""

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
