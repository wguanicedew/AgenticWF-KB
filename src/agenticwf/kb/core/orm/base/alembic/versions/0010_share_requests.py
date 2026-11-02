"""Requests to share a knowledge base more widely: kb_share_requests.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-30

The table comes from `create_all`, which `build_database()` runs before Alembic; this revision
records the schema step (docs/data-model.md, Sharing requests).
"""

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
