"""Web search providers: kb_web_search_providers.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-18

The table comes from `create_all`, which `build_database()` runs before Alembic; this revision
records the schema step (docs/web-search.md).
"""

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
