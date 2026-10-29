"""User settings and prompt overrides: kb_user_settings, kb_prompt_overrides.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-27

The tables come from `create_all`, which `build_database()` runs before Alembic; this revision
records the schema step (docs/web-search.md, docs/configuration.md, Prompt templates).
"""

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
