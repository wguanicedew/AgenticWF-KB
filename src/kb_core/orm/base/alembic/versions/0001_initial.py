"""Initial revision: the required extensions.

Revision ID: 0001
Revises:
Create Date: 2026-10-15

Phase 0 has no tables; this revision fixes the starting point and checks that a superuser has
created the extensions the KB needs (the container's entrypoint does). Phase 1's revision adds the
knowledge tables. Like every KB revision, later ones only add what is missing, because
`build_database()` runs `create_all` first.
"""
from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from kb_core.orm.base.utils import required_extensions

    found = {row[0] for row in op.get_bind().execute(sa.text('SELECT extname FROM pg_extension'))}
    missing = [name for name in required_extensions() if name not in found]
    if missing:
        raise RuntimeError(f'missing PostgreSQL extensions: {", ".join(missing)} (create them as a superuser)')


def downgrade() -> None:
    pass
