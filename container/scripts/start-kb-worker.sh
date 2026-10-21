#!/usr/bin/env bash
# supervisord program `kb-worker`: a worker process beside the API (runs as user `kb`). The API
# program `kb` migrates the schema; a worker waits for it.
set -euo pipefail

for _ in $(seq 1 120); do
    /opt/kb/venv/bin/python -c "import sqlalchemy as sa; from kb_core.orm.base.session import get_engine; \
sa.inspect(get_engine()).has_table('alembic_version') or exit(1)" 2>/dev/null && break
    sleep 2
done
exec /opt/kb/venv/bin/agenticwf-kb-worker
