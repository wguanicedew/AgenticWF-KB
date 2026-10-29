#!/usr/bin/env bash
# supervisord program `kb-worker`: a worker process beside the API (runs as user `kb`). The API
# program `kb` migrates the schema; a worker waits until it is at this code's Alembic head (on an
# existing database the tables are there before new columns are).
set -euo pipefail

for _ in $(seq 1 120); do
    /opt/kb/venv/bin/python -c "from agenticwf.kb.core.orm.base.migrate import at_head; \
at_head() or exit(1)" 2>/dev/null && break
    sleep 2
done
exec /opt/kb/venv/bin/agenticwf-kb-worker
