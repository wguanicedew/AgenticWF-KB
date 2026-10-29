#!/usr/bin/env bash
# supervisord program `kb-service`: the same app in service mode on KB_SERVICE_LISTEN, for the
# agent (docs/run-modes.md). No authentication: X-KB-Principal names the user. It never runs
# workers or schedulers (those live in `kb` or `kb-worker`). The entrypoint has checked that the
# address is a Unix socket or loopback (or KB_SERVICE_ALLOW_REMOTE=true).
set -euo pipefail

# Wait until `kb` has migrated the schema to this code's Alembic head (on an existing database the
# tables are there before new columns are).
for _ in $(seq 1 120); do
    /opt/kb/venv/bin/python -c "from agenticwf.kb.core.orm.base.migrate import at_head; \
at_head() or exit(1)" 2>/dev/null && break
    sleep 2
done

listen=${KB_SERVICE_LISTEN:?KB_SERVICE_LISTEN is not set}
if [[ "$listen" == unix:* ]]; then
    socket=${listen#unix:}
    rm -f "$socket"
    # The directory (2770, group kb) limits who can connect; the agent's gid is 1000 too.
    bind=(--uds "$socket")
else
    bind=(--host "${listen%:*}" --port "${listen##*:}")
fi
export KB_AUTH_MODE=service KB_RUN_WORKERS=false
exec /opt/kb/venv/bin/uvicorn --factory agenticwf.kb.api.app:create_app "${bind[@]}" \
    --workers 1 \
    --log-level "${LOG_LEVEL:-info}" \
    --no-server-header \
    --no-proxy-headers \
    --timeout-graceful-shutdown 5
