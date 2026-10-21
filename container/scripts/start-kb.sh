#!/usr/bin/env bash
# supervisord program `kb`: the KB API, the main listener in KB_AUTH_MODE (runs as user `kb`).
set -euo pipefail

for _ in $(seq 1 60); do
    pg_isready -q -h /run/postgresql -d "${KB_DB_NAME:-kb}" && break
    sleep 1
done

# Check the extensions, create missing tables, bring Alembic to head; safe on every start. Only
# this program migrates; kb-service and kb-worker wait for it. Retried with backoff (about 5
# minutes): an external database briefly down at boot must not exhaust supervisord's retries.
delay=2
for attempt in $(seq 1 12); do
    /opt/kb/venv/bin/python -m kb_api.db init && break
    if [[ $attempt -eq 12 ]]; then
        echo "[start-kb] the database is still unavailable; giving up" >&2
        exit 1
    fi
    echo "[start-kb] database not ready (attempt $attempt); retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$(( delay < 30 ? delay * 2 : 30 ))
done

if [[ "${KB_AUTH_MODE:-standalone}" == standalone ]]; then
    # The access token (created on first boot) and the sign-in link, printed to the logs.
    /opt/kb/venv/bin/agenticwf-kb token --ensure
fi

# One uvicorn worker: SSE fan-out and the task wake-up live in this process. Clients' addresses
# come from Caddy (authsvc), the agent (service) or the published port (standalone).
if [[ "$LISTEN_HOST" == unix:* ]]; then
    rm -f "${LISTEN_HOST#unix:}"
    bind=(--uds "${LISTEN_HOST#unix:}")
else
    bind=(--host "$LISTEN_HOST" --port "$PORT")
fi
exec /opt/kb/venv/bin/uvicorn --factory kb_api.app:create_app "${bind[@]}" \
    --workers 1 \
    --log-level "$LOG_LEVEL" \
    --no-server-header \
    --no-proxy-headers \
    --timeout-graceful-shutdown 5
