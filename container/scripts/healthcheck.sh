#!/usr/bin/env bash
# Container health: PostgreSQL accepts connections and the KB API answers /kb/health.
set -euo pipefail
# As `postgres`: Docker runs the check as root, and peer authentication has no role "root".
runuser -u postgres -- pg_isready -q -h /run/postgresql -d postgres
base=${KB_BASE_PATH:-/kb}
listen=${LISTEN_HOST:-127.0.0.1}
if [[ "$listen" == unix:* ]]; then
    curl -fsS -o /dev/null --unix-socket "${listen#unix:}" "http://kb${base}/health"
else
    [[ "$listen" == 0.0.0.0 || "$listen" == "::" ]] && listen=127.0.0.1
    curl -fsS -o /dev/null "http://${listen}:${PORT:-8020}${base}/health"
fi
