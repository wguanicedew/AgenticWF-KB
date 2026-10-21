#!/usr/bin/env bash
# Container entrypoint: prepare persistent state, then hand over to supervisord (docs/deployment.md).
#
#  1. validate KB_AUTH_MODE; create /var/lib/agenticwf-kb/{pgdata,files,secrets,logs}
#  2. the Fernet secrets key; per-mode secrets (standalone signing key, authsvc's shared secret)
#  3. initialise PostgreSQL on first boot (pg_search preloaded); on every boot ensure the `kb`
#     role, the database and the extensions vector, pg_trgm, pg_search (as the superuser)
#  4. listener, service-listener and worker settings for supervisord
#  5. exec supervisord (postgres, kb, kb-service, kb-worker)
#
# KB_AUTH_MODE selects who authenticates requests (docs/run-modes.md):
#   standalone (default)  a personal KB for one user: one access token, created on first boot
#   authsvc               behind Caddy + authsvc, which sign every request's AuthContext
#   service               the agent's backend: no authentication, X-KB-Principal names the user
set -euo pipefail

KB_DATA_ROOT=${KB_DATA_ROOT:-/var/lib/agenticwf-kb}
PGDATA=${PGDATA:-$KB_DATA_ROOT/pgdata}
PGBIN=${PGBIN:-/usr/pgsql-16/bin}
SECRETS_DIR=$KB_DATA_ROOT/secrets
KB_DB_NAME=${KB_DB_NAME:-kb}
KB_AUTH_MODE=${KB_AUTH_MODE:-standalone}
SOCKET_DIR=/run/agenticwf-kb

log() { echo "[entrypoint] $*" >&2; }
truthy() { [[ "${1,,}" =~ ^(1|true|yes|on)$ ]]; }
# A listener without authentication must be a Unix socket or loopback, unless allowed explicitly.
private_listener() {
    local listen=$1 host
    [[ "$listen" == unix:* || "$listen" == /* ]] && return 0
    host=${listen%:*}; host=${host#[}; host=${host%]}
    [[ "$host" == 127.* || "$host" == ::1 || "$host" == localhost ]] && return 0
    truthy "${KB_SERVICE_ALLOW_REMOTE:-false}"
}

case "$KB_AUTH_MODE" in
    standalone|authsvc|service) ;;
    *) log "ERROR: KB_AUTH_MODE must be 'standalone', 'authsvc' or 'service', not '$KB_AUTH_MODE'."; exit 1 ;;
esac

install -d -m 0755 "$KB_DATA_ROOT"
# 0711: the KB can open its own secret files but cannot list the directory.
install -d -m 0711 -o root -g root "$SECRETS_DIR"
install -d -m 0775 -o postgres -g postgres /run/postgresql
install -d -m 0750 -o kb -g kb "$KB_DATA_ROOT/files" "$KB_DATA_ROOT/logs"
# The service socket: group kb (gid 1000, the agent's gid too); setgid keeps the group.
install -d -m 2770 -o kb -g kb "$SOCKET_DIR"

secret_file() {   # secret_file <path> <command producing the value>
    local path=$1; shift
    if [[ ! -s "$path" ]]; then
        "$@" > "$path"
        chown kb:kb "$path"; chmod 0400 "$path"
        return 0
    fi
    return 1
}

# Key that encrypts secrets the KB stores in its database (model API keys). Created once; it
# never enters the database. Back it up with the volume.
export KB_SECRETS_KEY_FILE=${KB_SECRETS_KEY_FILE:-$SECRETS_DIR/secrets_key}
if secret_file "$KB_SECRETS_KEY_FILE" /opt/kb/venv/bin/python -c \
        'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'; then
    log "created the secrets key $KB_SECRETS_KEY_FILE"
    if [[ -s "$PGDATA/PG_VERSION" ]]; then
        log "WARNING: a NEW secrets key was created for an existing database: secrets stored before"
        log "WARNING: cannot be decrypted with it. Restore the old key (docs/deployment.md) and restart."
    fi
fi

case "$KB_AUTH_MODE" in
    standalone)
        unset AUTH_CONTEXT_SECRET
        # Signs the internal AuthContext and the browser cookie; kept so sign-ins survive restarts.
        export KB_STANDALONE_SECRET_FILE=$SECRETS_DIR/standalone_secret
        secret_file "$KB_STANDALONE_SECRET_FILE" /opt/kb/venv/bin/python -c \
            'import secrets; print(secrets.token_urlsafe(48))' || true
        # The access token, owned by kb (0600) so `agenticwf-kb token --rotate` can replace it.
        # `start-kb.sh` creates it on first boot and prints the sign-in link. A given KB_TOKEN
        # replaces it, and does not stay in the services' environment.
        export KB_TOKEN_FILE=${KB_TOKEN_FILE:-$SECRETS_DIR/token}
        if [[ -n "${KB_TOKEN:-}" ]]; then
            if (( ${#KB_TOKEN} < 16 )); then log "ERROR: KB_TOKEN must be at least 16 characters."; exit 1; fi
            printf '%s\n' "$KB_TOKEN" > "$KB_TOKEN_FILE"
            unset KB_TOKEN
        fi
        [[ -e "$KB_TOKEN_FILE" ]] || install -m 0600 -o kb -g kb /dev/null "$KB_TOKEN_FILE"
        chown kb:kb "$KB_TOKEN_FILE"; chmod 0600 "$KB_TOKEN_FILE"
        export KB_LOCAL_USER=${KB_LOCAL_USER:-user}
        ;;
    authsvc)
        # The same HMAC key as authsvc in the agenticwf-auth container. Preferred: a read-only file
        # mount at $AUTH_CONTEXT_SECRET_FILE; a value in AUTH_CONTEXT_SECRET is moved into a file.
        if [[ -n "${AUTH_CONTEXT_SECRET:-}" ]]; then
            printf '%s' "$AUTH_CONTEXT_SECRET" > "$SECRETS_DIR/auth_context_secret"
            chown kb:kb "$SECRETS_DIR/auth_context_secret"; chmod 0400 "$SECRETS_DIR/auth_context_secret"
            unset AUTH_CONTEXT_SECRET
            export AUTH_CONTEXT_SECRET_FILE=$SECRETS_DIR/auth_context_secret
        fi
        export AUTH_CONTEXT_SECRET_FILE=${AUTH_CONTEXT_SECRET_FILE:-/run/secrets/auth_context_secret}
        if ! runuser -u kb -- test -r "$AUTH_CONTEXT_SECRET_FILE"; then
            log "ERROR: $AUTH_CONTEXT_SECRET_FILE is missing or not readable by uid $(id -u kb)."
            log "Copy authsvc's secret from the agenticwf-auth container and mount it read-only (docs/deployment.md)."
            log "To run the KB on its own instead, use KB_AUTH_MODE=standalone (the default)."
            exit 1
        fi
        ;;
    service)
        unset AUTH_CONTEXT_SECRET
        ;;
esac

if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    log "initialising PostgreSQL cluster in $PGDATA"
    install -d -m 0700 -o postgres -g postgres "$PGDATA"
    runuser -u postgres -- "$PGBIN/initdb" -D "$PGDATA" \
        --encoding=UTF8 --locale=C.UTF-8 \
        --auth-local=peer --auth-host=reject >/dev/null
    cat >> "$PGDATA/postgresql.conf" <<EOF

# --- agenticwf-kb ---
# Unix socket only: no TCP port that could clash with other containers' PostgreSQL under host
# networking, and nothing reachable from outside the container.
listen_addresses = ''
unix_socket_directories = '/run/postgresql'
logging_collector = off
# ParadeDB pg_search (BM25) must be preloaded on PostgreSQL 16.
shared_preload_libraries = 'pg_search'
# HNSW and BM25 index builds.
maintenance_work_mem = 512MB
max_parallel_maintenance_workers = 2
EOF
fi

# On every boot: the role, the database and the extensions. The KB connects over the Unix socket
# with peer authentication (OS user `kb` = role `kb`), so there is no password. The `kb` role is
# not a superuser, so the extensions are created here.
runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -w start >/dev/null
trap 'runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true' EXIT
runuser -u postgres -- "$PGBIN/psql" -v ON_ERROR_STOP=1 -q -X -h /run/postgresql -v db="$KB_DB_NAME" <<'SQL'
SELECT 'CREATE ROLE kb LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'kb') \gexec
ALTER ROLE kb NOSUPERUSER NOCREATEDB NOCREATEROLE;
SELECT format('CREATE DATABASE %I OWNER kb ENCODING %L', :'db', 'UTF8')
 WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db') \gexec
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', :'db') \gexec
SQL
runuser -u postgres -- "$PGBIN/psql" -v ON_ERROR_STOP=1 -q -X -h /run/postgresql -d "$KB_DB_NAME" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_search;
-- after an image upgrade, bring the extensions to the installed versions
ALTER EXTENSION vector UPDATE;
ALTER EXTENSION pg_trgm UPDATE;
ALTER EXTENSION pg_search UPDATE;
SQL
runuser -u postgres -- "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null
trap - EXIT
log "PostgreSQL ready (vector, pg_trgm, pg_search)"

# Values consumed by supervisord's %(ENV_...)s expansions and the programs.
export KB_DATA_ROOT PGDATA PGBIN KB_AUTH_MODE KB_DB_NAME
[[ "$KB_AUTH_MODE" == authsvc ]] && export AUTH_CONTEXT_SECRET_FILE

# Main listener. authsvc and service: 127.0.0.1 (use `--network host`, only Caddy or the agent
# reaches it). standalone: 0.0.0.0, published by `-p 127.0.0.1:8766:8020`.
if [[ "$KB_AUTH_MODE" == standalone ]]; then
    export LISTEN_HOST=${LISTEN_HOST:-0.0.0.0}
else
    export LISTEN_HOST=${LISTEN_HOST:-127.0.0.1}
fi
export PORT=${PORT:-8020}
export LOG_LEVEL=${LOG_LEVEL:-info}
if [[ "$KB_AUTH_MODE" == service ]] && ! private_listener "$LISTEN_HOST"; then
    log "ERROR: service mode has no authentication; LISTEN_HOST=$LISTEN_HOST is not loopback or a Unix socket."
    log "Set KB_SERVICE_ALLOW_REMOTE=true only on a network that just the agent joins (docs/agent-integration.md)."
    exit 1
fi

# The extra service-mode listener for the agent (docs/run-modes.md), next to standalone or authsvc.
KB_SERVICE_LISTEN=${KB_SERVICE_LISTEN:-}
if [[ -n "$KB_SERVICE_LISTEN" ]]; then
    if [[ "$KB_AUTH_MODE" == service ]]; then
        log "ERROR: KB_SERVICE_LISTEN is for standalone or authsvc mode; in service mode the main listener is one."
        exit 1
    fi
    if ! private_listener "$KB_SERVICE_LISTEN"; then
        log "ERROR: KB_SERVICE_LISTEN=$KB_SERVICE_LISTEN is not loopback or a Unix socket (no authentication)."
        log "Set KB_SERVICE_ALLOW_REMOTE=true only on a network that just the agent joins (docs/agent-integration.md)."
        exit 1
    fi
    export KB_SERVICE_LISTEN KB_SERVICE_AUTOSTART=true
else
    export KB_SERVICE_AUTOSTART=false
fi

# Worker processes beside the API process: 0 (default) runs the tasks in the API process.
KB_WORKER_PROCESSES=${KB_WORKER_PROCESSES:-0}
if ! [[ "$KB_WORKER_PROCESSES" =~ ^[0-9]+$ ]]; then
    log "ERROR: KB_WORKER_PROCESSES must be a number, not '$KB_WORKER_PROCESSES'."; exit 1
fi
if (( KB_WORKER_PROCESSES > 0 )); then
    export KB_RUN_WORKERS=false KB_WORKERS_AUTOSTART=true KB_WORKER_SLOTS=$KB_WORKER_PROCESSES
else
    export KB_RUN_WORKERS=true KB_WORKERS_AUTOSTART=false KB_WORKER_SLOTS=1   # supervisord needs numprocs >= 1
fi

exec "$@"
