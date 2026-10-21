# Container and deployment

One container runs the whole KB service under supervisord:

* PostgreSQL with pgvector and ParadeDB `pg_search`;
* the KB API;
* the optional service listener for the agent;
* KB workers.

The layout copies AgenticWF-agent's `container/` file for file, so the two images are operated
the same way. The container is also how the KB runs on its own (`KB_AUTH_MODE=standalone`, the default); there is
no non-container run mode.

```
container/
  Dockerfile
  agenticwf-kb.env.example         -> /etc/agenticwf/kb.env on the host
  config/kb.cfg                    -> /etc/agenticwf/kb.cfg
  scripts/
    entrypoint.sh                  first-boot and every-boot setup, then exec supervisord
    start-kb.sh                    wait for PostgreSQL, migrate, exec uvicorn / agenticwf-kb serve
    start-kb-service.sh            the service-mode listener (KB_SERVICE_LISTEN)
    start-kb-worker.sh             wait for kb_tasks, exec agenticwf-kb-worker
    healthcheck.sh                 pg_isready + GET /kb/health
    supervisor-fatal.sh            any FATAL program stops the container (restart policy takes over)
  supervisord.conf
  supervisord.d/
    postgres.ini  kb.ini  kb-service.ini  kb-worker.ini  fatal.ini
```

## Image

| Item | Value |
|---|---|
| Base | `almalinux:9`, `ARG PYTHON=python3.12` |
| PostgreSQL | `ARG POSTGRES_MAJOR=16`, from the PGDG repository: `postgresql16-server`, `postgresql16-contrib` (`pg_trgm`), `pgvector_16`. The agent uses the AlmaLinux module stream, which has no pgvector. PGDG ships pgvector ≥ 0.8, which provides `halfvec` and `hnsw.iterative_scan` |
| BM25 (required) | ParadeDB `pg_search`: the el9 RPM for the PostgreSQL major, from ParadeDB's GitHub release, pinned with `ARG PG_SEARCH_VERSION` (0.25.11; CI tests against `paradedb/paradedb:0.25.11-pg16`, keep the two in step). The build fails if that RPM is missing for the target architecture. On PostgreSQL 16, `shared_preload_libraries = 'pg_search'` is required |
| Other packages | `supervisor`, `procps-ng`, `curl` (healthcheck). No system parser libraries: PDFs are read with the `pypdfium2` wheels |
| User | `kb`, uid 1000, group `kb` gid 1000 (matches the agent's user, for the shared socket), nologin |
| Python | venv `/opt/kb/venv`; `pip install /opt/kb/src/authsvc` (a named build context, as for the agent), then `pip install "/opt/kb/src/kb[parsers]"` |
| Packages installed | `kb_api`, `kb_core`, `kb_runtime` (do not collide with the agent's `api`, `core`, `runtime`) |
| Files | `supervisord.conf` → `/etc/supervisord.conf`, `supervisord.d/` → `/etc/supervisord.d/`, `config/kb.cfg` → `/etc/agenticwf/kb.cfg`, `src/kb_runtime/defaults/` → `/etc/agenticwf/kb/`, scripts → `/usr/local/bin/` |
| ENV | `KB_DATA_ROOT=/var/lib/agenticwf-kb`, `PGDATA=$KB_DATA_ROOT/pgdata`, `PGBIN=/usr/pgsql-16/bin`, `AGENTICWF_KB_CONFIG=/etc/agenticwf/kb.cfg`, `AUTH_CONTEXT_SECRET_FILE=/run/secrets/auth_context_secret`, `KB_CONTAINER=1` |
| Volume | `/var/lib/agenticwf-kb` |
| Expose | 8020 |
| Healthcheck | `--interval=30s --timeout=5s --start-period=60s --retries=3 CMD healthcheck.sh` |
| Entrypoint | `entrypoint.sh`, `CMD ["/usr/bin/supervisord","-n","-c","/etc/supervisord.conf"]` |

Build:

```bash
docker build --network=host -f container/Dockerfile \
  --build-context authsvc=../AgenticWF-auth/authsvc -t agenticwf-kb .
```

CI: `.github/workflows/build-container.yml`, copied from the agent. The image name is
`agenticwf-kb`, and `AGENTICWF_REF` is handled the same way for the authsvc build context. The
test job uses a pinned `paradedb/paradedb` service image, which has pgvector, pg_search and
pg_trgm. There is no SQLite test run.

## Data volume

```
/var/lib/agenticwf-kb/
  pgdata/          PostgreSQL cluster
  files/           originals, parsed.<hash>.md, images, versions   ([storage] root)
  secrets/         secrets_key; standalone: token, standalone_secret;
                   authsvc: auth_context_secret (if given by env)
  logs/
```

Back up three things together: `pgdata` (with `pg_dump`), `files/` and
`secrets/secrets_key`. The database refers to files by path, and model API keys are encrypted
with `secrets_key`.

## entrypoint.sh

Same sequence as the agent's, with KB names and an extension step:

1. **Validate.** `KB_AUTH_MODE` must be `standalone`, `authsvc` or `service` (default `standalone`).
   Set defaults for `KB_DATA_ROOT`, `PGDATA` and `KB_DB_NAME=kb`.
2. **Directories.**
   * `$KB_DATA_ROOT` (0755);
   * `secrets/` (0711, root);
   * `/run/postgresql` (0775, postgres);
   * `files/` and `logs/` (0750, kb);
   * `/run/agenticwf-kb` (2770, kb:kb), the service socket directory.
3. **Fernet key.** Create `KB_SECRETS_KEY_FILE` (default `secrets/secrets_key`, 0400 kb) if
   missing. Warn if `pgdata` already exists without it.
4. **Auth.**
   * `standalone`: unset `AUTH_CONTEXT_SECRET`; create `secrets/standalone_secret` (0400 kb)
     if missing; write a given `KB_TOKEN` to `secrets/token` (KB_TOKEN_FILE, 0600 kb) and unset
     it; default `KB_LOCAL_USER=user`.
   * `authsvc`: move `AUTH_CONTEXT_SECRET` into a file if given, then require
     `AUTH_CONTEXT_SECRET_FILE` to be readable by `kb`, else exit 1.
   * `service`: no secret. Refuse a non-loopback `LISTEN_HOST` unless
     `KB_SERVICE_ALLOW_REMOTE=true`.
5. **First boot** (no `PG_VERSION`).
   * `initdb --encoding=UTF8 --locale=C.UTF-8 --auth-local=peer --auth-host=reject`.
   * Settings: `listen_addresses = ''`, `unix_socket_directories = '/run/postgresql'`,
     `shared_preload_libraries = 'pg_search'`.
   * Index-build tuning: `maintenance_work_mem = 512MB`,
     `max_parallel_maintenance_workers = 2`.
6. **Every boot**, with a temporary `pg_ctl start`:
   * `CREATE ROLE kb LOGIN` if missing, then `ALTER ROLE kb NOSUPERUSER NOCREATEDB NOCREATEROLE`;
   * `CREATE DATABASE kb OWNER kb` if missing, then `REVOKE ALL … FROM PUBLIC`;
   * as the postgres superuser, in database `kb`: `CREATE EXTENSION IF NOT EXISTS vector`,
     `pg_trgm` and `pg_search`, plus `ALTER EXTENSION … UPDATE` after image upgrades. The
     `kb` role is not a superuser, so migrations cannot create extensions themselves;
   * `pg_ctl stop`.
7. **Listener defaults.** `LISTEN_HOST` / `PORT` per mode: `127.0.0.1:8020` for authsvc and
   service, `0.0.0.0:8020` for standalone. `LOG_LEVEL=info`.
8. **Service listener.** A non-empty `KB_SERVICE_LISTEN` sets `KB_SERVICE_AUTOSTART=true`,
   else `false`. It is not allowed with `KB_AUTH_MODE=service`, where the main listener
   already is one. A non-loopback address (e.g. `0.0.0.0:8021` on a Docker network) needs
   `KB_SERVICE_ALLOW_REMOTE=true`; see
   [agent-integration.md](agent-integration.md#connecting-the-containers).
9. **Workers.** `KB_WORKER_PROCESSES` defaults to 0 and must be numeric (any mode).
   * If N > 0: `KB_WORKERS_SHARED=true KB_WORKERS_AUTOSTART=true KB_WORKER_SLOTS=N`, and the
     API runs no workers.
   * Otherwise: `KB_WORKERS_AUTOSTART=false KB_WORKER_SLOTS=1`, and the API runs them
     in-process.
10. `exec "$@"`.

## supervisord programs

`supervisord.conf` is the agent's, unchanged: runs as root, nodaemon, socket
`/run/supervisor.sock` (0700), and includes `/etc/supervisord.d/*.ini`.

| File | Program | Command | User | Notes |
|---|---|---|---|---|
| `postgres.ini` | `postgres` | `%(ENV_PGBIN)s/postgres -D %(ENV_PGDATA)s` | postgres | priority 10, stopsignal INT, stopwaitsecs 60 |
| `kb.ini` | `kb` | `start-kb.sh` | kb | priority 30; the main listener, in `KB_AUTH_MODE` |
| `kb-service.ini` | `kb-service` | `start-kb-service.sh` | kb | priority 31, `autostart=%(ENV_KB_SERVICE_AUTOSTART)s`; service mode on `KB_SERVICE_LISTEN` |
| `kb-worker.ini` | `kb-worker` | `start-kb-worker.sh` | kb | `numprocs=%(ENV_KB_WORKER_SLOTS)s`, `process_name=%(program_name)s-%(process_num)s`, `autostart=%(ENV_KB_WORKERS_AUTOSTART)s`, priority 40, stopwaitsecs 60 (a task in flight is re-queued on SIGTERM) |
| `fatal.ini` | eventlistener `fatal` | `supervisor-fatal.sh` | root | PROCESS_STATE_FATAL → stop the container |

All programs log to stdout/stderr (`docker logs`).

## Scripts

**start-kb.sh**

1. Wait up to 60 s for `pg_isready -h /run/postgresql -d kb`.
2. Run `python -m kb_api.db init` (create_all, then Alembic `upgrade head`), retried with
   backoff from 2 to 30 s for about 5 minutes. Only this program migrates; the others wait
   for it.
3. In standalone mode, `agenticwf-kb token --ensure`: create the token if missing and print the
   `open: ${KB_PUBLIC_URL}/kb/?token=…` link.
4. Exec the server, in every mode:
   `uvicorn --factory kb_api.app:create_app --host $LISTEN_HOST --port $PORT --workers 1 --no-server-header --no-proxy-headers --timeout-graceful-shutdown 5`
   (`--uds` for `LISTEN_HOST=unix:/path`). `create_app` reads the mode from `KB_AUTH_MODE`
   and installs `StandaloneAuth`, `AuthContextMiddleware` or `ServicePrincipal`.

**start-kb-service.sh**

1. Wait until the `kb_tasks` table exists (that is, `kb` has migrated).
2. Exec `KB_AUTH_MODE=service KB_RUN_WORKERS=false uvicorn --factory kb_api.app:create_app`,
   either on `--uds /run/agenticwf-kb/kb.sock` (socket mode 0660, group `kb`) or on
   `--host/--port` from `KB_SERVICE_LISTEN`.

This listener never runs workers or schedulers, so in-process workers exist only once, in
`kb`.

**start-kb-worker.sh**: wait for `kb_tasks`, then exec `agenticwf-kb-worker`.

**healthcheck.sh**: `pg_isready` and `curl -fsS http://127.0.0.1:${PORT}/kb/health`.

`uvicorn --workers 1`: the API holds in-process state (the task wake-up event and the SSE
fan-out). Scale with `KB_WORKER_PROCESSES`, which is where the CPU goes (parsing and
embedding batches).

## kb.cfg

The sample below is the minimal file the image ships. Every section and key, with defaults, is
in [configuration.md](configuration.md).

ConfigParser INI, re-read on each `config_get`, like the agent's. It is found at
`$AGENTICWF_KB_CONFIG`, else `$AGENTICWF_HOME/etc/agenticwf/kb.cfg`, else
`/etc/agenticwf/kb.cfg`, else `$VIRTUAL_ENV/etc/agenticwf/kb.cfg`.

```ini
[database]
default = postgresql+psycopg://kb@/kb?host=/run/postgresql
pool_size = 10
max_overflow = 10
pool_recycle = 3600

[logging]
level = INFO

[storage]
backend = local
root = /var/lib/agenticwf-kb/files
max_file_mb = 50

[retrieval]
bm25_tokenizer = icu              # one BM25 index for the deployment; change => reindex-bm25
rrf_k = 60
vector_weight = 0.7
keyword_weight = 0.3
rerank_top_k = 30
rerank_threshold = 0.3

[ingestion]
chunk_size = 512
chunk_overlap = 50
strategy = auto                   # legacy | heading | heuristic | auto
document_timeout = 1800

[workers]
queues = default postprocess wiki
concurrency = 3
poll_seconds = 1

[models]
# system models (owner "system", visible to everyone); users add their own through the API
default_embedding = bge-m3
default_chat = small
default_rerank =
# [models.embedding.bge-m3]
# provider = openai_compat
# base_url = http://127.0.0.1:8080/v1
# model = BAAI/bge-m3
# dimension = 1024
# api_key_env = EMBEDDINGS_API_KEY

[security]
ssrf_whitelist =                  # hosts / CIDRs allowed for URL import despite being private

[agentic]
model =                           # tool-capable chat model; default: [models] default_chat
max_iterations = 10
max_iterations_limit = 20
timeout = 180
max_input_tokens = 200000
per_owner = 2

[features]
faq = true
records = false                   # phase 3
wiki = false                      # phase 3
memory = false                    # phase 3
```

## Running it

### Behind AgenticWF-auth, with the agent

```bash
# the secret, as for the agent (uid 1000)
docker exec agenticwf-auth cat /var/lib/agenticwf/secrets/auth_context_secret > /etc/agenticwf/auth_context_secret
chmod 0400 /etc/agenticwf/auth_context_secret

# merge deploy/policy.kb.yaml into /etc/agenticwf/policy.yaml, then
docker exec agenticwf-auth supervisorctl restart authsvc

docker run -d --name agenticwf-kb --restart unless-stopped --network host \
  --env-file /etc/agenticwf/kb.env \
  -e KB_SERVICE_LISTEN=unix:/run/agenticwf-kb/kb.sock \
  -v /etc/agenticwf/auth_context_secret:/run/secrets/auth_context_secret:ro \
  -v agenticwf-kb-data:/var/lib/agenticwf-kb \
  -v agenticwf-kb-sock:/run/agenticwf-kb \
  agenticwf-kb:latest
ss -ltnp | grep ':8020\b'      # 127.0.0.1:8020 only
```

Then add the Caddy `handle /kb/*` block
([run-modes.md](run-modes.md#authsvc-behind-agenticwf-auth)) and mount
`agenticwf-kb-sock:/run/agenticwf-kb` into the agent container
([agent-integration.md](agent-integration.md)).

### Only as the agent's backend

```bash
docker run -d --name agenticwf-kb --restart unless-stopped \
  -e KB_AUTH_MODE=service \
  -e LISTEN_HOST=unix:/run/agenticwf-kb/kb.sock \
  -v agenticwf-kb-data:/var/lib/agenticwf-kb \
  -v agenticwf-kb-sock:/run/agenticwf-kb \
  agenticwf-kb:latest
```

`LISTEN_HOST=unix:/path` makes `start-kb.sh` run uvicorn with `--uds` instead of
`--host/--port`. Use `--network host` and the default `127.0.0.1:8020` only if the agent
cannot mount the volume.

### Standalone

```bash
docker run -d --name agenticwf-kb --restart unless-stopped \
  -p 127.0.0.1:8766:8020 -v agenticwf-kb-data:/var/lib/agenticwf-kb agenticwf-kb
```

Keys, browser sign-in and exposure:
[run-modes.md](run-modes.md#standalone-a-personal-kb).

## Operations

Same commands as the agent:

| Task | Command |
|---|---|
| Logs | `docker logs -f agenticwf-kb` |
| Process control | `docker exec agenticwf-kb supervisorctl status`, `restart kb`, `restart kb-worker:*` |
| Database shell | `docker exec -u postgres agenticwf-kb psql -h /run/postgresql kb` |
| Standalone token | `docker exec agenticwf-kb agenticwf-kb token` prints it, `token --rotate` replaces it (no restart). The CLI re-runs itself as user `kb`, who owns the token file |
| Smoke test | `scripts/smoke-container.sh <image>`: all three modes, as CI runs it |
| Stop | `docker stop -t 60`, which lets workers re-queue running tasks |

Upgrades run migrations on start, and the entrypoint updates the extensions.

Changing the default embedding model does not re-embed existing KBs, because each KB's model
is fixed at creation. Re-embedding is an explicit
`POST /kb/api/v1/knowledge-bases/{id}/reindex`.
