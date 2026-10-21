# syntax=docker/dockerfile:1
#
# AgenticWF knowledge service (AlmaLinux 9): one container, every process under supervisord
#
#   supervisord
#   ├── postgres     PostgreSQL 16 (PGDG) + pgvector + pg_search (ParadeDB) + pg_trgm,
#   │                Unix socket /run/postgresql only, database `kb`
#   ├── kb           uvicorn kb_api.app:create_app on ${LISTEN_HOST}:8020, in KB_AUTH_MODE:
#   │                standalone (default; a personal KB, one access token), authsvc (behind Caddy +
#   │                authsvc), service (the agent's backend, X-KB-Principal)
#   ├── kb-service   optional second listener in service mode (KB_SERVICE_LISTEN), for the agent
#   └── kb-worker×N  worker processes (KB_WORKER_PROCESSES; default 0 = in the API process)
#
# The `authsvc` package (AuthContext signing and verification) comes from the AgenticWF-auth
# repository through a named build context. Build from the repository root:
#
#   docker build --network=host -f container/Dockerfile \
#     --build-context authsvc=../AgenticWF-auth/authsvc -t agenticwf-kb .
#
# See docs/deployment.md.

FROM almalinux:9

ARG POSTGRES_MAJOR=16
ARG PYTHON=python3.12
# ParadeDB pg_search (BM25), required: the el9 RPM for this PostgreSQL major. The build fails if
# the release has no RPM for the target architecture.
ARG PG_SEARCH_VERSION=0.25.11

LABEL org.opencontainers.image.title="agenticwf-kb" \
      org.opencontainers.image.description="AgenticWF knowledge service + PostgreSQL (pgvector, pg_search) on AlmaLinux 9" \
      org.opencontainers.image.licenses="Apache-2.0"

# PostgreSQL from the PGDG repository: the AlmaLinux module stream has no pgvector.
RUN set -eux; \
    arch="$(uname -m)"; \
    dnf -y install epel-release; \
    dnf -y install "https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-${arch}/pgdg-redhat-repo-latest.noarch.rpm"; \
    dnf -qy module disable postgresql; \
    dnf -y install --setopt=install_weak_deps=False \
        "postgresql${POSTGRES_MAJOR}-server" "postgresql${POSTGRES_MAJOR}-contrib" "pgvector_${POSTGRES_MAJOR}" \
        "${PYTHON}" "${PYTHON}-pip" \
        supervisor procps-ng; \
    dnf -y install \
        "https://github.com/paradedb/paradedb/releases/download/v${PG_SEARCH_VERSION}/pg_search_${POSTGRES_MAJOR}-${PG_SEARCH_VERSION}-1PARADEDB.el9.${arch}.rpm"; \
    test -f "/usr/pgsql-${POSTGRES_MAJOR}/share/extension/vector.control"; \
    test -f "/usr/pgsql-${POSTGRES_MAJOR}/share/extension/pg_search.control"; \
    test -f "/usr/pgsql-${POSTGRES_MAJOR}/share/extension/pg_trgm.control"; \
    dnf clean all; rm -rf /var/cache/dnf

# The KB runs as uid/gid 1000, like the agent: the agent (uid 1000) can then use the service
# socket (group kb), and the host's copy of the AuthContext secret is readable. PostgreSQL peer
# authentication maps this OS user to the `kb` role.
RUN groupadd --gid 1000 kb \
 && useradd --uid 1000 --gid kb --create-home --home-dir /home/kb --shell /sbin/nologin kb

# KB and authsvc in one virtualenv; code stays root-owned.
COPY --from=authsvc . /opt/kb/src/authsvc
COPY pyproject.toml /opt/kb/src/kb/
COPY src /opt/kb/src/kb/src
RUN "${PYTHON}" -m venv /opt/kb/venv \
 && /opt/kb/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/kb/venv/bin/pip install --no-cache-dir /opt/kb/src/authsvc \
 && /opt/kb/venv/bin/pip install --no-cache-dir "/opt/kb/src/kb[parsers]"

COPY container/supervisord.conf /etc/supervisord.conf
COPY container/supervisord.d/ /etc/supervisord.d/
COPY container/config/kb.cfg /etc/agenticwf/kb.cfg
COPY container/scripts/ /usr/local/bin/
RUN chmod 0755 /usr/local/bin/*.sh

# KB_AUTH_MODE is not set here: the entrypoint defaults it to standalone.
ENV KB_DATA_ROOT=/var/lib/agenticwf-kb \
    PGDATA=/var/lib/agenticwf-kb/pgdata \
    PGBIN=/usr/pgsql-${POSTGRES_MAJOR}/bin \
    AGENTICWF_KB_CONFIG=/etc/agenticwf/kb.cfg \
    AUTH_CONTEXT_SECRET_FILE=/run/secrets/auth_context_secret \
    KB_CONTAINER=1 \
    PATH=/opt/kb/venv/bin:/usr/pgsql-${POSTGRES_MAJOR}/bin:$PATH \
    PYTHONUNBUFFERED=1

# Database, files, secrets.
VOLUME ["/var/lib/agenticwf-kb"]

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]
