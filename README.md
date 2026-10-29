# AgenticWF-KB

The knowledge service of AgenticWF. It holds knowledge bases of documents, FAQs and
structured records, ingests them into PostgreSQL (pgvector + ParadeDB BM25), and serves them
through hybrid search. Authentication comes from
[AgenticWF-auth](https://github.com/wguanicedew/AgenticWF-auth); the all-in-one image is built by
[AgenticWF](https://github.com/wguanicedew/AgenticWF).

```
Users ── Caddy ─► authsvc /verify ─► agenticwf-kb :8020       (authsvc mode)
Agent ── Unix socket, X-KB-Principal ─► agenticwf-kb          (service mode: no auth, KB filters by the user)
You   ── token ─► 127.0.0.1:8766 ─► agenticwf-kb              (standalone mode: a personal KB)
       [ AlmaLinux 9 container: PostgreSQL + pgvector + pg_search, KB API, KB workers, under supervisord ]
```

Knowledge bases with KB and item visibility; file, URL and manual knowledge (md, txt, csv, json,
html, pdf, docx, xlsx, pptx, epub, images) parsed, chunked and indexed; hybrid search (pgvector +
BM25, RRF, rerank, MMR); models; a task queue and workers; the agent's `kb_search` / `kb_read`.

A personal KB in one container:

```bash
docker build --network=host -f container/Dockerfile --build-context authsvc=../AgenticWF-auth/authsvc -t agenticwf-kb .
docker run -d --name agenticwf-kb -p 127.0.0.1:8766:8020 -v agenticwf-kb-data:/var/lib/agenticwf-kb agenticwf-kb
docker logs agenticwf-kb | grep open:          # the sign-in link
docker exec agenticwf-kb agenticwf-kb token     # the token, for curl and scripts
```

Two pip packages, built into `dist/` by `scripts/build-packages.sh [server|client]`:

| Package | Source | Contains |
|---|---|---|
| `agenticwf-kb-server` | the repository root | the service (`agenticwf.kb.api`, `core`, `runtime`), `agenticwf-kb`, `agenticwf-kb-worker`; what the container installs |
| `agenticwf-kb-client` | [client/](client) | `KBClient` (`agenticwf.kb.client`; `[hpo]` adds the Optuna helper) and the `agenticwf kb` command |

The client needs only httpx. `agenticwf kb` is a subcommand of `agenticwf` (agenticwf-auth-client,
AgenticWF-auth `cli/`), which also gets the tokens:

```bash
pip install ../AgenticWF-auth/cli ./client
agenticwf login && agenticwf kb search "beam energy"     # behind Caddy (AGENTICWF_API_URL, …)
export AGENTICWF_KB_URL=http://127.0.0.1:8766/kb AGENTICWF_KB_TOKEN=$(docker exec agenticwf-kb agenticwf-kb token)
agenticwf kb list                                        # a standalone KB, with its own token
```

| Path | Content |
|---|---|
| [src/agenticwf/kb/api/](src/agenticwf/kb/api) | The KB API, the `agenticwf-kb` command and the workers |
| [src/agenticwf/kb/runtime/](src/agenticwf/kb/runtime) | Ingestion, indexing and retrieval |
| [src/agenticwf/kb/core/](src/agenticwf/kb/core) | Shared code: configuration, ORM and migrations |
| [client/](client) | The `agenticwf-kb-client` package: `KBClient` and `agenticwf kb` |
| [container/](container) | AlmaLinux 9 Dockerfile, supervisord config, scripts, default configuration |
| [deploy/](deploy) | authsvc route rules and the Caddy route |
| [scripts/](scripts) | Package builds and the container smoke test |
| [tests/](tests) | Tests |
| [eval/](eval) | Retrieval evaluation |
