# agenticwf-kb-client

The client of the AgenticWF knowledge service: the Python `KBClient` and the `agenticwf kb`
command (with [agenticwf-auth-client](https://pypi.org/project/agenticwf-auth-client/), installed
with it).

```bash
pip install agenticwf-kb-client            # [hpo] adds the Optuna warm-start helper
agenticwf kb list                          # your knowledge bases
agenticwf kb search "beam energy" --kb ID  # hybrid search
agenticwf kb ask "what changed in run 42?" # agentic search: an answer and its sources
agenticwf kb call GET /knowledge-bases     # any KB API call
```

Behind the AgenticWF server (Caddy, Logto) the commands sign in with `agenticwf login`; set the
connection once (an administrator gets the values with
`docker exec agenticwf cat /var/lib/agenticwf/logto/clients.env`):

```bash
export AGENTICWF_AUTH_URL=https://host.example.org
export AGENTICWF_API_URL=https://host.example.org/app
export AGENTICWF_CLIENT_ID=<Logto native app id>
export AGENTICWF_RESOURCE=<API resource indicator>
agenticwf login
```

Scripts and CI use a personal access token instead (`AGENTICWF_TOKEN=pat_…`), or a machine
client (`AGENTICWF_CLIENT_ID` + `AGENTICWF_CLIENT_SECRET`).

A standalone KB has its own token instead:

```bash
export AGENTICWF_KB_URL=http://127.0.0.1:8766/kb
export AGENTICWF_KB_TOKEN=$(docker exec agenticwf-kb agenticwf-kb token)
```

From Python (`KBClient.from_env()` finds the KB and its token as the command does):

```python
from agenticwf.kb.client import KBClient

with KBClient.from_env() as kb:
    for hit in kb.search("beam energy"):
        print(hit["score"], hit["knowledge_title"])
```

HPO warm start with Optuna: `agenticwf.kb.client.hpo` (`warm_start`, `upload_study`).
