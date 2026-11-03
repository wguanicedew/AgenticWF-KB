"""
A small Python client for the KB API (docs/api.md), with the HPO warm-start helper in
`agenticwf.kb.client.hpo` (docs/hpo-warm-start.md, Python client).

    kb = KBClient("http://127.0.0.1:8766/kb", api_key=token)        # standalone (the token)
    kb = KBClient("https://host/app/kb", bearer=agenticwf_token)      # authsvc, through Caddy
    kb = KBClient.from_env()            # as `agenticwf kb` does: AGENTICWF_KB_URL is enough

It speaks the `{success, data}` envelope and raises KBClientError on `success: false`. It has no
dependency on the server packages; `from_env` and the `agenticwf kb` command (cli.py) need
agenticwf-auth-client.
"""

from __future__ import annotations

import ssl
from typing import Any, Callable

import certifi
import httpx

__all__ = ["KBClient", "KBClientError"]


def _trust() -> ssl.SSLContext:
    """The system's CAs (SSL_CERT_FILE / SSL_CERT_DIR honoured) and certifi's, as httpx alone trusts
    only certifi's; with agenticwf-auth-client, also CERN's and AGENTICWF_CA_BUNDLE (its tls.py), so
    a CERN host works where the system lacks its CA. Added to the system's, never instead."""
    try:
        from agenticwf_auth_cli import ssl_context
    except ImportError:
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        return context
    return ssl_context()


class KBClientError(Exception):
    def __init__(self, status: int, code: int | None, message: str, details: Any = None):
        super().__init__(f"HTTP {status} ({code}): {message}")
        self.status, self.code, self.message, self.details = status, code, message, details


class KBClient:
    def __init__(self, base_url: str, api_key: str | None = None, bearer: str | None = None, timeout: float = 60.0,
                 http: Any = None, headers: Callable[[str, str], dict] | None = None):
        """`base_url` ends with the KB's base path (…/kb). `http` and `headers` let a caller pass
        its own client and per-request headers (tests, custom auth)."""
        self.base = base_url.rstrip("/")
        token = bearer or api_key
        self._auth = {"Authorization": f"Bearer {token}"} if token else {}
        self._headers = headers
        self._http = http or httpx.Client(timeout=timeout, verify=_trust())

    @classmethod
    def from_env(cls, url: str | None = None, token: str | None = None, **options) -> "KBClient":
        """The KB and its tokens as `agenticwf` finds them (agenticwf_auth_cli.service_access).
        AGENTICWF_KB_URL alone is enough: the KB tells how it signs clients in. Behind Caddy that
        is the Logto sign-in of `agenticwf login`, whose settings the server publishes; a
        standalone KB has its own token: AGENTICWF_KB_TOKEN, or its sign-in link as the URL.
        Without the URL, the KB is at AGENTICWF_API_URL + /kb. Logto tokens are refreshed as they
        expire. `options` are Config fields (auth_url, api_url, …) and the constructor's (timeout, http)."""
        from agenticwf_auth_cli import service_access

        auth = {k: options.pop(k) for k in ("auth_url", "api_url", "client_id", "resource", "scopes") if k in options}
        base, bearer = service_access("kb", "/kb", url=url, token=token, **auth)
        return cls(base, headers=lambda method, url: {"Authorization": f"Bearer {bearer()}"}, **options)

    def close(self) -> None:
        if isinstance(self._http, httpx.Client):
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def request(self, method: str, path: str, **kw) -> Any:
        url = f"{self.base}/api/v1{path}"
        headers = {**self._auth, **(kw.pop("headers", None) or {})}
        if self._headers:
            headers.update(self._headers(method, url))
        response = self._http.request(method, url, headers=headers, **kw)
        try:
            body = response.json()
        except ValueError:
            raise KBClientError(response.status_code, None, response.text[:300])
        if response.status_code >= 400 or not body.get("success", False):
            error = body.get("error") or {}
            raise KBClientError(response.status_code, error.get("code"), error.get("message", ""), error.get("details"))
        return body["data"]

    # -- a few conveniences -------------------------------------------------------------------------
    def knowledge_bases(self) -> list[dict]:
        return self.request("GET", "/knowledge-bases?page_size=100")

    def create_knowledge_base(self, name: str, group: str | None = None, **body) -> dict:
        """A new KB, yours (`u:<username>:<slug>`) or, with `group` (a group you administer: its name
        or id), the group's (`g:<group>:<slug>`). `body`: slug, description, type, visibility, groups,
        contribute, ... (the API's fields)."""
        payload = {"name": name, **body}
        if group:
            payload["group"] = group
        return self.request("POST", "/knowledge-bases", json=payload)

    def search(self, query: str, knowledge_base_ids: list[str] | None = None, **options) -> list[dict]:
        return self.request("POST", "/knowledge-search", json={"query": query,
                                                               "knowledge_base_ids": knowledge_base_ids, **options})

    def add_site(self, kb_id: str, url: str, **options) -> dict:
        """Import every page under `url` (max_pages, max_depth, include, exclude, refresh_interval …)."""
        return self.request("POST", f"/knowledge-bases/{kb_id}/sites", json={"url": url, **options})

    def refresh_site(self, site_id: str) -> dict:
        return self.request("POST", f"/sites/{site_id}/refresh")

    def upsert_records(self, kb_id: str, records: list[dict], record_type: str | None = None, **body) -> dict:
        payload = {"records": records, **body}
        if record_type:
            payload["record_type"] = record_type
        return self.request("POST", f"/knowledge-bases/{kb_id}/records", json=payload)

    def query_records(self, **body) -> dict:
        return self.request("POST", "/records/query", json=body)

    def aggregate_records(self, **body) -> dict:
        return self.request("POST", "/records/aggregate", json=body)

    def suggest(self, **body) -> dict:
        return self.request("POST", "/records/suggest", json=body)

    def agentic_search(self, query: str, **body) -> dict:
        return self.request("POST", "/agentic-search", json={"query": query, **body})
