"""
A small Python client for the KB API (docs/api.md), with the HPO warm-start helper in
`kb_client.hpo` (docs/hpo-warm-start.md, Python client).

    kb = KBClient("http://127.0.0.1:8766/kb", api_key=token)        # standalone (the token)
    kb = KBClient("https://host/app/kb", bearer=agenticwf_token)      # authsvc, through Caddy

It speaks the `{success, data}` envelope and raises KBClientError on `success: false`. It has no
dependency on the server packages.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

__all__ = ["KBClient", "KBClientError"]


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
        self._http = http or httpx.Client(timeout=timeout)

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

    def search(self, query: str, knowledge_base_ids: list[str] | None = None, **options) -> list[dict]:
        return self.request("POST", "/knowledge-search", json={"query": query,
                                                               "knowledge_base_ids": knowledge_base_ids, **options})

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
