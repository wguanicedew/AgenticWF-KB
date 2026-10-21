"""
Web search providers (docs/web-search.md). Each turns a query into `WebResult`s.

    duckduckgo   no key; the HTML endpoint (html.duckduckgo.com), parsed
    searxng      a SearXNG instance's JSON API (`base_url`), no key
    tavily       api.tavily.com (`api_key`); returns page text with the results
    brave        api.search.brave.com (`api_key`)
    bing         api.bing.microsoft.com v7 (`api_key`)
    google       Google Programmable Search, customsearch v1 (`api_key`, `cx`)
    exa          api.exa.ai (`api_key`); returns page text with the results
    bocha        api.bochaai.com (`api_key`); returns summaries

`transport` lets tests mock HTTP. Results are untrusted data.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; agenticwf-kb/0.1; +https://github.com/InnovateAILab)"


class WebSearchError(RuntimeError):
    pass


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str = ""
    content: str = ""            # page text, from the provider or from fetching the page
    published: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "content": self.content,
                "published": self.published, "provider": self.provider}


@dataclass(frozen=True)
class Kind:
    name: str
    title: str
    needs_key: bool
    params: tuple[str, ...] = ()            # other required parameters
    default_url: str = ""
    returns_content: bool = False


KINDS = {k.name: k for k in (
    Kind("duckduckgo", "DuckDuckGo", False, default_url="https://html.duckduckgo.com"),
    Kind("searxng", "SearXNG", False, ("base_url",)),
    Kind("tavily", "Tavily", True, default_url="https://api.tavily.com", returns_content=True),
    Kind("brave", "Brave Search", True, default_url="https://api.search.brave.com"),
    Kind("bing", "Bing Web Search", True, default_url="https://api.bing.microsoft.com"),
    Kind("google", "Google Programmable Search", True, ("cx",), default_url="https://www.googleapis.com"),
    Kind("exa", "Exa", True, default_url="https://api.exa.ai", returns_content=True),
    Kind("bocha", "Bocha", True, default_url="https://api.bochaai.com", returns_content=True),
)}


def _strip(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _json(response: httpx.Response, name: str) -> Any:
    if response.status_code in (401, 403):
        raise WebSearchError(f"{name}: the API key was refused (HTTP {response.status_code})")
    if response.status_code == 429:
        raise WebSearchError(f"{name}: rate limited (HTTP 429)")
    if response.status_code >= 400:
        raise WebSearchError(f"{name}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise WebSearchError(f"{name}: the answer was not JSON")


@dataclass
class Provider:
    kind: str
    params: dict = field(default_factory=dict)       # base_url, api_key, cx, …
    timeout: float = 15.0
    transport: httpx.AsyncBaseTransport | None = None

    @property
    def base(self) -> str:
        return (self.params.get("base_url") or KINDS[self.kind].default_url).rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=True,
                                 headers={"User-Agent": USER_AGENT})

    def _key(self) -> str:
        key = self.params.get("api_key") or ""
        if KINDS[self.kind].needs_key and not key:
            raise WebSearchError(f"{self.kind} needs an API key")
        return key

    async def search(self, query: str, n: int = 5) -> list[WebResult]:
        n = max(1, min(int(n), 20))
        handler = getattr(self, f"_{self.kind}", None)
        if handler is None:
            raise WebSearchError(f"unknown provider kind {self.kind!r}")
        async with self._client() as client:
            results = await handler(client, query, n)
        for r in results:
            r.provider = self.kind
            r.title, r.snippet = _strip(r.title)[:300], _strip(r.snippet)[:1500]
        return [r for r in results if r.url.startswith(("http://", "https://"))][:n]

    # -- the providers ----------------------------------------------------------------------------------
    async def _duckduckgo(self, client, query, n):
        response = await client.post(f"{self.base}/html/", data={"q": query, "kl": self.params.get("region", "")})
        if response.status_code >= 400:
            raise WebSearchError(f"duckduckgo: HTTP {response.status_code}")
        return parse_duckduckgo(response.text)[:n]

    async def _searxng(self, client, query, n):
        response = await client.get(f"{self.base}/search", params={"q": query, "format": "json", "safesearch": 1})
        data = _json(response, "searxng")
        return [WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""), published=r.get("publishedDate") or "")
                for r in data.get("results", [])[:n]]

    async def _tavily(self, client, query, n):
        response = await client.post(f"{self.base}/search", json={
            "query": query, "max_results": n, "search_depth": self.params.get("search_depth", "basic"),
            "include_raw_content": True}, headers={"Authorization": f"Bearer {self._key()}"})
        data = _json(response, "tavily")
        return [WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""), content=r.get("raw_content") or "",
                          published=r.get("published_date") or "") for r in data.get("results", [])[:n]]

    async def _brave(self, client, query, n):
        response = await client.get(f"{self.base}/res/v1/web/search", params={"q": query, "count": n},
                                    headers={"X-Subscription-Token": self._key(), "Accept": "application/json"})
        data = _json(response, "brave")
        return [WebResult(r.get("title", ""), r.get("url", ""), r.get("description", ""), published=r.get("age") or "")
                for r in (data.get("web") or {}).get("results", [])[:n]]

    async def _bing(self, client, query, n):
        response = await client.get(f"{self.base}/v7.0/search", params={"q": query, "count": n, "textDecorations": False},
                                    headers={"Ocp-Apim-Subscription-Key": self._key()})
        data = _json(response, "bing")
        return [WebResult(r.get("name", ""), r.get("url", ""), r.get("snippet", ""),
                          published=r.get("datePublished") or r.get("dateLastCrawled") or "")
                for r in (data.get("webPages") or {}).get("value", [])[:n]]

    async def _google(self, client, query, n):
        if not self.params.get("cx"):
            raise WebSearchError("google needs cx (the Programmable Search Engine id)")
        response = await client.get(f"{self.base}/customsearch/v1", params={
            "key": self._key(), "cx": self.params["cx"], "q": query, "num": min(n, 10)})
        data = _json(response, "google")
        return [WebResult(r.get("title", ""), r.get("link", ""), r.get("snippet", "")) for r in data.get("items", [])[:n]]

    async def _exa(self, client, query, n):
        response = await client.post(f"{self.base}/search", json={
            "query": query, "numResults": n, "contents": {"text": {"maxCharacters": 4000}}},
            headers={"x-api-key": self._key()})
        data = _json(response, "exa")
        return [WebResult(r.get("title") or "", r.get("url", ""), (r.get("text") or "")[:300], content=r.get("text") or "",
                          published=r.get("publishedDate") or "") for r in data.get("results", [])[:n]]

    async def _bocha(self, client, query, n):
        response = await client.post(f"{self.base}/v1/web-search", json={"query": query, "count": n, "summary": True},
                                     headers={"Authorization": f"Bearer {self._key()}"})
        data = _json(response, "bocha")
        pages = ((data.get("data") or {}).get("webPages") or {}).get("value", [])
        return [WebResult(r.get("name", ""), r.get("url", ""), r.get("snippet", ""), content=r.get("summary") or "",
                          published=r.get("datePublished") or "") for r in pages[:n]]


# --- DuckDuckGo's HTML ---------------------------------------------------------------------------------
_DDG_RESULT = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)'
                         r'(?=<a[^>]+class="[^"]*result__a|$)', re.S)
_DDG_SNIPPET = re.compile(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|td)>', re.S)


def ddg_target(href: str) -> str:
    """DuckDuckGo wraps results in a redirect (`//duckduckgo.com/l/?uddg=<url>`): the real URL."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def parse_duckduckgo(page: str) -> list[WebResult]:
    out = []
    for href, title, rest in _DDG_RESULT.findall(page or ""):
        url = ddg_target(href)
        if "duckduckgo.com/y.js" in url:          # ads
            continue
        snippet = _DDG_SNIPPET.search(rest)
        out.append(WebResult(title, url, snippet.group(1) if snippet else ""))
    return out


# --- results ------------------------------------------------------------------------------------------
_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_cid|mc_eid)")


def canonical_url(url: str) -> str:
    """For de-duplication: scheme and host lowercased, no fragment, no tracking parameters, no
    trailing slash, `www.` dropped."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    query = "&".join(sorted(p for p in parsed.query.split("&") if p and not _TRACKING.match(p)))
    path = parsed.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{query}" if query else "")


def dedupe(results: list[WebResult]) -> list[WebResult]:
    """One result per page (the first seen, keeping any content another provider returned)."""
    seen: dict[str, WebResult] = {}
    for r in results:
        key = canonical_url(r.url)
        if key in seen:
            kept = seen[key]
            kept.content = kept.content or r.content
            kept.snippet = kept.snippet or r.snippet
            kept.published = kept.published or r.published
            continue
        seen[key] = r
    return list(seen.values())
