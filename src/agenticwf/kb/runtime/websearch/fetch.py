"""
Web pages as markdown (docs/web-search.md, Fetching).

    fetch_page(url)     SSRF-safe GET (agenticwf.kb.runtime.security.ssrf: public addresses only, every
                        redirect re-checked, DNS pinned) → markdown through the ingestion parsers
                        (HTML, PDF, DOCX, XLSX, PPTX, EPUB, text, CSV, JSON)
    FetchCache          one per run (or request): a page is fetched once, so reading it in pages
                        (offset / limit) sees one version
    window(text, …)     a page of the text
"""

from __future__ import annotations

import asyncio
import mimetypes
from collections import OrderedDict
from dataclasses import dataclass, field
from urllib.parse import urlparse

import anyio

from agenticwf.kb.runtime.ingestion import parsers
from agenticwf.kb.runtime.security.ssrf import FetchError, fetch

MAX_BYTES = 10 * 1024 * 1024
MAX_LIMIT = 20_000
CONTENT_EXTENSIONS = {
    "text/html": ".html", "application/xhtml+xml": ".html", "application/pdf": ".pdf", "text/plain": ".txt",
    "text/markdown": ".md", "text/csv": ".csv", "application/json": ".json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/epub+zip": ".epub",
}


class PageError(RuntimeError):
    pass


@dataclass
class Page:
    url: str               # after redirects
    title: str
    markdown: str
    content_type: str


async def fetch_page(url: str, max_bytes: int = MAX_BYTES, timeout: float = 30.0) -> Page:
    try:
        got = await fetch(url, max_bytes=max_bytes, timeout=timeout)
    except FetchError as exc:
        raise PageError(str(exc)) from exc
    if got.status >= 400:
        raise PageError(f"{url}: HTTP {got.status}")
    ext = CONTENT_EXTENSIONS.get(got.content_type) or parsers.extension(urlparse(got.url).path) \
        or mimetypes.guess_extension(got.content_type or "") or ".html"
    if not parsers.supported("page" + ext):
        raise PageError(f"{url}: {got.content_type or 'this type'} cannot be read as text")
    try:
        result = await anyio.to_thread.run_sync(parsers.parse, got.body, "page" + ext)
    except parsers.ParseError as exc:
        raise PageError(f"{url}: {exc}") from exc
    return Page(got.url, str(result.metadata.get("title") or ""), result.markdown, got.content_type)


@dataclass
class FetchCache:
    """Per run: the last `size` pages, so paged reads see one version of a page."""

    size: int = 8
    pages: OrderedDict = field(default_factory=OrderedDict)
    _locks: dict = field(default_factory=dict)

    async def get(self, url: str) -> Page:
        lock = self._locks.setdefault(url, asyncio.Lock())
        async with lock:
            if url in self.pages:
                self.pages.move_to_end(url)
                return self.pages[url]
            page = await fetch_page(url)
            self.pages[url] = page
            while len(self.pages) > self.size:
                self.pages.popitem(last=False)
            return page


def window(text: str, offset: int = 0, limit: int = 8000) -> dict:
    offset = max(0, int(offset or 0))
    limit = max(200, min(int(limit or 8000), MAX_LIMIT))
    part = text[offset:offset + limit]
    end = offset + len(part)
    return {"offset": offset, "returned_chars": len(part), "content_length": len(text),
            "next_offset": end if end < len(text) else None, "content": part}
