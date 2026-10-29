"""
Page discovery for site import (docs/ingestion.md, Sites).

    await discover(root, options) -> Discovery(pages, truncated, sitemap_pages, errors)

The site is everything under the root URL: same scheme and host, a path under the root's
directory (`…/en/latest/` for `…/en/latest/` or `…/en/latest/index.html`). Pages come from

* the sitemaps (robots.txt `Sitemap:` lines, else `/sitemap.xml`): entries under the root;
* following links (`follow_links`, on by default) breadth first from the root and the sitemap
  entries, up to `max_depth` hops; Read the Docs' sitemap, for one, lists only the versions.

URLs are normalised (no fragment, no query, `index.html` dropped), robots.txt is respected,
requests are paced (`[ingestion] crawl_interval` seconds apart, 0.5 by default, or robots.txt's
Crawl-delay when longer: CDNs such as Read the Docs' challenge fast crawlers), a 429 or 503 is
retried after its Retry-After (or a backoff), and
only pages are kept: no extension or .html/.htm, nothing under `exclude` (Sphinx's
`_static/`, `_sources/`, `genindex` … by default), and, when `include` is set, only paths that
contain one of its patterns. Every fetch goes through the SSRF-safe fetcher.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.runtime.security.ssrf import USER_AGENT, FetchError, fetch

DEFAULT_MAX_PAGES = 500
MAX_PAGES_LIMIT = 5000
DEFAULT_MAX_DEPTH = 10
CONCURRENCY = 2
RETRIES = 4
MAX_RETRY_WAIT = 60.0
MAX_PAGE_BYTES = 10 * 1024 * 1024
MAX_SITEMAPS = 20
PAGE_EXTENSIONS = ("", ".html", ".htm", ".xhtml")
DEFAULT_EXCLUDE = ("/_sources/", "/_static/", "/_images/", "/_downloads/", "/_modules/", "/genindex",
                   "/py-modindex", "/search.html", "/searchindex")
HTML_TYPES = ("text/html", "application/xhtml+xml")


def crawl_interval() -> float:
    return float(config_get("ingestion", "crawl_interval")) if config_has_option("ingestion", "crawl_interval") \
        else 0.5


class Pace:
    """At most one request start every `interval` seconds."""

    def __init__(self, interval: float):
        self.interval = interval
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            loop = asyncio.get_running_loop()
            delay = self.next - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next = loop.time() + self.interval


@dataclass
class Options:
    max_pages: int = DEFAULT_MAX_PAGES
    max_depth: int = DEFAULT_MAX_DEPTH
    follow_links: bool = True
    use_sitemap: bool = True
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))

    @classmethod
    def from_config(cls, config: dict) -> "Options":
        def strings(value) -> list[str]:
            if isinstance(value, str):
                value = value.split(",")
            return [str(v).strip() for v in value or [] if str(v).strip()]

        return cls(max_pages=max(1, min(int(config.get("max_pages") or DEFAULT_MAX_PAGES), MAX_PAGES_LIMIT)),
                   max_depth=max(0, int(config.get("max_depth", DEFAULT_MAX_DEPTH))),
                   follow_links=bool(config.get("follow_links", True)),
                   use_sitemap=bool(config.get("use_sitemap", True)),
                   include=strings(config.get("include")),
                   exclude=strings(config["exclude"]) if "exclude" in config else list(DEFAULT_EXCLUDE))


@dataclass
class Discovery:
    pages: list[str]                                        # pages that answered with HTML
    truncated: bool = False
    sitemap_pages: int = 0
    unreachable: list[str] = field(default_factory=list)    # linked, but failed for now (not 404/410)
    errors: list[str] = field(default_factory=list)


def normalise(url: str, base: str | None = None) -> str | None:
    """The canonical form of a page URL, or None for what is not http(s)."""
    try:
        parts = urlsplit(urljoin(base, url.strip()) if base else url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if posixpath.basename(path) in ("index.html", "index.htm"):
        path = path[: -len(posixpath.basename(path))]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def scope(root: str) -> tuple[str, str]:
    """(scheme://host, path prefix) of the site under `root`."""
    parts = urlsplit(root)
    path = parts.path or "/"
    if not path.endswith("/"):
        path = posixpath.dirname(path) + "/" if "." in posixpath.basename(path) else path + "/"
    return f"{parts.scheme}://{parts.netloc}", path


class Site:
    def __init__(self, root: str, options: Options):
        self.root = normalise(root)
        if self.root is None:
            raise ValueError("the site URL must be an http or https URL")
        self.origin, self.prefix = scope(self.root)
        self.options = options
        self.robots: RobotFileParser | None = None

    def wanted(self, url: str) -> bool:
        if url == self.root:
            return True
        parts = urlsplit(url)
        if f"{parts.scheme}://{parts.netloc}" != self.origin or not (parts.path + "/").startswith(self.prefix):
            return False
        if posixpath.splitext(parts.path)[1].lower() not in PAGE_EXTENSIONS:
            return False
        if any(pattern in parts.path for pattern in self.options.exclude):
            return False
        if self.options.include and not any(pattern in parts.path for pattern in self.options.include):
            return False
        return self.robots is None or self.robots.can_fetch(USER_AGENT, url)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.base: str | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "base" and attrs.get("href") and self.base is None:
            self.base = attrs["href"]
        elif tag == "a" and attrs.get("href") and "nofollow" not in (attrs.get("rel") or ""):
            self.links.append(attrs["href"])


def links(html: str, page_url: str) -> list[str]:
    parser = _Links()
    try:
        parser.feed(html)
    except Exception:      # a broken page still gives the links found so far
        pass
    base = urljoin(page_url, parser.base) if parser.base else page_url
    return [u for u in (normalise(href, base) for href in parser.links) if u]


def sitemap_locations(body: bytes) -> tuple[list[str], list[str]]:
    """(page URLs, nested sitemap URLs) of a sitemap or sitemap index."""
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return [], []
    pages, nested = [], []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "loc" or not (node.text or "").strip():
            continue
        (nested if root.tag.rsplit("}", 1)[-1] == "sitemapindex" else pages).append(node.text.strip())
    return pages, nested


def _retry_after(page, attempt: int) -> float:
    value = page.retry_after or ""
    return min(MAX_RETRY_WAIT, float(value)) if value.strip().isdigit() else min(MAX_RETRY_WAIT, 2.0 * 2 ** attempt)


async def polite_fetch(url: str, max_bytes: int = MAX_PAGE_BYTES, pace: Pace | None = None):
    """`fetch`, waiting and trying again while the server says 429 (or 503)."""
    for attempt in range(RETRIES + 1):
        if pace is not None:
            await pace.wait()
        page = await fetch(url, max_bytes=max_bytes)
        if page.status not in (429, 503) or attempt == RETRIES:
            return page
        await asyncio.sleep(_retry_after(page, attempt))


async def _get(url: str, pace: Pace, max_bytes: int = MAX_PAGE_BYTES):
    try:
        page = await polite_fetch(url, max_bytes, pace)
    except FetchError:
        return None
    return page if page.status < 400 else None


async def _robots(site: Site, pace: Pace) -> list[str]:
    """Load robots.txt into `site.robots`; returns the sitemaps it names."""
    page = await _get(site.origin + "/robots.txt", pace, 1024 * 1024)
    if page is None:
        return []
    text = page.body.decode("utf-8", "replace")
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    site.robots = parser
    pace.interval = max(pace.interval, float(parser.crawl_delay(USER_AGENT) or 0))
    return [line.split(":", 1)[1].strip() for line in text.splitlines()
            if line.lower().startswith("sitemap:") and line.split(":", 1)[1].strip()]


async def _sitemaps(site: Site, named: list[str], pace: Pace) -> list[str]:
    queue = list(dict.fromkeys(named or [site.origin + "/sitemap.xml"]))
    seen, found = set(), []
    while queue and len(seen) < MAX_SITEMAPS:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        page = await _get(url, pace, 20 * 1024 * 1024)
        if page is None:
            continue
        pages, nested = sitemap_locations(page.body)
        queue.extend(nested)
        found.extend(u for u in (normalise(p) for p in pages) if u and site.wanted(u))
    return list(dict.fromkeys(found))


async def discover(root: str, options: Options | None = None) -> Discovery:
    options = options or Options()
    site = Site(root, options)
    errors: list[str] = []
    pace = Pace(crawl_interval())
    named = await _robots(site, pace)
    if site.robots is not None and not site.robots.can_fetch(USER_AGENT, site.root):
        raise FetchError(f"robots.txt does not allow fetching {site.root}")
    seeds = [site.root]
    sitemap_pages = 0
    if options.use_sitemap:
        from_sitemap = await _sitemaps(site, named, pace)
        sitemap_pages = len(from_sitemap)
        seeds += [u for u in from_sitemap if u != site.root]
    found: dict[str, None] = dict.fromkeys(seeds[: options.max_pages])
    truncated = len(seeds) > options.max_pages
    if not options.follow_links:
        return Discovery(list(found), truncated, sitemap_pages, errors)

    # breadth first, one depth at a time, CONCURRENCY fetches at once
    semaphore = asyncio.Semaphore(CONCURRENCY)
    gone, unreachable = set(), set()

    async def visit(url: str) -> list[str]:
        async with semaphore:
            try:
                page = await polite_fetch(url, pace=pace)
            except FetchError as exc:
                errors.append(str(exc))
                unreachable.add(url)
                return []
        if page.status in (404, 410):
            gone.add(url)
            return []
        if page.status >= 400:
            errors.append(f"{url}: HTTP {page.status}")
            unreachable.add(url)
            return []
        if page.content_type and page.content_type not in HTML_TYPES:
            gone.add(url)
            return []
        final = normalise(page.url) or url
        return links(page.body.decode("utf-8", "replace"), final)

    frontier, visited = list(found), set()
    for depth in range(options.max_depth + 1):
        frontier = [u for u in frontier if u not in visited]
        if not frontier:
            break
        visited.update(frontier)
        results = await asyncio.gather(*(visit(u) for u in frontier))
        if depth == options.max_depth:
            break
        following = []
        for page_links in results:
            for url in page_links:
                if url in found or not site.wanted(url):
                    continue
                if len(found) >= options.max_pages:
                    truncated = True
                    break
                found[url] = None
                following.append(url)
        frontier = following
    if site.root in unreachable or (site.root in gone and sitemap_pages == 0):
        raise FetchError(f"cannot fetch {site.root}: " + (errors[0] if errors else "not found"))
    return Discovery([u for u in found if u not in gone and u not in unreachable], truncated, sitemap_pages,
                     sorted(unreachable), errors[:20])
