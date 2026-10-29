"""Sites: import every page under one URL, refresh it (changed, new and removed pages)."""

import http.server
import threading

import pytest

from conftest import Caller, drain, make_client, user
from agenticwf.kb.runtime.ingestion import crawl

RW = ("app:access", "knowledge:read", "knowledge:write")


def page(title, body, links=()):
    anchors = "".join(f'<a href="{href}">{href}</a> ' for href in links)
    return f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{body}</p>{anchors}</body></html>"


class Docs:
    """A small Sphinx-like site under /docs/latest/; tests change `pages` between crawls."""

    def __init__(self):
        self.pages = {
            "/docs/latest/": page("Home", "Welcome to the beamline manual.",
                                  ["intro.html", "guide/setup.html", "_static/x.html", "genindex.html",
                                   "/other/page.html", "https://example.org/", "intro.html#part", "file.pdf"]),
            "/docs/latest/intro.html": page("Intro", "The magnets are ramped down every night.", ["guide/setup.html"]),
            "/docs/latest/guide/setup.html": page("Setup", "Cool the cryostat before the run.", ["../old.html"]),
            "/docs/latest/old.html": page("Old", "The legacy detector was decommissioned.", []),
            "/docs/latest/_static/x.html": page("Static", "not a page", []),
            "/docs/latest/genindex.html": page("Index", "index", []),
            "/other/page.html": page("Other", "outside the site", []),
        }
        self.status: dict[str, int] = {}          # path -> status to answer (once for 429)
        self.robots = "User-agent: *\nDisallow: /docs/hidden/\nSitemap: {base}/sitemap.xml\n"

    def handler(self):
        docs = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                base = f"http://127.0.0.1:{self.server.server_address[1]}"
                if path == "/robots.txt":
                    return self._send(200, "text/plain", docs.robots.format(base=base))
                if path == "/sitemap.xml":
                    return self._send(200, "application/xml", '<?xml version="1.0"?><urlset xmlns='
                                      '"http://www.sitemaps.org/schemas/sitemap/0.9">'
                                      f"<url><loc>{base}/docs/latest/</loc></url></urlset>")
                status = docs.status.get(path)
                if status == 429:
                    docs.status.pop(path)
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
                if status:
                    return self._send(status, "text/plain", "error")
                if path in docs.pages:
                    return self._send(200, "text/html; charset=utf-8", docs.pages[path])
                self._send(404, "text/plain", "not found")

            def _send(self, status, content_type, text):
                data = text.encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler


@pytest.fixture
def docs(monkeypatch):
    from agenticwf.kb.runtime.services import sites

    monkeypatch.setattr(sites, "PAGE_SPACING", 0)
    monkeypatch.setattr(crawl, "crawl_interval", lambda: 0)
    site = Docs()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), site.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    site.base = f"http://127.0.0.1:{server.server_address[1]}"
    yield site
    server.shutdown()


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def pages(caller, kb, site_id):
    items = caller.get(f"/kb/api/v1/knowledge-bases/{kb}/knowledge?page_size=100&site_id={site_id}").json()["data"]
    return {i["source"].split("/docs/latest/", 1)[1]: i for i in items}


# --- discovery (no database) ----------------------------------------------------------------------
def test_normalise_and_scope():
    assert crawl.normalise("HTTP://Host.org/a//b/index.html?x=1#y") == "http://host.org/a/b/"
    assert crawl.normalise("../c.html", "https://h.org/a/b/") == "https://h.org/a/c.html"
    assert crawl.normalise("mailto:someone@h.org") is None
    assert crawl.scope("https://h.org/en/latest/") == ("https://h.org", "/en/latest/")
    assert crawl.scope("https://h.org/en/latest/index.html") == ("https://h.org", "/en/latest/")
    assert crawl.scope("https://h.org/docs") == ("https://h.org", "/docs/")
    site = crawl.Site("https://h.org/docs", crawl.Options())
    assert site.wanted("https://h.org/docs/a.html") and site.wanted("https://h.org/docs/sub/")
    assert not site.wanted("https://h.org/docs-old/a.html")
    assert not site.wanted("https://h.org/docs/_static/a.html") and not site.wanted("https://h.org/docs/a.png")
    assert not site.wanted("https://other.org/docs/a.html")
    assert crawl.Options.from_config({"max_pages": 10 ** 6}).max_pages == crawl.MAX_PAGES_LIMIT


def test_discover_follows_links_within_the_site(docs):
    import asyncio

    docs.status["/docs/latest/intro.html"] = 429          # retried after Retry-After
    found = asyncio.run(crawl.discover(docs.base + "/docs/latest/index.html"))
    paths = sorted(u[len(docs.base):] for u in found.pages)
    assert paths == ["/docs/latest/", "/docs/latest/guide/setup.html", "/docs/latest/intro.html",
                     "/docs/latest/old.html"]
    assert found.sitemap_pages == 1 and not found.unreachable and not found.truncated
    limited = asyncio.run(crawl.discover(docs.base + "/docs/latest/", crawl.Options(max_pages=2)))
    assert len(limited.pages) == 2 and limited.truncated
    shallow = asyncio.run(crawl.discover(docs.base + "/docs/latest/", crawl.Options(max_depth=1)))
    assert len(shallow.pages) == 3                          # old.html is two links away


# --- the site lifecycle ---------------------------------------------------------------------------
def test_site_import_refresh_and_delete(client, alice, docs):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Manual"}).json()["data"]["id"]
    created = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/sites", json={"url": docs.base + "/docs/latest/"})
    assert created.status_code == 201, created.text
    site = created.json()["data"]
    assert site["status"] == "pending"
    again = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/sites", json={"url": docs.base + "/docs/latest/index.html"})
    assert again.status_code == 409
    drain(client)

    site = alice.get(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert site["status"] == "completed" and site["pages"] == 4, site
    assert site["stats"]["added"] == 4 and site["stats"]["removed"] == 0
    items = pages(alice, kb, site["id"])
    assert set(items) == {"", "intro.html", "guide/setup.html", "old.html"}
    assert all(i["parse_status"] == "completed" and i["site_id"] == site["id"] for i in items.values())
    assert items["intro.html"]["title"] == "Intro"
    hits = alice.post("/kb/api/v1/knowledge-search", json={"query": "magnets ramped down",
                                                          "knowledge_base_ids": [kb]}).json()["data"]
    assert hits and hits[0]["knowledge_title"] == "Intro"

    # the site changes: one page edited, one removed (still linked, but 404), one added
    docs.pages["/docs/latest/intro.html"] = page("Intro", "The magnets now stay on overnight.",
                                                 ["guide/setup.html", "faq.html"])
    del docs.pages["/docs/latest/old.html"]
    docs.pages["/docs/latest/faq.html"] = page("FAQ", "Ask the shift leader.", [])
    refreshed = alice.post(f"/kb/api/v1/sites/{site['id']}/refresh")
    assert refreshed.status_code == 200 and refreshed.json()["data"]["status"] == "pending"
    assert alice.post(f"/kb/api/v1/sites/{site['id']}/refresh").status_code == 409     # queued already
    drain(client)

    site = alice.get(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert site["stats"]["added"] == 1 and site["stats"]["removed"] == 1 and site["stats"]["refreshed"] == 3, site
    after = pages(alice, kb, site["id"])
    assert set(after) == {"", "intro.html", "guide/setup.html", "faq.html"}
    assert after["intro.html"]["version"] == 2                  # changed: re-ingested
    assert after["guide/setup.html"]["version"] == 1            # unchanged: one fetch, no re-embedding
    assert alice.get(f"/kb/api/v1/knowledge/{items['old.html']['id']}").status_code == 404

    # a page that fails for now keeps its item, and nothing is deleted on an incomplete crawl
    docs.status["/docs/latest/intro.html"] = 500
    del docs.pages["/docs/latest/faq.html"]
    alice.post(f"/kb/api/v1/sites/{site['id']}/refresh")
    drain(client)
    site = alice.get(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert site["stats"]["unreachable"] == 1 and site["stats"]["removal_skipped"] and site["stats"]["removed"] == 0
    assert set(pages(alice, kb, site["id"])) == {"", "intro.html", "guide/setup.html", "faq.html"}

    # deleting the site deletes its pages
    assert len(alice.get(f"/kb/api/v1/knowledge-bases/{kb}/sites").json()["data"]) == 1
    deleted = alice.delete(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert deleted["pages_deleted"] == 4
    assert alice.get(f"/kb/api/v1/knowledge-bases/{kb}/knowledge").json()["data"] == []
    assert alice.get(f"/kb/api/v1/sites/{site['id']}").status_code == 404


def test_site_keeps_existing_items_and_can_detach(client, alice, docs):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Manual"}).json()["data"]["id"]
    own = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/url",
                     json={"url": docs.base + "/docs/latest/intro.html"}).json()["data"]
    site = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/sites", json={"url": docs.base + "/docs/latest/"}).json()["data"]
    drain(client)
    site = alice.get(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert site["stats"]["skipped"] == 1 and site["pages"] == 3
    assert alice.get(f"/kb/api/v1/knowledge/{own['id']}").json()["data"]["site_id"] is None

    alice.delete(f"/kb/api/v1/sites/{site['id']}?delete_pages=false")
    items = alice.get(f"/kb/api/v1/knowledge-bases/{kb}/knowledge?page_size=100").json()["data"]
    assert len(items) == 4 and all(i["site_id"] is None for i in items)


def test_failed_crawl_and_access(client, alice, docs):
    kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Manual"}).json()["data"]["id"]
    bob = Caller(client, user("bob", RW, org="eic"))
    assert alice.post(f"/kb/api/v1/knowledge-bases/{kb}/sites", json={"url": "ftp://x"}).status_code == 400
    assert bob.post(f"/kb/api/v1/knowledge-bases/{kb}/sites",
                    json={"url": docs.base + "/docs/latest/"}).status_code in (403, 404)
    site = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/sites",
                      json={"url": docs.base + "/docs/missing/"}).json()["data"]
    drain(client)
    site = alice.get(f"/kb/api/v1/sites/{site['id']}").json()["data"]
    assert site["status"] == "failed" and "cannot fetch" in site["error_message"] and site["pages"] == 0
