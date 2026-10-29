"""
Source citations (docs/ingestion.md, Source citations): every knowledge item has one, so every
search result and every answer can name where its content came from.

    build(supplied, parsed, defaults)   the item's citation: what the caller gave at injection
                                        wins, then the document's own metadata (PDF / Office /
                                        EPUB properties, the web page's title, author, date),
                                        then defaults (title or file name, URL, dates)
    format_text(citation)               "Authors (Year). Title. Publisher. Version. URL"
    locate(citation, section, page)     the citation with where in the document: "…, §Section, p. 3"

Fields: title, authors (list), published (ISO date), publisher, version, url, doi, accessed (ISO
date, for web pages), text (the formatted citation; a caller-supplied text is kept as is).
"""

from __future__ import annotations

import datetime
import re

FIELDS = ("title", "authors", "published", "publisher", "version", "url", "doi", "accessed", "text")
SUPPLIED_KEYS = FIELDS + ("citation_text", "source_url")


def _authors(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = re.split(r"\s*;\s*|\s+and\s+|\s*,\s*(?=[^,]+$)|\n", value) if ";" in value or " and " in value \
            else [value]
    return [str(a).strip()[:200] for a in value if str(a).strip()][:30]


def _date(value) -> str | None:
    """An ISO date (YYYY-MM-DD, or YYYY / YYYY-MM when that is all there is)."""
    if value in (None, ""):
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.date().isoformat() if isinstance(value, datetime.datetime) else value.isoformat()
    text = str(value).strip()
    pdf = re.match(r"^D:(\d{4})(\d{2})?(\d{2})?", text)          # PDF dates: D:20240131120000Z
    if pdf:
        return "-".join(p for p in pdf.groups() if p)
    match = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    return "-".join(p for p in match.groups() if p) if match else None


def normalise_supplied(raw) -> dict:
    """What a caller may give at injection: a `citation` object, or the flat convenience fields
    (`citation_text`, `authors`, `published`, `source_url`, `doi`, `publisher`, `version`)."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("citation must be an object")
    out = {}
    for key in SUPPLIED_KEYS:
        if raw.get(key) in (None, "", []):
            continue
        name = {"citation_text": "text", "source_url": "url"}.get(key, key)
        value = raw[key]
        if name == "authors":
            value = _authors(value)
        elif name in ("published", "accessed"):
            value = _date(value)
            if value is None:
                raise ValueError(f"{key} is not a date")
        else:
            value = str(value).strip()[:2000]
        if value:
            out[name] = value
    return out


def build(supplied: dict | None, parsed: dict | None, defaults: dict) -> dict:
    citation: dict = {}
    for source in (defaults, {k: v for k, v in (parsed or {}).items() if k in FIELDS}, supplied or {}):
        for key, value in source.items():
            if value not in (None, "", []):
                citation[key] = value
    if "authors" in citation:
        citation["authors"] = _authors(citation["authors"])
    for key in ("published", "accessed"):
        if key in citation:
            citation[key] = _date(citation[key])
    if not (supplied or {}).get("text"):
        citation["text"] = format_text(citation)
    return {k: citation[k] for k in FIELDS if citation.get(k) not in (None, "", [])}


def format_text(c: dict) -> str:
    parts = []
    authors = c.get("authors") or []
    if authors:
        names = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        parts.append(names + (f" ({c['published'][:4]})" if c.get("published") else ""))
    elif c.get("published"):
        parts.append(f"({c['published'][:4]})")
    if c.get("title"):
        parts.append(c["title"])
    for key in ("publisher", "version"):
        if c.get(key):
            parts.append(("version " if key == "version" and not str(c[key]).lower().startswith("v") else "")
                         + str(c[key]))
    text = ". ".join(p.rstrip(".") for p in parts)
    if c.get("doi"):
        text += f". doi:{c['doi']}"
    if c.get("url"):
        text += f". {c['url']}"
        if c.get("accessed"):
            text += f" (accessed {c['accessed']})"
    # a final period only after prose: a DOI or URL is copied as it is
    return (text + ".") if text and not text.endswith(".") and not (c.get("url") or c.get("doi")) else text


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def heading_index(markdown: str) -> list[tuple[int, str]]:
    """(offset, "H1 > H2 > …") for every Markdown heading, in order."""
    out, path = [], []
    for match in _HEADING.finditer(markdown or ""):
        level, title = len(match.group(1)), match.group(2).strip()
        path = path[:level - 1] + [title]
        out.append((match.start(), " > ".join(path)))
    return out


def section_at(index: list[tuple[int, str]], start: int, content: str = "") -> str | None:
    """The section a chunk belongs to: the last heading before its first line of body text
    (a chunk that opens with headings belongs to the last of them)."""
    body = start
    for line in content.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        body += len(line)
    current = None
    for offset, path in index:
        if offset > body:
            break
        current = path
    return current


def locate(citation: dict | None, section: str | None = None, page: int | None = None,
           page_end: int | None = None) -> str:
    """The citation text with the place in the document."""
    citation = citation or {}
    text = citation.get("text") or citation.get("title") or ""
    where = []
    if section:
        where.append(f"§ {section.strip()[:200]}")
    if page and page_end and page_end > page:
        where.append(f"pp. {page}–{page_end}")
    elif page:
        where.append(f"p. {page}")
    return f"{text} [{', '.join(where)}]" if where and text else (text or ", ".join(where))
