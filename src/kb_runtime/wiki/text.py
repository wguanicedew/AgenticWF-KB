"""
Wiki text helpers (docs/llm-wiki.md, Rules), pure functions:

    slugify / page_slug       `<type>/<name>`, lowercase, hyphenated
    Handles                   real slugs ↔ `ref-N` in prompts, restored in the model's output
    links / replace_links     `[[slug|title]]` parsing and rewriting
    strip_dead_links          unknown targets become plain text
    inject_cross_links        the first unlinked mention of a page's title or alias becomes a link
    split_summary             `SUMMARY: …` + body from a page write
"""

from __future__ import annotations

import re
import unicodedata

PAGE_TYPES = ("summary", "entity", "concept", "index", "community", "synthesis", "comparison")
GENERATED_TYPES = ("summary", "entity", "concept")
API_TYPES = ("entity", "concept", "synthesis", "comparison")
LINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]+?))?\]\]")
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def slugify(text: str) -> str:
    """Lowercase, hyphenated; accents folded; CJK characters kept as they are."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w぀-ヿ㐀-鿿가-힯]+", "-", text.lower(), flags=re.UNICODE)
    return re.sub(r"-{2,}", "-", text.replace("_", "-")).strip("-")[:200]


def page_slug(page_type: str, name: str) -> str:
    if page_type not in PAGE_TYPES:
        page_type = "concept"
    name = slugify(name)
    if not name:
        raise ValueError("a page needs a name")
    return f"{page_type}/{name}"


def normalise_slug(slug: str, default_type: str = "concept") -> str:
    """A model- or user-given slug into `<type>/<name>`."""
    slug = str(slug or "").strip().strip("/")
    kind, sep, name = slug.partition("/")
    kind = kind.lower()
    if sep and kind in PAGE_TYPES:
        if kind == "summary":
            return f"summary/{name.lower()}"
        return page_slug(kind, name)
    return page_slug(default_type, slug)


def title_key(title: str) -> str:
    return slugify(title)


class Handles:
    """`ref-N` stand-ins for real slugs in prompts."""

    def __init__(self, slugs=()):
        self.by_ref: dict[str, str] = {}
        self.by_slug: dict[str, str] = {}
        for slug in slugs:
            self.ref(slug)

    def ref(self, slug: str) -> str:
        if slug not in self.by_slug:
            ref = f"ref-{len(self.by_slug) + 1}"
            self.by_slug[slug], self.by_ref[ref] = ref, slug
        return self.by_slug[slug]

    def slug(self, ref_or_slug: str) -> str:
        key = str(ref_or_slug or "").strip()
        return self.by_ref.get(key.lower(), key)

    def restore(self, text: str) -> str:
        """`[[ref-3|Title]]` → `[[entity/x|Title]]`; a ref-N that is not known becomes plain text."""
        def repl(match):
            target, title = match.group(1).strip(), match.group(2)
            if re.fullmatch(r"ref-\d+", target.lower()):
                real = self.by_ref.get(target.lower())
                if real is None:
                    return title or ""
                return f"[[{real}|{title}]]" if title else f"[[{real}]]"
            return match.group(0)
        return LINK.sub(repl, text or "")


def links(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1).strip() for m in LINK.finditer(text or "")))


def replace_links(text: str, mapping: dict[str, str | None]) -> str:
    """Rewrite link targets; a target mapped to None becomes its title as plain text."""
    def repl(match):
        target, title = match.group(1).strip(), match.group(2)
        if target not in mapping:
            return match.group(0)
        new = mapping[target]
        if new is None:
            return title or target.rpartition("/")[2].replace("-", " ")
        return f"[[{new}|{title}]]" if title else f"[[{new}]]"
    return LINK.sub(repl, text or "")


def strip_self_links(text: str, slug: str) -> str:
    return replace_links(text, {slug: None})


def strip_dead_links(text: str, known: set[str]) -> tuple[str, list[str]]:
    dead = [t for t in links(text) if t not in known and not t.startswith("community/")]
    return replace_links(text, {t: None for t in dead}), dead


def inject_cross_links(text: str, slug: str, targets: list[tuple[str, list[str]]], limit: int = 20) -> str:
    """For each (slug, [title, aliases…]) not yet linked from `text`, link its first unlinked
    mention (whole word, case-insensitive). Single-character CJK titles are skipped."""
    linked = set(links(text)) | {slug}
    added = 0
    for target, names in targets:
        if target in linked or added >= limit:
            continue
        for name in sorted({n for n in names if n}, key=len, reverse=True):
            if len(name) < 2 or (len(name) == 1 and _CJK.match(name)):
                continue
            boundary = "" if _CJK.search(name) else r"\b"
            pattern = re.compile(f"{boundary}{re.escape(name)}{boundary}", re.IGNORECASE)
            position = _first_outside_links(text, pattern)
            if position is None:
                continue
            start, end = position
            text = f"{text[:start]}[[{target}|{text[start:end]}]]{text[end:]}"
            linked.add(target)
            added += 1
            break
    return text


def _first_outside_links(text: str, pattern: re.Pattern) -> tuple[int, int] | None:
    spans = [m.span() for m in LINK.finditer(text)]
    headings = [m.span() for m in re.finditer(r"^#.*$", text, re.MULTILINE)]
    for match in pattern.finditer(text):
        s, e = match.span()
        if any(a <= s < b for a, b in spans + headings):
            continue
        return s, e
    return None


def split_summary(output: str) -> tuple[str, str]:
    """`SUMMARY: one paragraph` then the markdown body."""
    text = (output or "").strip()
    match = re.match(r"^\s*SUMMARY:\s*(.+?)(?:\n\s*\n|\n(?=#)|$)(.*)$", text, re.DOTALL)
    if not match:
        return "", text
    return match.group(1).strip(), match.group(2).strip()


def chunk_label(index: int) -> str:
    return f"c{index + 1:03d}"
