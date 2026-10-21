"""
Exact text search over chunk content (docs/agentic-search.md, grep_knowledge implementation).

    readable KBs → SQL on kb_chunks (pg_trgm index): `~*` / `~` for a regex, strpos for a
    literal; 5 s statement timeout → the matching lines of each chunk with ±1 line, ≤ 300 chars

The freshness filters are search's: chunk enabled, item live, enabled, inside its validity
window, deprecated only when the KB down-ranks. An invalid regex falls back to a literal match.
"""

from __future__ import annotations

import re

import sqlalchemy as sa

from authsvc.context import AuthContext

from kb_core.orm.base.session import async_session_scope
from kb_runtime import access
from kb_runtime.citation import locate
from kb_runtime.errors import Invalid
from kb_runtime.services.knowledge_bases import KnowledgeBaseService

MAX_PATTERN = 500
MAX_LIMIT = 50
LINE_CHARS = 300

_SQL = """
SELECT c.id, c.seq_id, c.chunk_index, c.chunk_type, c.content, k.id AS knowledge_id, k.title,
       k.knowledge_base_id, k.deprecated_at IS NOT NULL AS deprecated, k.citation, c.metadata AS chunk_meta,
       c.context_header
FROM kb_chunks c JOIN kb_knowledges k ON k.id = c.knowledge_id
WHERE k.knowledge_base_id = ANY(:kbs) AND c.is_enabled AND k.deleted_at IS NULL AND k.enable_status = 'enabled'
  AND k.parse_status = 'completed'
  AND (k.owner_subject = :subject OR k.visibility = 'all' OR (k.visibility = 'groups' AND k.groups && CAST(:groups AS text[])))
  AND (k.valid_from IS NULL OR k.valid_from <= now()) AND (k.valid_until IS NULL OR k.valid_until > now())
  AND (k.expires_at IS NULL OR k.expires_at > now())
  AND (k.deprecated_at IS NULL OR k.knowledge_base_id = ANY(:downrank_kbs))
  AND {match} {knowledge_filter}
ORDER BY k.title, k.id, c.chunk_index
LIMIT :limit
"""


def _matcher(pattern: str, regex: bool, case_sensitive: bool) -> tuple[re.Pattern, bool]:
    """(compiled Python pattern, regex mode) — literal when the regex does not compile."""
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        try:
            return re.compile(pattern, flags), True
        except re.error:
            pass
    return re.compile(re.escape(pattern), flags), False


def matching_lines(content: str, matcher: re.Pattern) -> list[dict]:
    """The lines of `content` that match, each with one line before and after."""
    lines = content.splitlines() or [content]
    out, used = [], set()
    for number, line in enumerate(lines):
        if not matcher.search(line) or number in used:
            continue
        window = range(max(0, number - 1), min(len(lines), number + 2))
        used.update(window)
        out.append({"line": number + 1,
                    "text": "\n".join(lines[i][:LINE_CHARS] for i in window).strip("\n")})
    return out


async def grep(ctx: AuthContext, kbs: KnowledgeBaseService, pattern: str, *, kb_refs: list[str] | None = None,
               knowledge_ids: list[str] | None = None, tag_ids: list[str] | None = None, regex: bool = True,
               case_sensitive: bool = False, limit: int = 20) -> dict:
    pattern = pattern or ""
    if not pattern.strip():
        raise Invalid("pattern is required")
    if len(pattern) > MAX_PATTERN:
        raise Invalid(f"pattern is longer than {MAX_PATTERN} characters")
    limit = max(1, min(int(limit or 20), MAX_LIMIT))
    matcher, regex_mode = _matcher(pattern, regex, case_sensitive)
    if regex_mode:
        match = "c.content ~ :p" if case_sensitive else "c.content ~* :p"
    else:
        match = "strpos(c.content, :p) > 0" if case_sensitive else "strpos(lower(c.content), lower(:p)) > 0"
    async with async_session_scope(commit=False) as session:
        kb_ids = await kbs.readable_ids(session, ctx, kb_refs)
        if not kb_ids:
            return {"pattern": pattern, "mode": "regex" if regex_mode else "literal", "matches": [],
                    "limit_reached": False}
        rows_kb = (await session.execute(sa.text(
            "SELECT id, freshness_config FROM kb_knowledge_bases WHERE id = ANY(:ids)"), {"ids": kb_ids})).all()
        tag_filter, tags = "", ([], False)
        if tag_ids:
            from kb_runtime.services.tags import resolve_tags

            tags = await resolve_tags(session, kb_ids, tag_ids)
            if not tags[0] and not tags[1]:
                return {"pattern": pattern, "mode": "regex" if regex_mode else "literal", "matches": [],
                        "limit_reached": False}
            tag_filter = ("AND (c.tag_id = ANY(:tag_ids) OR k.id IN (SELECT knowledge_id FROM kb_knowledge_tag_relations "
                          "WHERE tag_id = ANY(:tag_ids))" + (" OR (c.tag_id = '' AND NOT EXISTS (SELECT 1 FROM "
                          "kb_knowledge_tag_relations r WHERE r.knowledge_id = k.id))" if tags[1] else "") + ")")
        params = {"kbs": kb_ids, "tag_ids": tags[0], "subject": ctx.subject, "groups": access.principal_groups(ctx), "p": pattern,
                  "downrank_kbs": [i for i, cfg in rows_kb if (cfg or {}).get("deprecated_mode") == "downrank"],
                  "knowledge_ids": list(knowledge_ids or []), "limit": limit}
        sql = _SQL.format(match=match, knowledge_filter=("AND k.id = ANY(:knowledge_ids) " if knowledge_ids else "")
                          + tag_filter)
        await session.execute(sa.text("SET LOCAL statement_timeout = '5s'"))
        try:
            rows = (await session.execute(sa.text(sql), params)).mappings().all()
        except sa.exc.DBAPIError as exc:
            text = str(exc.orig if hasattr(exc, "orig") else exc).lower()
            if "statement timeout" in text or "canceling statement" in text:
                raise Invalid("the pattern took too long; use a more specific pattern") from None
            if regex_mode and "regular expression" in text:       # PostgreSQL's dialect rejected it
                return await grep(ctx, kbs, pattern, kb_refs=kb_refs, knowledge_ids=knowledge_ids, tag_ids=tag_ids,
                                  regex=False,
                                  case_sensitive=case_sensitive, limit=limit)
            raise
    matches = []
    for row in rows:
        lines = matching_lines(row["content"], matcher)
        if not lines:              # PostgreSQL and Python regex dialects can differ; keep the chunk visible
            lines = [{"line": 1, "text": row["content"][:LINE_CHARS]}]
        matches.append({"chunk_id": row["id"], "seq": row["seq_id"], "chunk_index": row["chunk_index"],
                        "chunk_type": row["chunk_type"], "knowledge_id": row["knowledge_id"],
                        "knowledge_title": row["title"], "knowledge_base_id": row["knowledge_base_id"],
                        "deprecated": bool(row["deprecated"]), "lines": lines,
                        "citation_text": locate(row["citation"] or {"text": row["title"]},
                                                row["context_header"] or (row["chunk_meta"] or {}).get("section"),
                                                (row["chunk_meta"] or {}).get("page"),
                                                (row["chunk_meta"] or {}).get("page_end"))})
    return {"pattern": pattern, "mode": "regex" if regex_mode else "literal", "matches": matches,
            "limit_reached": len(rows) >= limit}
