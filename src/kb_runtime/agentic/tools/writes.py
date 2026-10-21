"""
Write tools (docs/agentic-search.md, Write tools): offered only when the request sets
`allow_writes: true` and the caller has `knowledge:write`. They go through the same services as
the API, under the caller's principal, so the KB's own rules apply (write access to the KB; the
wiki keeps a revision of every page it overwrites, so a write can be reverted). Every write is
listed in the response's `writes`.

    wiki_write_page     create a page, or rewrite one
    wiki_replace_text   replace one exact passage of a page
    wiki_rename_page    change a page's slug (links to it are rewritten)
    wiki_delete_page    delete a page (links to it become plain text)
    wiki_flag_issue     report a problem with a page
    remember            add an episode to the caller's own memory
"""

from __future__ import annotations

from kb_runtime.agentic.tools import Tool, ToolContext, ToolError, list_arg
from kb_runtime.errors import Conflict, Forbidden, Invalid, NotFound

ERRORS = (Invalid, NotFound, Forbidden, Conflict)


def _wiki_kb(tc: ToolContext, args: dict) -> str:
    from kb_runtime.agentic.tools.wiki import wiki_kb_ids

    ids = wiki_kb_ids(tc, list_arg(args, "knowledge_base_ids") or ([args["knowledge_base_id"]]
                                                                     if args.get("knowledge_base_id") else None))
    if len(ids) > 1:
        raise ToolError("several wiki-enabled knowledge bases are in scope: give knowledge_base_id (bN)")
    return ids[0]


def _slug(tc: ToolContext, value) -> str:
    handle = tc.handles.lookup(str(value or ""))
    if handle is not None and handle.prefix == "p":
        return handle.key.partition(":")[2]
    slug = str(value or "").strip()
    if not slug:
        raise ToolError("slug is required")
    return slug


def _record(tc: ToolContext, tool: str, target: str, result: dict) -> None:
    tc.writes.append({"tool": tool, "target": target, **result})


async def wiki_write_page(tc: ToolContext, args: dict) -> str:
    kb_id = _wiki_kb(tc, args)
    wiki = tc.services.wiki
    content = str(args.get("content") or "")
    if not content.strip():
        raise ToolError("content is required")
    try:
        if args.get("slug"):
            slug = _slug(tc, args["slug"])
            try:
                await wiki.get_page(tc.auth, kb_id, slug)
                page = await wiki.update_page(tc.auth, kb_id, slug, {
                    "content": content, **({"title": args["title"]} if args.get("title") else {}),
                    **({"summary": args["summary"]} if args.get("summary") is not None else {})})
                _record(tc, "wiki_write_page", page["slug"], {"action": "updated", "version": page["version"]})
                return f"Updated {page['slug']} (version {page['version']}); the previous version is kept."
            except NotFound:
                pass
        page = await wiki.create_page(tc.auth, kb_id, {
            "title": args.get("title") or "", "content": content, "summary": args.get("summary") or "",
            "page_type": args.get("page_type") or "synthesis", **({"slug": args["slug"]} if args.get("slug") else {})})
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "wiki_write_page", page["slug"], {"action": "created", "version": page["version"]})
    return f"Created {page['slug']}."


async def wiki_replace_text(tc: ToolContext, args: dict) -> str:
    kb_id, slug = _wiki_kb(tc, args), _slug(tc, args.get("slug"))
    old, new = str(args.get("old_text") or ""), str(args.get("new_text") if args.get("new_text") is not None else "")
    if not old:
        raise ToolError("old_text is required")
    try:
        page = await tc.services.wiki.get_page(tc.auth, kb_id, slug)
        count = page["content"].count(old)
        if count != 1:
            raise ToolError(f"old_text occurs {count} times in {slug}; give a passage that occurs exactly once")
        page = await tc.services.wiki.update_page(tc.auth, kb_id, slug, {"content": page["content"].replace(old, new, 1)})
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "wiki_replace_text", slug, {"action": "updated", "version": page["version"]})
    return f"Replaced the passage in {slug} (version {page['version']})."


async def wiki_rename_page(tc: ToolContext, args: dict) -> str:
    kb_id, slug = _wiki_kb(tc, args), _slug(tc, args.get("slug"))
    if not args.get("new_slug"):
        raise ToolError("new_slug is required")
    try:
        page = await tc.services.wiki.update_page(tc.auth, kb_id, slug, {"slug": args["new_slug"],
                                                                          **({"title": args["title"]} if args.get("title") else {})})
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "wiki_rename_page", slug, {"action": "renamed", "new_slug": page["slug"]})
    return f"Renamed {slug} to {page['slug']}; links to it were rewritten."


async def wiki_delete_page(tc: ToolContext, args: dict) -> str:
    kb_id, slug = _wiki_kb(tc, args), _slug(tc, args.get("slug"))
    try:
        await tc.services.wiki.delete_page(tc.auth, kb_id, slug)
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "wiki_delete_page", slug, {"action": "deleted"})
    return f"Deleted {slug}; links to it are now plain text. A revision is kept."


async def wiki_flag_issue(tc: ToolContext, args: dict) -> str:
    kb_id, slug = _wiki_kb(tc, args), _slug(tc, args.get("slug"))
    try:
        issue = await tc.services.wiki.flag_issue(tc.auth, kb_id, {
            "slug": slug, "issue_type": args.get("issue_type") or "other", "description": args.get("description") or ""})
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "wiki_flag_issue", slug, {"action": "flagged", "issue_id": issue["id"]})
    return f"Flagged {issue['issue_type']} on {slug}."


async def remember(tc: ToolContext, args: dict) -> str:
    content = str(args.get("content") or "").strip()
    if not content:
        raise ToolError("content is required")
    try:
        episode = await tc.services.memory.add_episode(tc.auth, {"content": content, "source": "conversation",
                                                                 "group_id": args.get("group_id"),
                                                                 "name": str(args.get("name") or "")[:255]})
    except ERRORS as exc:
        raise ToolError(str(exc))
    _record(tc, "remember", episode["id"], {"action": "remembered", "group_id": episode["group_id"]})
    return "Remembered; facts are extracted from it in the background."


KB = {"type": "string", "description": "bN of the wiki-enabled knowledge base (needed when several are in scope)"}
SLUG = {"type": "string", "description": "page slug or pN"}

TOOLS = [
    Tool("wiki_write_page", "Create a wiki page, or rewrite an existing one (give its slug). Ground every statement in "
         "evidence you found and keep [src:N] markers you were shown. The previous version is kept.",
         {"type": "object", "properties": {"knowledge_base_id": KB, "slug": SLUG, "title": {"type": "string"},
                                           "summary": {"type": "string"}, "content": {"type": "string"},
                                           "page_type": {"type": "string", "enum": ["synthesis", "comparison",
                                                                                     "entity", "concept"]}},
          "required": ["content"]}, wiki_write_page, group="write"),
    Tool("wiki_replace_text", "Replace one passage of a wiki page (old_text must occur exactly once).",
         {"type": "object", "properties": {"knowledge_base_id": KB, "slug": SLUG, "old_text": {"type": "string"},
                                           "new_text": {"type": "string"}}, "required": ["slug", "old_text"]},
         wiki_replace_text, group="write"),
    Tool("wiki_rename_page", "Rename a wiki page; links to it are rewritten.",
         {"type": "object", "properties": {"knowledge_base_id": KB, "slug": SLUG, "new_slug": {"type": "string"},
                                           "title": {"type": "string"}}, "required": ["slug", "new_slug"]},
         wiki_rename_page, group="write"),
    Tool("wiki_delete_page", "Delete a wiki page; links to it become plain text.",
         {"type": "object", "properties": {"knowledge_base_id": KB, "slug": SLUG}, "required": ["slug"]},
         wiki_delete_page, group="write"),
    Tool("wiki_flag_issue", "Report a problem with a wiki page (mixed entities, contradictory facts, out of date, …).",
         {"type": "object", "properties": {"knowledge_base_id": KB, "slug": SLUG,
                                           "issue_type": {"type": "string", "enum": ["mixed_entities", "contradictory_facts",
                                                                                     "out_of_date", "dead_link", "other"]},
                                           "description": {"type": "string"}}, "required": ["slug", "description"]},
         wiki_flag_issue, group="write"),
    Tool("remember", "Add something to the user's own long-term memory (a fact, a decision, a preference).",
         {"type": "object", "properties": {"content": {"type": "string"}, "name": {"type": "string"},
                                           "group_id": {"type": "string"}}, "required": ["content"]},
         remember, group="write"),
]
