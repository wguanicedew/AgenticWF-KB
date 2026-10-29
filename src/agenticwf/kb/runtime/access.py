"""
Who may read and change what (docs/data-model.md, Ownership and visibility).

Two levels: the knowledge base is the room (who can see it at all), and each item has its own
visibility inside it. An item is readable when its KB is readable and the item's visibility
allows it. Ownership is the user (`subject`), not the token.

    KB / model visible:  owner, or visibility = all, or visibility = groups and a group matches
                         (admin sees every KB and model, to manage them)
    item readable:       KB readable, and (item owner, or item visibility = all, or
                         visibility = groups and a group matches). Admin is not special here:
                         nobody reads someone else's private item

The same predicates apply to kb_knowledges and kb_embeddings (which carry copies of the item's
owner, visibility and groups), so the filters run inside the vector and BM25 queries.
"""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import and_, false, or_, true

from agenticwf.auth.context import AuthContext

GROUP_PREFIXES = ("org:", "role:", "idp:", "kb:")
VISIBILITIES = ("private", "groups", "all")
SCOPE_ADMIN = "admin"
SYSTEM_OWNER = "system"


def normalise_group(name: str) -> str:
    """A group as stored: bare names (groups a KB defines itself) become `kb:<name>`."""
    name = name.strip()
    return name if name.startswith(GROUP_PREFIXES) else f"kb:{name}"


def principal_groups(ctx: AuthContext) -> list[str]:
    """The principal's groups, as prefixed strings so sources cannot collide:

    `org:<id>` per organization the user belongs to (authsvc's groups, AuthContext.organizations)
    and the token's own organization, `role:<name>` per role, `idp:<name>` per IdP group
    (`external_identity.groups`, e.g. INDIGO IAM). Names already in one of these forms are kept.
    In standalone mode the one local user has no groups.
    """
    groups: set[str] = set()
    if ctx.organization_id:
        groups.add(f"org:{ctx.organization_id}")
    groups.update(f"org:{org_id}" for org_id in getattr(ctx, "organizations", None) or ())
    groups.update(f"role:{role}" for role in ctx.roles)
    external = ctx.external_identity or {}
    for name in external.get("groups") or ():
        if isinstance(name, str) and name.strip():
            name = name.strip()
            groups.add(name if name.startswith(GROUP_PREFIXES) else f"idp:{name}")
    return sorted(groups)


def group_names(ctx: AuthContext) -> dict[str, str]:
    """Display names of the caller's groups that have one (organizations, from authsvc)."""
    return {f"org:{org_id}": name for org_id, name in (getattr(ctx, "organizations", None) or {}).items()}


def is_admin(ctx: AuthContext) -> bool:
    return SCOPE_ADMIN in ctx.scopes


def _visible(model, ctx: AuthContext):
    groups = principal_groups(ctx)
    clauses = [model.owner_subject == ctx.subject, model.visibility == "all"]
    if groups:
        clauses.append(and_(model.visibility == "groups", model.groups.overlap(groups)))
    return or_(*clauses)


def visible_clause(model, ctx: AuthContext):
    """WHERE clause: the KBs (or models) the principal can see."""
    return true() if is_admin(ctx) else _visible(model, ctx)


def items_clause(model, ctx: AuthContext, kb_ids: Iterable[str]):
    """WHERE clause on kb_knowledges or kb_embeddings: items of `kb_ids` (already readable KBs)
    whose own visibility lets the principal read them."""
    kb_ids = list(kb_ids)
    if not kb_ids:
        return false()
    return and_(model.knowledge_base_id.in_(kb_ids), _visible(model, ctx))


def can_see(ctx: AuthContext, row) -> bool:
    """The Python form of `visible_clause` for one loaded KB or model row."""
    if is_admin(ctx) or row.owner_subject == ctx.subject or row.visibility == "all":
        return True
    return row.visibility == "groups" and bool(set(row.groups or ()) & set(principal_groups(ctx)))


def can_read_item(ctx: AuthContext, item) -> bool:
    """For an item whose KB is readable."""
    if item.owner_subject == ctx.subject or item.visibility == "all":
        return True
    return item.visibility == "groups" and bool(set(item.groups or ()) & set(principal_groups(ctx)))


def can_manage(ctx: AuthContext, row) -> bool:
    """Change a KB's or model's settings, delete it: the owner or an admin."""
    return is_admin(ctx) or row.owner_subject == ctx.subject


def can_contribute(ctx: AuthContext, kb) -> bool:
    """Add items: the KB owner, an admin, or anyone who can see it when `contribute = readers`."""
    return can_manage(ctx, kb) or (kb.contribute == "readers" and can_see(ctx, kb))


def can_manage_item(ctx: AuthContext, item, kb) -> bool:
    """Edit, replace, delete an item or change its visibility: its owner, the KB owner, an admin."""
    return is_admin(ctx) or item.owner_subject == ctx.subject or kb.owner_subject == ctx.subject


def check_share(ctx: AuthContext, visibility: str, groups: Iterable[str]) -> list[str]:
    """Validate a visibility and the groups an item or KB is shared with; returns the stored
    groups. Only groups the principal belongs to (an admin: any)."""
    from agenticwf.kb.runtime.errors import Forbidden, Invalid

    if visibility not in VISIBILITIES:
        raise Invalid(f"visibility must be one of {', '.join(VISIBILITIES)}")
    stored = sorted({normalise_group(g) for g in groups if g and g.strip()})
    if visibility != "groups":
        return []
    if not stored:
        raise Invalid("visibility 'groups' needs at least one group")
    if not is_admin(ctx):
        foreign = set(stored) - set(principal_groups(ctx))
        if foreign:
            raise Forbidden(f"you are not in the groups: {', '.join(sorted(foreign))}")
    return stored
