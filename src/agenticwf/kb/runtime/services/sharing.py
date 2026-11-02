"""
ShareService: requests to share a knowledge base more widely than its owner may (docs/data-model.md,
Sharing requests).

A KB's owner shares it directly with groups they belong to. Wider sharing is a request, approved by:

    visibility `all`                    an admin
    `groups` with `org:<id>` groups     each of those groups' admins (AuthContext.managed_organizations,
                                        from authsvc), or an admin
    any other group (role:, idp:, kb:)  an admin

An approver must have that authority over every group asked for. Approving applies the visibility
to the KB and, with `apply_to_items`, to its documents and sites, in one transaction. A KB has at
most one pending request: a new one replaces it. The requester may cancel a pending request.
"""

from __future__ import annotations

import sqlalchemy as sa

from agenticwf.auth.context import AuthContext
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.kb import KnowledgeBase, KnowledgeShareRequest, utcnow
from agenticwf.kb.runtime import access
from agenticwf.kb.runtime.errors import Conflict, Forbidden, Invalid, NotFound

LOG = get_logger(__name__)

ORG = "org:"


def managed_groups(ctx: AuthContext) -> set[str]:
    """The groups whose sharing requests the principal approves, besides an admin's any."""
    return {f"{ORG}{org_id}" for org_id in getattr(ctx, "managed_organizations", ()) or ()}


def can_approve(ctx: AuthContext, visibility: str, groups) -> bool:
    if access.is_admin(ctx):
        return True
    return visibility == "groups" and bool(groups) and set(groups) <= managed_groups(ctx)


def may_share_directly(ctx: AuthContext, visibility: str, groups) -> bool:
    """What the KB's settings accept from its owner: private, or groups they are in; `all` is for admins."""
    if visibility == "private" or access.is_admin(ctx):
        return True
    return visibility == "groups" and bool(groups) and set(groups) <= set(access.principal_groups(ctx))


def request_dict(r: KnowledgeShareRequest, kb: KnowledgeBase | None = None) -> dict:
    iso = lambda value: value.isoformat() if value else None   # noqa: E731
    return {"id": r.id, "knowledge_base_id": r.knowledge_base_id, "knowledge_base_name": kb.name if kb else None,
            "knowledge_base_ref": f"{kb.owner_subject}:{kb.slug}" if kb else None,
            "current_visibility": kb.visibility if kb else None, "requester": r.owner_subject,
            "visibility": r.visibility, "groups": list(r.groups or []), "apply_to_items": r.apply_to_items,
            "note": r.note, "status": r.status, "decided_by": r.decided_by, "decided_at": iso(r.decided_at),
            "decision_note": r.decision_note, "created_at": iso(r.created_at)}


class ShareService:
    def __init__(self, kbs, knowledge):
        self.kbs = kbs
        self.knowledge = knowledge          # KnowledgeService: reshare_kb_in

    async def create(self, ctx: AuthContext, kb_ref: str, body: dict) -> dict:
        visibility = body.get("visibility")
        if visibility not in ("groups", "all"):
            raise Invalid("a sharing request asks for visibility 'groups' or 'all'")
        groups = sorted({access.normalise_group(g) for g in body.get("groups") or [] if g and str(g).strip()}) \
            if visibility == "groups" else []
        if visibility == "groups" and not groups:
            raise Invalid("visibility 'groups' needs at least one group")
        async with async_session_scope() as session:
            kb = await self.kbs.load(session, ctx, kb_ref, for_update=True)
            if not access.can_manage(ctx, kb):
                raise Forbidden("only the owner asks to share a knowledge base")
            if may_share_directly(ctx, visibility, groups):
                raise Conflict("you may share it this way yourself: change the visibility in its settings")
            pending = (await session.execute(sa.select(KnowledgeShareRequest).where(
                KnowledgeShareRequest.knowledge_base_id == kb.id, KnowledgeShareRequest.status == "pending")
                .with_for_update())).scalars().all()
            for old in pending:             # one pending request per KB: the newest
                old.status, old.decided_by, old.decided_at = "cancelled", ctx.subject, utcnow()
                old.decision_note = "replaced by a newer request"
            request = KnowledgeShareRequest(knowledge_base_id=kb.id, owner_subject=ctx.subject,
                                            owner_org=ctx.organization_id, visibility=visibility, groups=groups,
                                            apply_to_items=bool(body.get("apply_to_items", True)),
                                            note=str(body.get("note") or "")[:2000])
            session.add(request)
            await session.flush()
            LOG.info("share request %s: %s asks %s %s for KB %s", request.id, ctx.subject, visibility, groups, kb.id)
            return request_dict(request, kb)

    async def list(self, ctx: AuthContext, scope: str = "mine", status: str | None = None) -> list[dict]:
        """`mine`: the caller's requests; `approve`: the pending ones the caller may decide."""
        if scope not in ("mine", "approve"):
            raise Invalid("scope must be mine or approve")
        async with async_session_scope(commit=False) as session:
            query = sa.select(KnowledgeShareRequest, KnowledgeBase).join(
                KnowledgeBase, KnowledgeBase.id == KnowledgeShareRequest.knowledge_base_id).where(
                KnowledgeBase.deleted_at.is_(None))
            if scope == "mine":
                query = query.where(KnowledgeShareRequest.owner_subject == ctx.subject)
                if status:
                    query = query.where(KnowledgeShareRequest.status == status)
            else:
                if not (access.is_admin(ctx) or managed_groups(ctx)):
                    return []
                query = query.where(KnowledgeShareRequest.status == "pending")
                if not access.is_admin(ctx):
                    query = query.where(KnowledgeShareRequest.visibility == "groups",
                                        KnowledgeShareRequest.groups.overlap(sorted(managed_groups(ctx))))
            rows = (await session.execute(query.order_by(KnowledgeShareRequest.created_at.desc()).limit(200))).all()
            return [request_dict(r, kb) for r, kb in rows
                    if scope == "mine" or can_approve(ctx, r.visibility, r.groups)]

    async def decide(self, ctx: AuthContext, request_id: str, approve: bool, note: str = "") -> dict:
        async with async_session_scope() as session:
            request = (await session.execute(sa.select(KnowledgeShareRequest).where(
                KnowledgeShareRequest.id == request_id).with_for_update())).scalar_one_or_none()
            if request is None or not (can_approve(ctx, request.visibility, request.groups)
                                       or request.owner_subject == ctx.subject):
                raise NotFound("sharing request not found")
            if not can_approve(ctx, request.visibility, request.groups):
                raise Forbidden("an admin, or the admins of every group asked for, decide this request")
            if request.status != "pending":
                raise Conflict(f"this request is {request.status} already")
            kb = (await session.execute(sa.select(KnowledgeBase).where(
                KnowledgeBase.id == request.knowledge_base_id, KnowledgeBase.deleted_at.is_(None))
                .with_for_update())).scalar_one_or_none()
            if kb is None:
                raise NotFound("the knowledge base is gone")
            request.status = "approved" if approve else "declined"
            request.decided_by, request.decided_at, request.decision_note = ctx.subject, utcnow(), str(note or "")[:2000]
            result = request_dict(request, kb)
            if approve:
                kb.visibility, kb.groups, kb.updated_at = request.visibility, list(request.groups or []), utcnow()
                await session.flush()
                if request.apply_to_items:
                    result["applied_to_items"] = await self.knowledge.reshare_kb_in(session, ctx, kb)
                result["current_visibility"] = kb.visibility
            LOG.info("share request %s %s by %s", request.id, request.status, ctx.subject)
            return result

    async def cancel(self, ctx: AuthContext, request_id: str) -> dict:
        async with async_session_scope() as session:
            request = (await session.execute(sa.select(KnowledgeShareRequest).where(
                KnowledgeShareRequest.id == request_id).with_for_update())).scalar_one_or_none()
            if request is None or request.owner_subject != ctx.subject:
                raise NotFound("sharing request not found")
            if request.status != "pending":
                raise Conflict(f"this request is {request.status} already")
            request.status, request.decided_by, request.decided_at = "cancelled", ctx.subject, utcnow()
            return request_dict(request)
