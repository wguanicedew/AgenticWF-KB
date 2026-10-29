"""deploy/policy.kb.yaml, evaluated by authsvc's own policy engine."""

import re
from pathlib import Path

import pytest

from conftest import user

POLICY = Path(__file__).resolve().parents[1] / "deploy" / "policy.kb.yaml"


@pytest.fixture(scope="module")
def policy():
    from agenticwf.auth.authz import Policy

    return Policy.load(POLICY)


def decide(policy, ctx, method, path):
    return policy.authorize(ctx, method, path)


def test_public_and_pending_routes(policy):
    assert decide(policy, None, "GET", "/kb/health").allowed
    pending = user(scopes=())
    assert decide(policy, pending, "GET", "/kb/api/v1/me").allowed
    assert decide(policy, pending, "GET", "/kb/api/v1/system/info").code == "approval_pending"
    assert decide(policy, user(), "GET", "/kb/").allowed and decide(policy, user(), "GET", "/kb").allowed


def test_reads_writes_and_admin(policy):
    reader = user(scopes=("app:access", "knowledge:read"))
    writer = user(scopes=("app:access", "knowledge:read", "knowledge:write"))
    assert decide(policy, reader, "POST", "/kb/api/v1/knowledge-search").allowed      # read-only POST
    assert decide(policy, reader, "POST", "/kb/api/v1/agentic-search").allowed
    assert not decide(policy, reader, "POST", "/kb/api/v1/knowledge-bases/k1/knowledge/file").allowed
    assert decide(policy, writer, "POST", "/kb/api/v1/knowledge-bases/k1/knowledge/file").allowed
    assert not decide(policy, writer, "POST", "/kb/api/v1/system/housekeeping").allowed
    admin = user(scopes=("app:access", "admin"))
    assert decide(policy, admin, "POST", "/kb/api/v1/system/housekeeping").allowed


def test_people_only_kb_management(policy):
    machine = user(subject="app_ci", client_id="app_ci",
                   scopes=("app:access", "knowledge:read", "knowledge:write"))
    assert not decide(policy, machine, "POST", "/kb/api/v1/knowledge-bases").allowed
    assert decide(policy, machine, "POST", "/kb/api/v1/knowledge-bases/k1/knowledge/url").allowed   # pipelines


def test_every_authsvc_route_has_a_rule(policy):
    from agenticwf.kb.api.app import create_app
    from agenticwf.kb.api.settings import Settings

    app = create_app(Settings(mode="authsvc", context_secret=b"s" * 48))
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/kb"):
            continue
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        for method in getattr(route, "methods", None) or ():
            assert policy.match(method, concrete) is not None, f"no policy rule for {method} {path}"
