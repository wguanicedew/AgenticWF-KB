"""authsvc mode: behind Caddy + authsvc, which sign every request's AuthContext."""

from conftest import SECRET, signed, user


def get(client, path, ctx=None, secret=SECRET, sign_path=None):
    headers = signed(ctx, "GET", sign_path or path.split("?", 1)[0], secret) if ctx else {}
    return client.get(path, headers=headers)


def test_health_is_public(authsvc_client):
    response = authsvc_client.get("/kb/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_me_with_signed_context(authsvc_client):
    ctx = user(org="eic", roles=("researcher",), groups=["atlas"])
    body = get(authsvc_client, "/kb/api/v1/me", ctx).json()
    assert body["success"] is True
    data = body["data"]
    assert data["subject"] == "user_alice" and data["mode"] == "authsvc" and data["approved"] is True
    assert data["groups"] == ["idp:atlas", "org:eic", "role:researcher"]


def test_unsigned_request_is_401_in_the_envelope(authsvc_client):
    response = authsvc_client.get("/kb/api/v1/me")
    assert response.status_code == 401
    assert response.json() == {"success": False, "error": {"code": 1001, "message": "authentication required",
                                                           "details": None}}


def test_forged_or_replayed_context_is_rejected(authsvc_client):
    assert get(authsvc_client, "/kb/api/v1/me", user(), secret=b"x" * 48).status_code == 401
    # signed for another path: no replay against this one
    response = get(authsvc_client, "/kb/api/v1/me", user(), sign_path="/kb/api/v1/system/info")
    assert response.status_code == 401 and response.json()["error"] == "invalid_auth_context"


def test_pending_account_sees_me_but_not_info(authsvc_client):
    pending = user(scopes=())
    me = get(authsvc_client, "/kb/api/v1/me", pending)
    assert me.status_code == 200 and me.json()["data"]["approved"] is False
    # system/info only needs an authenticated caller
    assert get(authsvc_client, "/kb/api/v1/system/info", pending).status_code == 200
    assert authsvc_client.get("/kb/api/v1/system/info").status_code == 401


def test_system_info(authsvc_client):
    data = get(authsvc_client, "/kb/api/v1/system/info", user()).json()["data"]
    assert data["mode"] == "authsvc" and data["base_path"] == "/kb"
    assert data["features"]["faq"] is True and data["features"]["records"] is True     # the test kb.cfg
    assert "vector" in data["extensions"] and "pg_trgm" in data["extensions"]


def test_there_are_no_api_key_routes(authsvc_client):
    response = get(authsvc_client, "/kb/api/v1/api-keys", user(scopes=("app:access", "admin")))
    assert response.status_code == 404 and response.json()["error"]["code"] == 1003


def test_routes_live_under_the_base_path(authsvc_client):
    assert get(authsvc_client, "/api/v1/me", user()).status_code == 404
